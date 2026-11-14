import argparse
import logging
import random
import statistics
import time
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, SimpleConv, global_add_pool

from constants import EXPERIMENT_ENTRIES_DIR
from datasets.dataset_handler import DatasetHandler
from datasets.enums import DatasetSource, FeatureType
from experiment_helper import ExperimentHelper

# feature types the Siamese network can embed: graph feature types are encoded with
# the GCN encoder, every other (fixed-length vector) feature type with the MLP encoder
GRAPH_FEATURE_TYPES = [FeatureType.PYG_WITH_BONDS]
VECTOR_FEATURE_TYPES = [FeatureType.ECFP, FeatureType.CHEMBERTA, FeatureType.MOLFORMER]
SUPPORTED_FEATURE_TYPES = GRAPH_FEATURE_TYPES + VECTOR_FEATURE_TYPES


class RoundEntry(TypedDict):
    assay_target: str
    support_positive_count: int
    support_negative_count: int
    round_no: int
    seed: int
    query_positive_count: int
    query_negative_count: int
    pair_count: int
    positive_pair_count: int
    accuracy: float
    roc: float
    prc: float
    duration_s: float


class SummaryRow(TypedDict):
    assay_target: str
    support_positive_count: int
    support_negative_count: int
    total_positive_count: int
    total_negative_count: int
    accuracy: str
    roc: str
    prc: str
    duration_s: float


class TrainingRow(TypedDict):
    support_positive_count: int
    support_negative_count: int
    train_tasks: int
    episodes_run: int
    final_learning_rate: float
    first_episode_loss: float
    final_episode_loss: float
    mean_loss: float
    train_accuracy: float
    train_roc: float
    train_prc: float
    duration_s: float


class GCNEncoder(nn.Module):
    """
    Graph convolutional encoder that embeds a molecular graph into a fixed-length
    vector, replicating the architecture in Table 3 of Vella and Ebejer (2023).

    Three graph convolutions (177 -> 64 -> 128 -> 64), each followed by a ReLU and
    by a max aggregation over the neighbourhood of every node, are followed by a
    sum readout over the nodes of the graph, a tanh activation, a dense layer and a
    final tanh activation producing the molecular embedding.

    The original work used the Deep Graph Library; this is the PyTorch Geometric
    equivalent. Self-loops are already present in the graphs produced by
    `PyGBondsDatasetCreator`, so the convolutions do not add them again. Bond
    (edge) features are not consumed, mirroring the original implementation.
    """

    def __init__(self, in_channels: int, embedding_size: int = 128) -> None:
        """
        Initialises the GCNEncoder.

        Args:
            in_channels (int): Dimensionality of the atom (node) feature vectors.
            embedding_size (int): Dimensionality of the molecular embedding produced.
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, 64, add_self_loops=False)
        self.conv2 = GCNConv(64, 128, add_self_loops=False)
        self.conv3 = GCNConv(128, 64, add_self_loops=False)
        # parameterless message passing, equivalent to DGL's
        # `update_all(copy_u('h', 'm'), max('m', 'h'))` max pooling step
        self.max_aggregation = SimpleConv(aggr="max")
        self.dense = nn.Linear(64, embedding_size)

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Embeds a batch of molecular graphs.

        Args:
            batch (Batch): A PyG batch of molecular graphs.

        Returns:
            torch.Tensor: Float tensor of shape (num_graphs, embedding_size).
        """
        x, edge_index = batch.x, batch.edge_index

        h = self.max_aggregation(F.relu(self.conv1(x, edge_index)), edge_index)
        h = self.max_aggregation(F.relu(self.conv2(h, edge_index)), edge_index)
        h = self.max_aggregation(F.relu(self.conv3(h, edge_index)), edge_index)

        out = global_add_pool(h, batch.batch)  # sum readout
        out = torch.tanh(out)
        out = self.dense(out)

        return torch.tanh(out)


class MLPEncoder(nn.Module):
    """
    Feedforward encoder that embeds a fixed-length molecular feature vector (ECFP,
    ChemBERTa or MoLFormer) into the same embedding space as `GCNEncoder`.

    This replicates the network Vella and Ebejer (2023) used to learn "a
    differentiable molecular embedding, from the ECFP, of the same size (vector of
    size 128) as the one produced by the GCN" for their ECFP vs. graph-learned
    embedding comparison: with the default hidden widths this is a
    2048 -> 1000 -> 500 -> 128 funnel, with ReLU after each hidden layer and a tanh
    on the embedding, matching the final tanh of `GCNEncoder`.

    The hidden widths are configurable because the reference network was written
    for 2048-bit ECFPs, whereas the ChemBERTa (384) and MoLFormer (768) embeddings
    this study also explores start from a much narrower input.
    """

    def __init__(
        self,
        in_features: int,
        embedding_size: int = 128,
        hidden_dims: tuple[int, ...] = (1000, 500),
    ) -> None:
        """
        Initialises the MLPEncoder.

        Args:
            in_features (int): Dimensionality of the molecular feature vectors.
            embedding_size (int): Dimensionality of the molecular embedding produced.
            hidden_dims (tuple[int, ...]): Widths of the hidden layers. Defaults to
                the (1000, 500) of the reference implementation.
        """
        super().__init__()

        layers: list[nn.Module] = []
        previous_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.dense = nn.Linear(previous_dim, embedding_size)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Embeds a batch of molecular feature vectors.

        Args:
            batch (torch.Tensor): Float tensor of shape (num_molecules, in_features).

        Returns:
            torch.Tensor: Float tensor of shape (num_molecules, embedding_size).
        """
        return torch.tanh(self.dense(self.hidden(batch)))


class SiameseNetwork(nn.Module):
    """
    Siamese network for few-shot ligand-based virtual screening.

    A single (shared weight) encoder embeds both molecules of a pair; the L1
    distance between the two embeddings is passed through a linear layer that
    outputs the logit of the two molecules belonging to the same (active) class.

    The twin towers are not evaluated separately per pair: because the encoder
    processes each molecule independently, every molecule of an episode is
    embedded exactly once and the pair head is then applied to the broadcast
    distances. This is mathematically identical to encoding each pair twice and
    makes the all-pairs test episodes tractable.
    """

    def __init__(self, encoder: nn.Module, embedding_size: int = 128) -> None:
        """
        Initialises the SiameseNetwork.

        Args:
            encoder (nn.Module): Molecular encoder shared by both twins.
            embedding_size (int): Dimensionality of the embeddings the encoder produces.
        """
        super().__init__()
        self.encoder = encoder
        self.out = nn.Linear(embedding_size, 1)

    def encode(self, batch) -> torch.Tensor:
        """
        Embeds a batch of molecules with the shared encoder.

        Args:
            batch: A collated batch of molecules, as returned by `collate_molecules`.

        Returns:
            torch.Tensor: Float tensor of shape (num_molecules, embedding_size).
        """
        return self.encoder(batch)

    def forward(self, support_embeddings: torch.Tensor, query_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Scores every (support, query) pair from their embeddings.

        Args:
            support_embeddings (torch.Tensor): Support embeddings of shape (n_support, embedding_size).
            query_embeddings (torch.Tensor): Query embeddings of shape (n_query, embedding_size).

        Returns:
            torch.Tensor: Flat tensor of n_support * n_query logits, ordered
                support-major (all queries for the first support molecule first),
                matching the pair target ordering from `create_episode`.
        """
        distance = torch.abs(support_embeddings.unsqueeze(1) - query_embeddings.unsqueeze(0))

        return self.out(distance).reshape(-1)


def build_encoder(
    feature_type: FeatureType,
    sample_molecule,
    embedding_size: int,
    mlp_hidden_dims: tuple[int, ...],
) -> tuple[nn.Module, int]:
    """
    Builds the encoder matching a feature type and infers its input dimensionality
    from a sample molecule.

    Args:
        feature_type (FeatureType): Feature representation of the dataset.
        sample_molecule: A single molecule taken from the dataset's 'mol' column,
            i.e. a PyG graph for graph feature types and a feature vector otherwise.
        embedding_size (int): Dimensionality of the embeddings the encoder produces.
        mlp_hidden_dims (tuple[int, ...]): Hidden layer widths of the MLP encoder.
            Ignored for graph feature types.

    Returns:
        tuple[nn.Module, int]: The encoder and its input dimensionality.
    """
    if feature_type in GRAPH_FEATURE_TYPES:
        in_features = int(sample_molecule.x.shape[-1])
        return GCNEncoder(in_channels=in_features, embedding_size=embedding_size), in_features

    in_features = int(np.asarray(sample_molecule).shape[-1])
    encoder = MLPEncoder(
        in_features=in_features,
        embedding_size=embedding_size,
        hidden_dims=mlp_hidden_dims,
    )

    return encoder, in_features


def collate_molecules(molecules: list, feature_type: FeatureType, device: torch.device):
    """
    Collates a list of molecules into a batch the encoder for `feature_type` accepts.

    Args:
        molecules (list): Molecules taken from a dataset's 'mol' column.
        feature_type (FeatureType): Feature representation of the dataset.
        device (torch.device): Device the batch is moved to.

    Returns:
        Batch | torch.Tensor: A PyG batch for graph feature types, otherwise a
            float tensor of shape (num_molecules, in_features).
    """
    if feature_type in GRAPH_FEATURE_TYPES:
        return Batch.from_data_list(molecules).to(device)

    return torch.as_tensor(np.stack(molecules), dtype=torch.float32, device=device)


def create_episode(
    task_df: pd.DataFrame,
    n_support_positive: int,
    n_support_negative: int,
    n_query_per_class: int,
    seed: int,
    test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Samples a single few-shot episode from an assay target.

    A support set of `n_support_positive` actives and `n_support_negative`
    inactives/decoys is drawn from the target; the query set is drawn from the
    molecules left over. During training the query set is balanced, with
    `n_query_per_class` molecules per class; the active class is over-sampled when
    the target holds fewer remaining actives than that, as described in the
    methodology of Vella and Ebejer (2023). During testing every remaining
    molecule of the target is used as a query.

    Each (support, query) pair is labelled 1 when both of its molecules are active
    and 0 otherwise, which is the pair target the Siamese network is trained on.

    Args:
        task_df (pd.DataFrame): Molecules of a single assay target, with 'y' and 'mol' columns.
        n_support_positive (int): Number of actives in the support set.
        n_support_negative (int): Number of inactives/decoys in the support set.
        n_query_per_class (int): Number of query molecules per class. Ignored when `test` is True.
        seed (int): Seed controlling the sampling, for reproducibility.
        test (bool): Whether the episode is a test episode (all remaining molecules
            are used as queries) rather than a training episode.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: The support set, the query
            set, and the flat pair targets ordered support-major.
    """
    support_negative_df = task_df[task_df["y"] == 0].sample(n_support_negative, random_state=seed)
    support_positive_df = task_df[task_df["y"] == 1].sample(n_support_positive, random_state=seed)
    support_df = pd.concat([support_negative_df, support_positive_df])

    remaining_df = task_df.drop(support_df.index)

    if test:
        query_df = remaining_df
    else:
        query_parts = []
        for label in (0, 1):
            class_df = remaining_df[remaining_df["y"] == label]
            # over-sample the class when it holds fewer molecules than the query set needs
            oversample = len(class_df) < n_query_per_class
            query_parts.append(
                class_df.sample(n_query_per_class, random_state=seed, replace=oversample)
            )
        query_df = pd.concat(query_parts)

    support_y = support_df["y"].to_numpy(dtype=np.int16)
    query_y = query_df["y"].to_numpy(dtype=np.int16)

    # a pair is positive only when both of its molecules are active
    pair_targets = ((support_y[:, None] == 1) & (query_y[None, :] == 1)).astype(np.float32).reshape(-1)

    return support_df, query_df, pair_targets


def encode_molecules(
    model: SiameseNetwork,
    molecules: list,
    feature_type: FeatureType,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """
    Embeds a list of molecules, in batches.

    Args:
        model (SiameseNetwork): The Siamese network whose shared encoder is used.
        molecules (list): Molecules taken from a dataset's 'mol' column.
        feature_type (FeatureType): Feature representation of the dataset.
        device (torch.device): Device the batches are moved to.
        batch_size (int): Maximum number of molecules embedded in one forward pass.

    Returns:
        torch.Tensor: Float tensor of shape (num_molecules, embedding_size).
    """
    embeddings = []
    for start in range(0, len(molecules), batch_size):
        batch = collate_molecules(molecules[start:start + batch_size], feature_type, device)
        embeddings.append(model.encode(batch))

    return torch.cat(embeddings)


def score_pairs(
    model: SiameseNetwork,
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Scores every (support, query) pair, in chunks over the query set.

    Test episodes pair a support set with every remaining molecule of a target, so
    the broadcast distance tensor is materialised one chunk of queries at a time to
    bound memory use. The returned ordering matches the pair targets from
    `create_episode`.

    Args:
        model (SiameseNetwork): The Siamese network scoring the pairs.
        support_embeddings (torch.Tensor): Support embeddings of shape (n_support, embedding_size).
        query_embeddings (torch.Tensor): Query embeddings of shape (n_query, embedding_size).
        chunk_size (int): Maximum number of queries scored at once.

    Returns:
        torch.Tensor: Flat tensor of n_support * n_query logits, ordered support-major.
    """
    n_support = support_embeddings.shape[0]
    chunks = []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        logits = model(support_embeddings, query_embeddings[start:start + chunk_size])
        chunks.append(logits.reshape(n_support, -1))

    return torch.cat(chunks, dim=1).reshape(-1)


def safe_roc_auc_score(targets: np.ndarray, scores: np.ndarray) -> float:
    """
    Computes the ROC-AUC, returning NaN when only one class is present.

    Args:
        targets (np.ndarray): Binary pair targets.
        scores (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        float: The ROC-AUC, or NaN if it is undefined for these targets.
    """
    if len(np.unique(targets)) < 2:
        return float("nan")

    return float(roc_auc_score(targets, scores))


def format_mean_std(values: list[float]) -> str:
    """
    Formats a list of per-round scores as a 'mean ± standard deviation' string.

    NaN scores, which arise when a metric is undefined for a round, are ignored.
    The standard deviation is reported as 0 when fewer than two scores remain.

    Args:
        values (list[float]): The per-round scores.

    Returns:
        str: The formatted summary of the scores.
    """
    scores = np.asarray(values, dtype=np.float64)
    scores = scores[~np.isnan(scores)]

    if scores.size == 0:
        return "nan ± nan"

    standard_deviation = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0

    return f"{float(np.mean(scores)):.3f} ± {standard_deviation:.3f}"


def eligible_train_tasks(
    train_dfs: dict[str, pd.DataFrame],
    n_support_positive: int,
    n_support_negative: int,
    n_query_per_class: int,
    logger: logging.Logger,
) -> list[str]:
    """
    Selects the training assay targets that hold enough molecules for an episode.

    A target is kept when it can provide the support set and still leave at least
    one molecule of each class to be sampled (and, where needed, over-sampled) into
    the query set.

    Args:
        train_dfs (dict[str, pd.DataFrame]): Per-target training DataFrames.
        n_support_positive (int): Number of actives in the support set.
        n_support_negative (int): Number of inactives/decoys in the support set.
        n_query_per_class (int): Number of query molecules per class.
        logger (logging.Logger): Logger used to report skipped targets.

    Returns:
        list[str]: Names of the usable training assay targets.
    """
    eligible = []
    for task, task_df in train_dfs.items():
        n_positive = int((task_df["y"] == 1).sum())
        n_negative = int((task_df["y"] == 0).sum())

        if n_positive <= n_support_positive or n_negative <= n_support_negative:
            logger.info(
                f"Skipping training assay target '{task}': '{n_positive}' actives and '{n_negative}' "
                f"inactives/decoys cannot fill a '{n_support_positive}'+/'{n_support_negative}'- support "
                f"set and a query set of '{n_query_per_class}' molecules per class."
            )
            continue

        eligible.append(task)

    return eligible


def train_siamese_network(
    train_dfs: dict[str, pd.DataFrame],
    feature_type: FeatureType,
    n_support_positive: int,
    n_support_negative: int,
    n_query_per_class: int,
    episodes: int,
    learning_rate: float,
    min_learning_rate: float,
    lr_patience: int,
    embedding_size: int,
    mlp_hidden_dims: tuple[int, ...],
    device: torch.device,
    seed: int,
    log_every: int,
    logger: logging.Logger,
) -> tuple[SiameseNetwork, TrainingRow, list[float]]:
    """
    Trains a Siamese network episodically over the training assay targets.

    Every episode visits each eligible training target once, in a random order,
    sampling a fresh episode from it and taking one optimiser step on the binary
    cross-entropy of its pair predictions. The learning rate is reduced on plateau
    and training stops early once it falls below `min_learning_rate`, mirroring the
    training procedure of Vella and Ebejer (2023). The support set composition
    matches the one used at test time, so that training and testing conditions agree.

    Args:
        train_dfs (dict[str, pd.DataFrame]): Per-target training DataFrames.
        feature_type (FeatureType): Feature representation of the dataset.
        n_support_positive (int): Number of actives in the support set.
        n_support_negative (int): Number of inactives/decoys in the support set.
        n_query_per_class (int): Number of query molecules per class.
        episodes (int): Maximum number of training episodes.
        learning_rate (float): Initial learning rate of the Adam optimiser.
        min_learning_rate (float): Learning rate below which training stops early.
        lr_patience (int): Number of episodes without improvement before the
            learning rate is reduced.
        embedding_size (int): Dimensionality of the molecular embeddings.
        mlp_hidden_dims (tuple[int, ...]): Hidden layer widths of the MLP encoder,
            used for the fixed-length vector feature types.
        device (torch.device): Device the network is trained on.
        seed (int): Seed controlling episode sampling and task shuffling.
        log_every (int): Number of episodes between progress log entries.
        logger (logging.Logger): Logger used to report training progress.

    Returns:
        tuple[SiameseNetwork, TrainingRow, list[float]]: The trained network, a
            summary of the training run, and the per-episode mean losses.
    """
    train_tasks = eligible_train_tasks(
        train_dfs=train_dfs,
        n_support_positive=n_support_positive,
        n_support_negative=n_support_negative,
        n_query_per_class=n_query_per_class,
        logger=logger,
    )
    if not train_tasks:
        raise ValueError(
            f"No training assay target can provide a '{n_support_positive}'+/'{n_support_negative}'- support set."
        )

    sample_molecule = train_dfs[train_tasks[0]]["mol"].iloc[0]
    encoder, in_features = build_encoder(
        feature_type=feature_type,
        sample_molecule=sample_molecule,
        embedding_size=embedding_size,
        mlp_hidden_dims=mlp_hidden_dims,
    )
    logger.info(f"Built the '{type(encoder).__name__}' encoder over '{in_features}' input features.")

    model = SiameseNetwork(encoder=encoder, embedding_size=embedding_size).to(device)
    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=lr_patience)

    episode_rng = np.random.default_rng(seed)
    task_rng = random.Random(seed)

    episode_losses: list[float] = []
    running_accuracy: list[float] = []
    running_roc: list[float] = []
    running_prc: list[float] = []

    start_time = time.perf_counter()
    episodes_run = 0

    model.train()
    for episode in range(episodes):
        episode_idx = episode + 1
        episodes_run = episode_idx
        episode_loss = 0.0

        for task in task_rng.sample(train_tasks, len(train_tasks)):
            episode_seed = int(episode_rng.integers(0, 2**31 - 1))

            support_df, query_df, pair_targets = create_episode(
                task_df=train_dfs[task],
                n_support_positive=n_support_positive,
                n_support_negative=n_support_negative,
                n_query_per_class=n_query_per_class,
                seed=episode_seed,
                test=False,
            )

            targets = torch.as_tensor(pair_targets, device=device)

            support_embeddings = model.encode(
                collate_molecules(support_df["mol"].tolist(), feature_type, device)
            )
            query_embeddings = model.encode(
                collate_molecules(query_df["mol"].tolist(), feature_type, device)
            )
            logits = model(support_embeddings, query_embeddings)

            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_loss += loss.item()

            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            running_accuracy.append(accuracy_score(pair_targets, (probabilities >= 0.5).astype(np.float32)))
            running_roc.append(safe_roc_auc_score(pair_targets, probabilities))
            running_prc.append(float(average_precision_score(pair_targets, probabilities)))

        mean_episode_loss = episode_loss / len(train_tasks)
        episode_losses.append(mean_episode_loss)
        scheduler.step(mean_episode_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if episode_idx == 1 or episode_idx % log_every == 0:
            logger.info(
                f"Episode '{episode_idx}' of '{episodes}'. Loss: '{mean_episode_loss:.6f}'. "
                f"Accuracy: '{running_accuracy[-1]:.4f}'. ROC: '{running_roc[-1]:.4f}'. "
                f"PRC: '{running_prc[-1]:.4f}'. Learning rate: '{current_lr}'"
            )

        if current_lr < min_learning_rate:
            logger.info(
                f"Stopping early at episode '{episode_idx}': the learning rate '{current_lr}' "
                f"fell below the '{min_learning_rate}' minimum."
            )
            break

    duration = time.perf_counter() - start_time

    training_row = TrainingRow(
        support_positive_count=n_support_positive,
        support_negative_count=n_support_negative,
        train_tasks=len(train_tasks),
        episodes_run=episodes_run,
        final_learning_rate=optimizer.param_groups[0]["lr"],
        first_episode_loss=episode_losses[0],
        final_episode_loss=episode_losses[-1],
        mean_loss=statistics.mean(episode_losses),
        train_accuracy=statistics.mean(running_accuracy),
        train_roc=float(np.nanmean(running_roc)),
        train_prc=statistics.mean(running_prc),
        duration_s=duration,
    )

    logger.info(
        f"Finished training after '{episodes_run}' episodes in '{duration:.1f}' seconds. "
        f"Mean loss: '{training_row['mean_loss']:.6f}'. Mean train ROC: '{training_row['train_roc']:.4f}'. "
        f"Mean train PRC: '{training_row['train_prc']:.4f}'"
    )

    return model, training_row, episode_losses


def save_loss_plot(plot_save_path: Path, plot_title: str, losses: list[float]) -> None:
    """
    Plots and saves the per-episode training loss curve.

    Args:
        plot_save_path (Path): File path to save the plot to. Parent directories are created if missing.
        plot_title (str): Title shown on the plot.
        losses (list[float]): Mean loss of each training episode.
    """
    plot_save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses) + 1), losses, color="tab:green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title(plot_title)
    fig.savefig(plot_save_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser("Siamese Network Experiment")
    parser.add_argument(
        "--dataset",
        type=DatasetSource,
        choices=list(DatasetSource),
        required=True,
        help="Dataset to run the experiments on.",
    )
    parser.add_argument(
        "--feature_type",
        type=FeatureType,
        choices=SUPPORTED_FEATURE_TYPES,
        required=True,
        help="Molecular feature representation the Siamese network embeds. Graph feature "
             "types are encoded with a GCN, fixed-length feature vectors with an MLP.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10000,
        help="Maximum number of training episodes. Every episode visits each training assay target once.",
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=64,
        help="Number of query molecules sampled per class in a training episode.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Initial learning rate of the Adam optimiser.",
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=0.000001,
        help="Training stops early once the learning rate falls below this value.",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=200,
        help="Number of episodes without improvement before the learning rate is reduced.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=128,
        help="Dimensionality of the molecular embedding learnt by the shared encoder.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        type=int,
        nargs="+",
        default=[1000, 500],
        help="Hidden layer widths of the MLP encoder used for the fixed-length feature "
             "vector feature types. Defaults to the 2048 -> 1000 -> 500 -> 128 funnel of "
             "the reference implementation. Ignored for graph feature types.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=20,
        help="Experiment parameter that defines how many rounds of testing are run per assay target.",
    )
    parser.add_argument(
        "--initial_random_seed",
        type=int,
        default=12,
        help="The initial random seed to be used for the first round. For every round after that, the random seed is increased by 1.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="Number of molecules embedded in one forward pass during testing.",
    )
    parser.add_argument(
        "--eval_chunk_size",
        type=int,
        default=1024,
        help="Number of query molecules scored against the support set at once during testing.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Number of training episodes between progress log entries.",
    )
    parser.add_argument(
        "--persist_plots",
        action="store_true",
        help="Generate plots and save to disk.",
    )

    args = parser.parse_args()
    dataset_source = DatasetSource(args.dataset)
    feature_type = FeatureType(args.feature_type)
    episodes = args.episodes
    n_query = args.n_query
    learning_rate = args.learning_rate
    min_learning_rate = args.min_learning_rate
    lr_patience = args.lr_patience
    embedding_size = args.embedding_size
    mlp_hidden_dims = tuple(args.mlp_hidden_dims)
    n_rounds = args.n_rounds
    random_seed = args.initial_random_seed
    eval_batch_size = args.eval_batch_size
    eval_chunk_size = args.eval_chunk_size
    log_every = args.log_every
    persist_plots = args.persist_plots

    dataset_source_val = dataset_source.value
    feature_type_val = feature_type.value

    combinations = [
        [10, 10],
        [5, 10],
        [1, 10],
        [1, 5],
        [1, 1]
    ]

    experiment_name = "siamese_network"
    _, experiment_dir, logger = ExperimentHelper.start_experiment(
        name=experiment_name,
        base_dir=EXPERIMENT_ENTRIES_DIR
    )

    handler = DatasetHandler()

    _ = ExperimentHelper.get_device_info(logger=logger)
    device = ExperimentHelper.set_global_seed(random_seed)

    logger.info("Experiment Parameters:")
    logger.info(f"Dataset: '{dataset_source_val}'")
    logger.info(f"Feature Type: '{feature_type_val}'")
    logger.info(f"Device: '{device}'")
    logger.info(f"Maximum number of training episodes: '{episodes}'")
    logger.info(f"Number of query molecules per class (training): '{n_query}'")
    logger.info(f"Learning rate: '{learning_rate}'")
    logger.info(f"Minimum learning rate (early stop): '{min_learning_rate}'")
    logger.info(f"Learning rate plateau patience: '{lr_patience}'")
    logger.info(f"Embedding size: '{embedding_size}'")
    logger.info(f"MLP encoder hidden dimensions: '{mlp_hidden_dims}'")
    logger.info(f"Number of test rounds: '{n_rounds}'")
    logger.info(f"Generate & persist plots: '{persist_plots}'")
    logger.info(f"Initial random seed: '{random_seed}'")
    logger.info(f"Combinations of no. of positive vs. no. of negative to be used for the support set: '{combinations}'")

    train_dfs, test_dfs = handler.load_train_test_set(dataset_source=dataset_source, feature_type=feature_type)

    round_entry_rows: list[RoundEntry] = []
    summary_rows: list[SummaryRow] = []
    training_rows: list[TrainingRow] = []

    round_entry_csv = experiment_dir / "rounds.csv"
    summary_csv = experiment_dir / "summary.csv"
    training_csv = experiment_dir / "training.csv"
    plots_dir = experiment_dir / "plots"
    models_dir = experiment_dir / "models"

    for no_positive, no_negative in combinations:
        logger.info(
            f"Training a Siamese network on the '{dataset_source_val}' training assay targets using "
            f"'{no_positive}' positives and '{no_negative}' negatives in the support set."
        )

        # re-seed per combination so that each network starts from the same initialisation
        _ = ExperimentHelper.set_global_seed(random_seed)

        model, training_row, episode_losses = train_siamese_network(
            train_dfs=train_dfs,
            feature_type=feature_type,
            n_support_positive=no_positive,
            n_support_negative=no_negative,
            n_query_per_class=n_query,
            episodes=episodes,
            learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            lr_patience=lr_patience,
            embedding_size=embedding_size,
            mlp_hidden_dims=mlp_hidden_dims,
            device=device,
            seed=random_seed,
            log_every=log_every,
            logger=logger,
        )
        training_rows.append(training_row)

        model_path = models_dir / f"pos{no_positive}_neg{no_negative}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving the trained Siamese network at '{model_path}'")
        torch.save(model.state_dict(), model_path)

        if persist_plots:
            loss_save_path = plots_dir / "training" / f"pos{no_positive}_neg{no_negative}.png"
            loss_plot_title = f"Training Loss - {no_positive} Positive - {no_negative} Negative"
            logger.info(f"Saving training loss plot at '{loss_save_path}'")
            save_loss_plot(plot_save_path=loss_save_path, plot_title=loss_plot_title, losses=episode_losses)

        model.eval()

        for target in test_dfs.keys():
            logger.info(
                f"Running a '{n_rounds}' round test on the '{target}' assay target using "
                f"'{no_positive}' positives and '{no_negative}' negatives in the support set."
            )
            running_acc = []
            running_roc = []
            running_prc = []
            running_duration = []

            test_df = test_dfs[target]
            total_positive_count = int((test_df["y"] == 1).sum())
            total_negative_count = int((test_df["y"] == 0).sum())

            for round in range(n_rounds):
                seed = random_seed + round
                round_idx = round + 1
                round_start_time = time.perf_counter()
                logger.info(f"Running round '{round_idx}' of '{n_rounds}'. Seed: '{seed}'")

                support_df, query_df, pair_targets = create_episode(
                    task_df=test_df,
                    n_support_positive=no_positive,
                    n_support_negative=no_negative,
                    n_query_per_class=n_query,
                    seed=seed,
                    test=True,
                )

                with torch.no_grad():
                    support_embeddings = encode_molecules(
                        model=model,
                        molecules=support_df["mol"].tolist(),
                        feature_type=feature_type,
                        device=device,
                        batch_size=eval_batch_size,
                    )
                    query_embeddings = encode_molecules(
                        model=model,
                        molecules=query_df["mol"].tolist(),
                        feature_type=feature_type,
                        device=device,
                        batch_size=eval_batch_size,
                    )
                    logits = score_pairs(
                        model=model,
                        support_embeddings=support_embeddings,
                        query_embeddings=query_embeddings,
                        chunk_size=eval_chunk_size,
                    )
                    probs_y = torch.sigmoid(logits).cpu().numpy()

                preds_y = (probs_y >= 0.5).astype(np.float32)

                round_accuracy = accuracy_score(pair_targets, preds_y)
                round_roc = safe_roc_auc_score(pair_targets, probs_y)
                round_prc = average_precision_score(pair_targets, probs_y)

                false_positive_rate, true_positive_rate, _ = roc_curve(pair_targets, probs_y)
                precision, recall, _ = precision_recall_curve(pair_targets, probs_y)

                if persist_plots:
                    png_file_name = f"pos{no_positive}_neg{no_negative}/round{round_idx}.png"

                    roc_save_path = plots_dir / target / "roc" / png_file_name
                    roc_plot_title = f"{target} - Round {round_idx} - {no_positive} Positive - {no_negative} Negative - AUC={round_roc:.3f}"
                    logger.info(f"Saving ROC curve plot at '{roc_save_path}'")
                    ExperimentHelper.save_roc_plot(
                        plot_save_path=roc_save_path,
                        plot_title=roc_plot_title,
                        false_positive_rate=false_positive_rate,
                        true_positive_rate=true_positive_rate,
                    )

                    prc_save_path = plots_dir / target / "prc" / png_file_name
                    prc_plot_title = f"{target} - Round {round_idx} - {no_positive} Positive - {no_negative} Negative - AP={round_prc:.3f}"
                    logger.info(f"Saving PRC curve plot at '{prc_save_path}'")
                    ExperimentHelper.save_prc_plot(
                        plot_save_path=prc_save_path,
                        plot_title=prc_plot_title,
                        precision=precision,
                        recall=recall,
                    )

                round_end_time = time.perf_counter()
                round_duration = round_end_time - round_start_time

                round_entry_rows.append(
                    RoundEntry(
                        assay_target=target,
                        support_positive_count=no_positive,
                        support_negative_count=no_negative,
                        round_no=round_idx,
                        seed=seed,
                        query_positive_count=int((query_df["y"] == 1).sum()),
                        query_negative_count=int((query_df["y"] == 0).sum()),
                        pair_count=int(pair_targets.size),
                        positive_pair_count=int(pair_targets.sum()),
                        accuracy=round_accuracy,
                        roc=round_roc,
                        prc=round_prc,
                        duration_s=round_duration,
                    )
                )

                running_acc.append(round_accuracy)
                running_roc.append(round_roc)
                running_prc.append(round_prc)
                running_duration.append(round_duration)

            final_accuracy = format_mean_std(running_acc)
            final_roc = format_mean_std(running_roc)
            final_prc = format_mean_std(running_prc)
            total_duration = sum(running_duration)

            logger.info(
                f"Results for the '{target}' assay target using '{no_positive}' positives and "
                f"'{no_negative}' negatives. Accuracy: '{final_accuracy}'. ROC: '{final_roc}'. PRC: '{final_prc}'"
            )

            summary_rows.append(
                SummaryRow(
                    assay_target=target,
                    support_positive_count=no_positive,
                    support_negative_count=no_negative,
                    total_positive_count=total_positive_count,
                    total_negative_count=total_negative_count,
                    accuracy=final_accuracy,
                    roc=final_roc,
                    prc=final_prc,
                    duration_s=total_duration,
                )
            )

        logger.info(f"Saving per-round experiment results to CSV at '{round_entry_csv}'")
        ExperimentHelper.save_typed_dict_to_csv(round_entry_csv, RoundEntry, round_entry_rows)

        logger.info(f"Saving results summary to CSV at '{summary_csv}'")
        ExperimentHelper.save_typed_dict_to_csv(summary_csv, SummaryRow, summary_rows)

        logger.info(f"Saving training summary to CSV at '{training_csv}'")
        ExperimentHelper.save_typed_dict_to_csv(training_csv, TrainingRow, training_rows)

main()
