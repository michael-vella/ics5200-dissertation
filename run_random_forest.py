import argparse
import time
import statistics
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    accuracy_score,
)

from datasets.enums import DatasetSource, FeatureType
from datasets.dataset_handler import DatasetHandler
from experiment_helper import ExperimentHelper
from constants import EXPERIMENT_ENTRIES_DIR


class RoundEntry(TypedDict):
    assay_target: str
    round_no: int
    seed: int
    train_positive_count: int
    test_positive_count: int
    train_negative_count: int
    test_negative_count: int
    accuracy: float
    roc: float
    prc: float
    duration_s: int


class SummaryRow(TypedDict):
    assay_target: str
    total_positive_count: int
    total_negative_count: int
    train_positive_count: int
    test_positive_count: int
    train_negative_count: int
    test_negative_count: int
    accuracy: str
    roc: str
    prc: str
    duration_s: int


def main() -> None:
    parser = argparse.ArgumentParser("Random Forest Experiment")
    parser.add_argument(
        "--dataset",
        type=DatasetSource,
        choices=list(DatasetSource),
        required=True,
        help="Dataset to run the experiments on."
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Hyper-parameter that defines the total number of decision trees in the forest.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=20,
        help="Experiment parameter that defines how many rounds a Random Forest classification is run.",
    )
    parser.add_argument(
        "--initial_random_seed",
        type=int,
        default=12,
        help="The initial random seed to be used for the first round. For every round after that, the random seed is increased by 1.",
    )
    parser.add_argument(
        "--persist_plots",
        action="store_true",
        help="Generate plots and save to disk.",
    )
    
    args = parser.parse_args()
    n_estimators = args.n_estimators
    n_rounds = args.n_rounds
    dataset_source = DatasetSource(args.dataset)
    persist_plots = args.persist_plots
    random_seed = args.initial_random_seed

    # hard-code the feature type as only ECFP fingerprints can be used for RF
    feature_type = FeatureType.ECFP

    dataset_source_val = dataset_source.value
    feature_type_val = feature_type.value

    combinations = [
        [10, 10],
        [5, 10], 
        [1, 10], 
        [1, 5], 
        [1, 1]
    ]

    experiment_name = "random_forest"
    _, experiment_dir, logger = ExperimentHelper.start_experiment(
        name=experiment_name,
        base_dir=EXPERIMENT_ENTRIES_DIR
    )

    handler = DatasetHandler()

    _ = ExperimentHelper.get_device_info(logger=logger)
    _ = ExperimentHelper.set_global_seed(random_seed)

    logger.info("Experiment Parameters:")
    logger.info(f"Dataset: '{dataset_source_val}'")
    logger.info(f"Feature Type: '{feature_type_val}'")
    logger.info(f"No. of (RF) estimators: '{n_estimators}'")
    logger.info(f"Number of rounds: '{n_rounds}'")
    logger.info(f"Generate & persist plots: '{persist_plots}'")
    logger.info(f"Initial random seed: '{random_seed}'")
    logger.info(f"Combinations of no. of positive vs. no. of negative to be used for training: '{combinations}'")

    # train tasks discarded: RF is a no-transfer baseline, fit per-target on the support set only
    _, test_dfs = handler.load_train_test_set(dataset_source=dataset_source, feature_type=feature_type)

    round_entry_rows: list[RoundEntry] = []
    summary_rows: list[SummaryRow] = []

    round_entry_csv = experiment_dir / "rounds.csv"
    summary_csv = experiment_dir / "summary.csv"
    plots_dir = experiment_dir / "plots"

    for target in test_dfs.keys():
        for no_positive, no_negative in combinations:
            logger.info(f"Running a '{n_rounds}' round experiment on the '{target}' assay target using '{no_positive}' positives and '{no_negative}' negatives for training.")
            running_acc = []
            running_roc = []
            running_prc = []
            running_duration = []

            for round in range(n_rounds):
                seed = random_seed + round
                round_idx = round + 1
                round_start_time = time.perf_counter()
                logger.info(f"Running round '{round_idx}' of '{n_rounds}'. Seed: '{seed}'")
                test_df = test_dfs[target]

                test_negative_df = test_df[test_df['y'] == 0].sample(no_negative, random_state=seed)
                test_positive_df = test_df[test_df['y'] == 1].sample(no_positive, random_state=seed)
    
                train_df = pd.concat([test_negative_df, test_positive_df])
                test_df = test_df.drop(train_df.index)

                train_X, train_y = list(train_df['mol'].to_numpy()), train_df['y'].to_numpy(dtype=np.int16)
                test_X, test_y = list(test_df['mol'].to_numpy()), test_df['y'].to_numpy(dtype=np.int16)

                train_positive_count = int((train_y == 1).sum())
                test_positive_count = int((test_y == 1).sum())
                train_negative_count = int((train_y == 0).sum())
                test_negative_count = int((test_y == 0).sum())

                total_positive_count = train_positive_count + test_positive_count
                total_negative_count = train_negative_count + test_negative_count

                model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
                model.fit(train_X, train_y)
                probs_y = model.predict_proba(test_X)
                preds_y = (probs_y[:, 1] >= 0.5).astype(int)

                round_accuracy = accuracy_score(test_y, preds_y)
                round_roc = roc_auc_score(test_y, probs_y[:, 1])
                round_prc = average_precision_score(test_y, probs_y[:, 1])

                false_positive_rate, true_positive_rate, _ = roc_curve(test_y, probs_y[:, 1])
                precision, recall, _ = precision_recall_curve(test_y, probs_y[:, 1])

                if persist_plots:
                    # png_file_name = f"round{round_idx}_{no_positive}pos_{no_negative}neg.png"
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
                        round_no=round_idx,
                        seed=seed,
                        train_positive_count=train_positive_count,
                        test_positive_count=test_positive_count,
                        train_negative_count= train_negative_count,
                        test_negative_count=test_negative_count,
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

            final_accuracy = f"{statistics.mean(running_acc):.3f} \u00B1 {statistics.stdev(running_acc):.3f}"
            final_roc = f"{statistics.mean(running_roc):.3f} \u00B1 {statistics.stdev(running_roc):.3f}"
            final_prc = f"{statistics.mean(running_prc):.3f} \u00B1 {statistics.stdev(running_prc):.3f}"
            total_duration = sum(running_duration)

            summary_rows.append(
                SummaryRow(
                    assay_target=target,
                    total_positive_count=total_positive_count,
                    total_negative_count=total_negative_count,
                    train_positive_count=train_positive_count,
                    test_positive_count=test_positive_count,
                    train_negative_count=train_negative_count,
                    test_negative_count=test_negative_count,
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

main()