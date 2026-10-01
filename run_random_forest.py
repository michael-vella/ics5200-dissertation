import argparse
import time
from typing import TypedDict

import numpy as np
import pandas as pd

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
    false_positive_rate: float
    true_positive_rate: float
    precision: float
    recall: float
    roc: float
    prc: float
    duration_ms: int


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
    duration_ms: int


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

            target_start_time = time.perf_counter()
            for round in n_rounds:
                seed = random_seed + round
                round_idx = round + 1
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

                # todo: continue below + add logging when running.

main()