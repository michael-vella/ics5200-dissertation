import argparse
from typing import TypedDict

from datasets.enums import DatasetSource, FeatureType
from experiment_helper import ExperimentHelper
from constants import EXPERIMENT_ENTRIES_DIR


class RoundEntry(TypedDict):
    assay_target: str
    round_no: int
    seed: int
    training_positive_no: int
    training_negative_no: int
    accuracy: float
    false_positive_rate: float
    true_positive_rate: float
    precision: float
    recall: float
    roc: float
    prc: float
    duration: int


class SummaryRow(TypedDict):
    assay_target: str
    total_positive_no: int
    total_negative_no: int
    training_positive_no: int
    training_negative_no: int
    accuracy: str
    roc: str
    prc: str
    duration: int


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
        help="Hyperparameter that defines the total number of decision trees in the forest.",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=20,
        help="Experiment parameter that defines how many rounds a Random Forest classification is run.",
    )
    

    args = parser.parse_args()
    n_estimators = args.n_estimators
    n_rounds = args.n_rounds
    dataset_source = DatasetSource(args.dataset)
    dataset_source_val = dataset_source.value
    
    feature_type = FeatureType.ECFP
    feature_type_val = feature_type.value

    experiment_name = "random_forest"
    _, experiment_dir, logger = ExperimentHelper.start_experiment(
        name=experiment_name,
        base_dir=EXPERIMENT_ENTRIES_DIR
    )

    _ = ExperimentHelper.get_device_info(logger=logger)

    logger.info("Experiment Parameters:")
    logger.info(f"No. of (RF) estimators: '{n_estimators}'")
    logger.info(f"Number of rounds: '{n_rounds}'")
    logger.info(f"Dataset: '{dataset_source_val}'")
    logger.info(f"Feature Type: '{feature_type_val}'")



    #todo: continue building on this.


main()