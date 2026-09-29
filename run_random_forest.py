import argparse

from experiment_helper import ExperimentHelper
from constants import EXPERIMENT_ENTRIES_DIR


def main() -> None:
    parser = argparse.ArgumentParser("Random Forest Experiment")
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

    experiment_name = "random_forest"
    _, experiment_dir, logger = ExperimentHelper.start_experiment(
        name=experiment_name,
        base_dir=EXPERIMENT_ENTRIES_DIR
    )

    _ = ExperimentHelper.get_device_info(logger=logger)

    logger.info("Experiment Parameters:")
    logger.info(f"No. of (RF) estimators: '{n_estimators}'")
    logger.info(f"Number of rounds: '{n_rounds}'")



    #todo: continue building on this & add dataset source as an argument.


main()