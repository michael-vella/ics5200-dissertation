import logging
from typing import override

import pandas as pd

from .base_strategy import TrainTestStrategy, TrainTestData


class ECFPTrainTestStrategy(TrainTestStrategy):
    """
    Class implementing the train-test set retrieval strategy for the ECFP feature type.
    """
    def __init__(self) -> None:
        """
        Initialises the ECFPTrainTestStrategy class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def get_train_test(
        self,
        pdf: pd.DataFrame,
        train_assays: list[str],
        test_assays: list[str]
    ) -> TrainTestData:
        """
        Splits an ECFP processed dataset into per-assay train and test sets.

        Args:
            pdf (pd.DataFrame): Processed dataset containing a 'mol' column of
                ECFP fingerprints and one column per assay.
            train_assays (list[str]): Names of the assay columns to use for training.
            test_assays (list[str]): Names of the assay columns to use for testing.

        Returns:
            TrainTestData: A tuple of (train assays dict, test assays dict),
                each mapping an assay name to a DataFrame with 'y' and 'mol' columns.
        """
        self._logger.info("Retrieving train-test sets for the ECFP feature type")

        self._logger.info("Iterating over assays in the training set ")
        train_assays_dict = self._build_assay_dict(pdf, train_assays)
        test_assays_dict = self._build_assay_dict(pdf, test_assays)

        self._logger.info("Finished train-test sets retrieval for the ECFP feature type")

        return train_assays_dict, test_assays_dict

    def _build_assay_dict(
        self,
        pdf: pd.DataFrame,
        assays: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Builds a per-assay DataFrame of target and ECFP fingerprint pairs.

        For each assay, selects the assay's target column together with the
        'mol' feature column, drops rows with missing values, and renames the
        columns to 'y' (target) and 'mol' (ECFP fingerprint).

        Args:
            pdf (pd.DataFrame): Processed dataset containing a 'mol' column of
                ECFP fingerprints and one column per assay.
            assays (list[str]): Names of the assay columns to build DataFrames for.

        Returns:
            dict[str, pd.DataFrame]: Mapping of assay name to a DataFrame with
                'y' and 'mol' columns.
        """
        assays_dict = dict.fromkeys(assays)
        for assay in assays:
            task_df = pdf[[assay, "mol"]].dropna()
            task_df.columns = ["y", "mol"]  # y is the target, mol is the embedding (ECFP)

            assays_dict[assay] = task_df

        return assays_dict