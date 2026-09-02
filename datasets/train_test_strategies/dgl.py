import logging
from typing import override

import pandas as pd

from .base_strategy import TrainTestData
from .ecfp import ECFPTrainTestStrategy


class DGLTrainTestStrategy(ECFPTrainTestStrategy):
    """
    Class implementing the train-test set retrieval strategy for the DGL feature type.

    Reuses the assay dict building logic from ECFPTrainTestStrategy, since both
    feature types share the same 'mol' column structure (target plus a single
    feature object per row) and differ only in what that object represents.
    """
    def __init__(self) -> None:
        """
        Initialises the DGLTrainTestStrategy class.
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
        Splits a DGL processed dataset into per-assay train and test sets.

        Args:
            pdf (pd.DataFrame): Processed dataset containing a 'mol' column of
                DGL graph objects and one column per assay.
            train_assays (list[str]): Names of the assay columns to use for training.
            test_assays (list[str]): Names of the assay columns to use for testing.

        Returns:
            TrainTestData: A tuple of (train assays dict, test assays dict),
                each mapping an assay name to a DataFrame with 'y' and 'mol' columns.
        """
        self._logger.info("Retrieving train-test sets for the DGL feature type")

        self._logger.info("Iterating over assays in the training set ")
        train_assays_dict = self._build_assay_dict(pdf, train_assays)
        test_assays_dict = self._build_assay_dict(pdf, test_assays)

        self._logger.info("Finished train-test sets retrieval for the DGL feature type")

        return train_assays_dict, test_assays_dict