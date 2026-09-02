import logging
from typing import override

import pandas as pd

from .base_strategy import TrainTestData
from .ecfp import ECFPTrainTestStrategy


class DGLTrainTestStrategy(ECFPTrainTestStrategy):
    """
    todo
    """
    def __init__(self) -> None:
        """
        todo
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
        todo
        """
        self._logger.info("Retrieving train-test sets for the DGL feature type")

        self._logger.info("Iterating over assays in the training set ")
        train_assays_dict = self._build_assay_dict(pdf, train_assays)
        test_assays_dict = self._build_assay_dict(pdf, test_assays)

        self._logger.info("Finished train-test sets retrieval for the DGL feature type")

        return train_assays_dict, test_assays_dict