import logging
from abc import ABC, abstractmethod

import pandas as pd


type TrainTestData = tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]


class TrainTestStrategy(ABC):
    """
    todo
    """
    _logger: logging.Logger

    @abstractmethod
    def get_train_test(
        self,
        pdf: pd.DataFrame,
        train_assays: list[str],
        test_assays: list[str]
    ) -> TrainTestData:
        """
        todo
        """
        pass