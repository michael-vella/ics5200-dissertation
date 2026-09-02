import logging
from abc import ABC, abstractmethod

import pandas as pd


type TrainTestData = tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]


class TrainTestStrategy(ABC):
    """
    Abstract base class for all train-test set retrieval strategies.

    Each concrete strategy encapsulates how a processed dataset is split into
    per-assay train and test sets for a specific feature type.
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
        Splits a processed dataset into per-assay train and test sets.

        Args:
            pdf (pd.DataFrame): Processed dataset containing a 'mol' column of
                features and one column per assay.
            train_assays (list[str]): Names of the assay columns to use for training.
            test_assays (list[str]): Names of the assay columns to use for testing.

        Returns:
            TrainTestData: A tuple of (train assays dict, test assays dict),
                each mapping an assay name to a DataFrame with 'y' and 'mol' columns.
        """
        pass