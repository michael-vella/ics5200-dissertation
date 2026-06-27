from abc import ABC, abstractmethod
import logging

import pandas as pd


class DatasetCreationStrategy(ABC):
    """
    Abstract base class for all dataset creation strategies.

    Each concrete strategy encapsulates a specific pipeline for transforming
    a raw CSV into a processed dataset.
    """
    @abstractmethod
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform a raw DataFrame into a processed dataset.

        Args:
            df (DataFrame): Raw DataFrame loaded from a CSV file.
            **kwargs: Strategy-specific parameters (injected only if required
                for that specific dataset creation strategy).
        """
        pass