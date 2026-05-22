from typing import override

import pandas as pd

from datasets.creation_strategies.base_strategy import DatasetCreationStrategy


class ECFPDatasetCreator(DatasetCreationStrategy):
    """
    Class implementing the ECFP dataset creation strategy.

    An ECFP (Extended-Connectivity Fingerprint) is a highly
    popular, circular topological fingerprint used in chemistry
    and cheminformatics to represent molecular structures mathematically. 
    """
    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        todo
        """
        pass