from typing import override

import pandas as pd
import numpy as np
import deepchem as dc

from datasets.creation_strategies.base_strategy import DatasetCreationStrategy


class ECFPDatasetCreator(DatasetCreationStrategy):
    """
    Class implementing the ECFP dataset creation strategy.

    An ECFP (Extended-Connectivity Fingerprint) is a highly
    popular, circular topological fingerprint used in chemistry
    and cheminformatics to represent molecular structures mathematically. 
    """
    def __init__(self) -> None:
        """
        Initialises the ECFPDatasetCreator by calling the parent class constructor.
        """
        super().__init__()

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates an ECFP-based dataset from a DataFrame containing SMILES strings.

        Each SMILES string is featurised using a CircularFingerprint (ECFP4) with a
        radius of 2 and a fingerprint size of 2048 bits. The resulting fingerprint
        vector is stored as a new column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                            SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                        containing the ECFP fingerprint as a NumPy float32 array.
        """
        self._logger.info("Initialising ECFP featuriser for feature extraction")
        featurizer = dc.feat.CircularFingerprint(size=2048, radius=2)

        self._logger.info("Featurising SMILES strings into ECFP fingerprints")
        df['mol'] = df['smiles'].apply(lambda x: np.float32(featurizer(x)[0])) # todo: getting deprecation warning to be replaced with MorganGenerator

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df