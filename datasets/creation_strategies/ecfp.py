import logging
from typing import override

import pandas as pd
import numpy as np
from rdkit.Chem import rdFingerprintGenerator

from .base_strategy import DatasetCreationStrategy


class ECFPDatasetCreator(DatasetCreationStrategy):
    """
    Class implementing the ECFP dataset creation strategy.

    An ECFP (Extended-Connectivity Fingerprint) is a highly
    popular, circular topological fingerprint used in chemistry
    and cheminformatics to represent molecular structures mathematically. 
    """
    def __init__(self) -> None:
        """
        Initialises the ECFPDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates an ECFP-based dataset from a DataFrame containing SMILES strings.

        Each SMILES string is featurised using a MorganGenerator (ECFP4) with a
        radius of 2 and a fingerprint size of 2048 bits. The resulting fingerprint
        vector is stored as a new column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                            SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                        containing the ECFP fingerprint as a NumPy float32 array.
                        Molecules whose SMILES string could not be parsed are dropped.
        """
        self._logger.info("Initialising Morgan fingerprint generator for feature extraction")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        self._logger.info("Featurising SMILES strings into ECFP fingerprints. Please wait this process might take a while")
        df['mol'] = df['smiles'].apply(lambda x: self._create_features(smiles=x, generator=generator))
        df = self._drop_unfeaturised(df)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    def _create_features(self, smiles: str, generator) -> np.ndarray | None:
        """
        Converts a SMILES string into an ECFP fingerprint vector.

        Args:
            smiles (str): A SMILES string representing a molecular structure.
            generator: A MorganGenerator used to compute the fingerprint.

        Returns:
            np.ndarray | None: The fingerprint as a NumPy float32 array, or None
                if the SMILES string could not be parsed.
        """
        mol = self._parse_smiles(smiles)
        if mol is None:
            return None

        return np.float32(generator.GetFingerprintAsNumPy(mol))
