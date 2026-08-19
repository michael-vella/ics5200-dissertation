import logging
from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem


class DatasetCreationStrategy(ABC):
    """
    Abstract base class for all dataset creation strategies.

    Each concrete strategy encapsulates a specific pipeline for transforming
    a raw CSV into a processed dataset.
    """
    _logger: logging.Logger

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

    def _parse_smiles(self, smiles: str) -> Chem.Mol | None:
        """
        Parses a SMILES string into an RDKit molecule.

        RDKit rejects a small number of SMILES strings that violate its valence
        rules (for example hypervalent aluminium centres in TOX21), returning
        None rather than raising. Such molecules are logged and reported back as
        None so that the calling strategy can skip them instead of failing.

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            Chem.Mol | None: The parsed molecule, or None if the SMILES string
                could not be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._logger.warning(f"Unable to parse SMILES string '{smiles}', molecule will be dropped")

        return mol

    def _drop_unfeaturised(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes rows whose SMILES string could not be featurised.

        Rows for which featurisation returned None (i.e. the SMILES string could
        not be parsed) are dropped and the index is reset so that the processed
        dataset remains contiguous.

        Args:
            df (pd.DataFrame): DataFrame containing a 'mol' column of features.

        Returns:
            pd.DataFrame: The DataFrame with unfeaturisable molecules removed.
        """
        unfeaturised = df['mol'].isna()

        if unfeaturised.any():
            self._logger.warning(f"Dropping '{unfeaturised.sum()}' of '{len(df)}' molecules that could not be featurised")

        return df[~unfeaturised].reset_index(drop=True)
