import logging
from abc import ABC, abstractmethod

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


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
        Parses a SMILES string into a standardized RDKit molecule.

        RDKit rejects a small number of SMILES strings that violate its valence
        rules (for example hypervalent aluminium centres in TOX21), returning
        None rather than raising. Such molecules are logged and reported back as
        None so that the calling strategy can skip them instead of failing.

        Molecules that do parse are then standardized so that molecules from
        different sources are represented in a single, consistent way (for use
        in ML, catalogue, etc.): Hs are removed, metal atoms disconnected, the
        molecule normalized and reionized, the parent fragment extracted (in
        case of multiple fragments), the molecule neutralized where possible,
        and a canonical tautomer chosen. No attempt is made at reionization at
        this step, nor at ionization at some pH (RDKit has no pKa calculator).

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            Chem.Mol | None: The standardized molecule, or None if the SMILES
                string could not be parsed.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            self._logger.warning(f"Unable to parse SMILES string '{smiles}', molecule will be dropped")
            return None

        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)

        # if many fragments, get the "parent" (the actual mol we are interested in)
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        te = rdMolStandardize.TautomerEnumerator() # idem
        return te.Canonicalize(uncharged_parent_clean_mol)

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
