import logging
from typing import override

import dgl
import torch
import pandas as pd
from rdkit import Chem
from dgllife import utils
from chembl_structure_pipeline import standardizer

from .base_strategy import DatasetCreationStrategy


class DGLDatasetCreator(DatasetCreationStrategy):
    """
    Dataset creation strategy that represents molecules as DGL graphs
    with atom-level node features only.

    Each molecule's SMILES string is parsed and standardised before being
    converted into a bidirectional DGL graph (bigraph), where:
      - Nodes represent atoms, featurised via ``featurize_atoms``.
      - Edges represent bonds but carry no feature information.
      - Self-loops are added to every node.
    """
    def __init__(self) -> None:
        """
        Initialises the DGLDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates a DGL graph dataset from a DataFrame containing SMILES strings.

        Each SMILES string is standardised using the ChEMBL structure pipeline
        and then converted into a bidirectional DGL graph (bigraph) with
        atom-level node features. Self-loops are added to every node.
        The resulting graph is stored as a new 'mol' column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                               SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                          containing the DGL graph for each molecule.
        """
        self._logger.info("Featurising SMILES strings into DGL graphs. Please wait, this process might take a while")
        df['mol'] = df['smiles'].apply(self._create_features)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    def _create_features(self, smiles: str) -> dgl.DGLGraph:
        """
        Converts a SMILES string into a DGL bidirectional graph with atom features.

        The molecule is first standardised via the ChEMBL structure pipeline to
        ensure consistent representations, then converted to a bigraph. Self-loops
        are appended so that each atom attends to itself during message passing.

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            dgl.DGLGraph: A DGL graph with node feature key ``'feats'`` containing
                          concatenated one-hot atom descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = standardizer.standardize_mol(mol)

        dgl_graph = utils.mol_to_bigraph(
            mol=mol,
            node_featurizer=self._featurise_atoms,
            canonical_atom_order=True
        )

        return dgl.add_self_loop(dgl_graph)

    def _featurise_atoms(self, mol) -> dict:
        """
        Computes concatenated one-hot atom features for all atoms in a molecule.

        The following per-atom properties are encoded as one-hot vectors and
        concatenated into a single feature vector:
          - Atom type (element symbol)
          - Atomic number
          - Degree (number of bonded neighbours)
          - Explicit valence
          - Formal charge
          - Number of radical electrons
          - Hybridisation state
          - Aromaticity flag

        Args:
            mol: An RDKit ``Mol`` object whose atoms are to be featurised.

        Returns:
            dict: A dictionary with key ``'feats'`` mapping to a float32 tensor
                  of shape ``(num_atoms, feature_dim)``.
        """
        feats = []

        atom_features = utils.ConcatFeaturizer([
            utils.atom_type_one_hot,
            utils.atomic_number_one_hot,
            utils.atom_degree_one_hot,
            utils.atom_explicit_valence_one_hot,
            utils.atom_formal_charge_one_hot,
            utils.atom_num_radical_electrons_one_hot,
            utils.atom_hybridization_one_hot,
            utils.atom_is_aromatic_one_hot
        ])

        for atom in mol.GetAtoms():
            feats.append(atom_features(atom))
        return {'feats': torch.tensor(feats).float()}
