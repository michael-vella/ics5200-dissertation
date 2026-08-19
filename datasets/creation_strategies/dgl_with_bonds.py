import logging
from typing import override

import dgl
import torch
import pandas as pd
from rdkit import Chem
from dgllife import utils
from chembl_structure_pipeline import standardizer

from .dgl import DGLDatasetCreator


class DGLBondsDatasetCreator(DGLDatasetCreator):
    """
    Dataset creation strategy that represents molecules as DGL graphs
    with both atom-level node features and bond-level edge features.

    Each molecule's SMILES string is parsed and standardised before being
    converted into a bidirectional DGL graph (bigraph), where:
      - Nodes represent atoms, featurised via ``featurize_atoms``.
      - Edges represent bonds, featurised via ``featurize_bonds``.
      - Self-loops are added to every node.

    This dataset creation strategy is an extension of the `DGLDatasetCreator`.
    """
    def __init__(self) -> None:
        """
        Initialises the DGLBondsDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates a DGL graph dataset from a DataFrame containing SMILES strings.

        Each SMILES string is standardised using the ChEMBL structure pipeline
        and then converted into a bidirectional DGL graph (bigraph) carrying both
        atom-level node features and bond-level edge features. Self-loops are added
        to every node. The resulting graph is stored as a new 'mol' column in the
        DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                               SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                          containing the DGL graph for each molecule.
        """
        self._logger.info("Featurising SMILES strings into DGL graphs with bonds. Please wait, this process might take a while")
        df['mol'] = df['smiles'].apply(self._create_features)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    @override
    def _create_features(self, smiles: str) -> dgl.DGLGraph:
        """
        Converts a SMILES string into a DGL bidirectional graph with atom and bond features.

        The molecule is first standardised via the ChEMBL structure pipeline to
        ensure consistent representations, then converted to a bigraph in which each
        bond yields a pair of directed edges. Self-loops are appended so that each
        atom attends to itself during message passing.

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            dgl.DGLGraph: A DGL graph with node feature key ``'feats'`` containing
                          concatenated one-hot atom descriptors, and edge feature key
                          ``'edge_feats'`` containing concatenated one-hot bond descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        mol = standardizer.standardize_mol(mol)

        dgl_graph = utils.mol_to_bigraph(
            mol=mol,
            node_featurizer=self._featurise_atoms,
            edge_featurizer=self._featurise_bonds,
            canonical_atom_order=True
        )

        return dgl.add_self_loop(dgl_graph)

    def _featurise_bonds(self, mol) -> dict:
        """
        Computes concatenated one-hot bond features for all bonds in a molecule.

        The following per-bond properties are encoded as one-hot vectors and
        concatenated into a single feature vector:
          - Bond type (single, double, triple, aromatic)
          - Conjugation flag
          - Ring membership flag
          - Stereochemistry configuration
          - Bond direction

        Each bond is featurised twice, once for each of the two directed edges that
        represent it in the bidirectional graph.

        Args:
            mol: An RDKit ``Mol`` object whose bonds are to be featurised.

        Returns:
            dict: A dictionary with key ``'edge_feats'`` mapping to a float32 tensor
                  of shape ``(2 * num_bonds, feature_dim)``.
        """
        feats = []

        bond_features = utils.ConcatFeaturizer([
            utils.bond_type_one_hot,
            utils.bond_is_conjugated_one_hot,
            utils.bond_is_in_ring_one_hot,
            utils.bond_stereo_one_hot,
            utils.bond_direction_one_hot,
        ])

        for bond in mol.GetBonds():
            feats.append(bond_features(bond))
            feats.append(bond_features(bond))
        return {'edge_feats': torch.tensor(feats).float()}
