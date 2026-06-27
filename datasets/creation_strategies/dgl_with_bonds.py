import logging
from typing import override

import dgl
import torch
import pandas as pd
from rdkit import Chem
from dgllife import utils
from chembl_structure_pipeline import standardizer

from datasets.creation_strategies.dgl import DGLDatasetCreator


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
        Initialises the DGLDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        todo
        """
        self._logger.info("Featurising SMILES strings into DGL graphs with bonds. Please wait, this process might take a while")
        df['mol'] = df['smiles'].apply(self._create_features)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    @override
    def _create_features(self, smiles: str) -> dgl.DGLGraph:
        """
        todo
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
        todo
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