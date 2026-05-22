from typing import override

import pandas as pd

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
    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        todo
        """
        pass