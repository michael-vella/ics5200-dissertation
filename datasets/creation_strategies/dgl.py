from typing import override

import pandas as pd

from datasets.creation_strategies.base_strategy import DatasetCreationStrategy


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
    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        todo
        """
        pass