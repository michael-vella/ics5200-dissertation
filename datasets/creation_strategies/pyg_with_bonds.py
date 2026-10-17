import logging
from typing import Any, override

import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.data import Data
from chembl_structure_pipeline import standardizer

from .base_strategy import DatasetCreationStrategy

# allowable sets for the one-hot atom features (mirroring the dgllife defaults
# used by the deprecated DGL strategies, so that the feature vectors are identical)
_ATOM_TYPES = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al',
    'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
    'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'
]
_ATOMIC_NUMBERS = list(range(1, 101))
_ATOM_DEGREES = list(range(11))
_ATOM_EXPLICIT_VALENCES = list(range(1, 7))
_ATOM_FORMAL_CHARGES = list(range(-2, 3))
_ATOM_NUM_RADICAL_ELECTRONS = list(range(5))
_ATOM_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
_BOOLEANS = [False, True]

# allowable sets for the one-hot bond features (idem)
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS
]
_BOND_DIRECTIONS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

# 43 + 100 + 11 + 6 + 5 + 5 + 5 + 2
ATOM_FEATURE_DIM = 177
# 4 + 2 + 2 + 6 + 3
BOND_FEATURE_DIM = 17


class PyGBondsDatasetCreator(DatasetCreationStrategy):
    """
    Dataset creation strategy that represents molecules as PyTorch Geometric
    graphs with both atom-level node features and bond-level edge features.

    Each molecule's SMILES string is parsed and standardised before being
    converted into a bidirectional PyG graph, where:
      - Nodes represent atoms, featurised via ``_featurise_atoms``.
      - Edges represent bonds, featurised via ``_featurise_bonds``.
      - Self-loops are added to every node.

    This dataset creation strategy is the PyG equivalent of the deprecated
    `DGLBondsDatasetCreator`, producing the same node features, edge features,
    atom ordering and edge ordering.
    """
    def __init__(self) -> None:
        """
        Initialises the PyGBondsDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates a PyG graph dataset from a DataFrame containing SMILES strings.

        Each SMILES string is standardised using the ChEMBL structure pipeline
        and then converted into a bidirectional PyG graph carrying both atom-level
        node features and bond-level edge features. Self-loops are added to every
        node. The resulting graph is stored as a new 'mol' column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                               SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                          containing the PyG graph for each molecule. Molecules whose
                          SMILES string could not be parsed are dropped.
        """
        self._logger.info("Featurising SMILES strings into PyG graphs with bonds. Please wait, this process might take a while")
        df['mol'] = df['smiles'].apply(self._create_features)
        df = self._drop_unfeaturised(df)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    def _create_features(self, smiles: str) -> Data | None:
        """
        Converts a SMILES string into a PyG bidirectional graph with atom and bond features.

        The molecule is first standardised via the ChEMBL structure pipeline to
        ensure consistent representations, and its atoms are renumbered into a
        canonical order. Each bond yields a pair of directed edges (u -> v followed
        by v -> u). Self-loops are appended after all bond edges so that each atom
        attends to itself during message passing; as with ``dgl.add_self_loop``, the
        self-loop edge features are filled with ones.

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            Data | None: A PyG graph with ``x`` containing concatenated one-hot atom
                         descriptors of shape ``(num_atoms, 177)``, ``edge_index`` of
                         shape ``(2, 2 * num_bonds + num_atoms)`` and ``edge_attr``
                         containing concatenated one-hot bond descriptors of shape
                         ``(2 * num_bonds + num_atoms, 17)``. Returns None if the
                         SMILES string could not be parsed.
        """
        mol = self._parse_smiles(smiles)
        if mol is None:
            return None

        mol = standardizer.standardize_mol(mol)

        # same canonical renumbering as dgllife's `mol_to_bigraph(canonical_atom_order=True)`
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)

        num_atoms = mol.GetNumAtoms()

        src, dst = [], []
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([u, v])
            dst.extend([v, u])

        self_loops = list(range(num_atoms))
        edge_index = torch.tensor([src + self_loops, dst + self_loops], dtype=torch.long)

        edge_attr = torch.cat([
            self._featurise_bonds(mol),
            torch.ones((num_atoms, BOND_FEATURE_DIM), dtype=torch.float32)
        ])

        return Data(
            x=self._featurise_atoms(mol),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms
        )

    def _featurise_atoms(self, mol: Chem.Mol) -> torch.Tensor:
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
            mol (Chem.Mol): An RDKit ``Mol`` object whose atoms are to be featurised.

        Returns:
            torch.Tensor: A float32 tensor of shape ``(num_atoms, 177)``.
        """
        feats = []

        for atom in mol.GetAtoms():
            feats.append(
                self._one_hot(atom.GetSymbol(), _ATOM_TYPES)
                + self._one_hot(atom.GetAtomicNum(), _ATOMIC_NUMBERS)
                + self._one_hot(atom.GetDegree(), _ATOM_DEGREES)
                + self._one_hot(atom.GetValence(Chem.ValenceType.EXPLICIT), _ATOM_EXPLICIT_VALENCES)
                + self._one_hot(atom.GetFormalCharge(), _ATOM_FORMAL_CHARGES)
                + self._one_hot(atom.GetNumRadicalElectrons(), _ATOM_NUM_RADICAL_ELECTRONS)
                + self._one_hot(atom.GetHybridization(), _ATOM_HYBRIDIZATIONS)
                + self._one_hot(atom.GetIsAromatic(), _BOOLEANS)
            )

        return torch.tensor(feats, dtype=torch.float32).reshape(-1, ATOM_FEATURE_DIM)

    def _featurise_bonds(self, mol: Chem.Mol) -> torch.Tensor:
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
            mol (Chem.Mol): An RDKit ``Mol`` object whose bonds are to be featurised.

        Returns:
            torch.Tensor: A float32 tensor of shape ``(2 * num_bonds, 17)``.
        """
        feats = []

        for bond in mol.GetBonds():
            bond_feats = (
                self._one_hot(bond.GetBondType(), _BOND_TYPES)
                + self._one_hot(bond.GetIsConjugated(), _BOOLEANS)
                + self._one_hot(bond.IsInRing(), _BOOLEANS)
                + self._one_hot(bond.GetStereo(), _BOND_STEREOS)
                + self._one_hot(bond.GetBondDir(), _BOND_DIRECTIONS)
            )
            feats.append(bond_feats)
            feats.append(bond_feats)

        return torch.tensor(feats, dtype=torch.float32).reshape(-1, BOND_FEATURE_DIM)

    @staticmethod
    def _one_hot(value: Any, allowable_set: list) -> list[bool]:
        """
        One-hot encodes a value against an allowable set.

        Values outside the allowable set are encoded as an all-zero vector (no
        "unknown" slot), matching dgllife's default behaviour.

        Args:
            value (Any): The value to encode.
            allowable_set (list): The set of values to encode against.

        Returns:
            list[bool]: The one-hot encoding of the value.
        """
        return [value == s for s in allowable_set]
