import logging
from typing import override

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer

from .base_strategy import DatasetCreationStrategy

# MLM-pretrained RoBERTa checkpoint; its pooler head is untrained, so molecule
# embeddings are obtained by mean-pooling the token embeddings (see `_embed`)
_MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
# number of molecules featurised per forward pass through the transformer
_BATCH_SIZE = 64


class ChemBERTaDatasetCreator(DatasetCreationStrategy):
    """
    Dataset creation strategy that represents molecules as ChemBERTa embeddings.

    ChemBERTa is a RoBERTa-style transformer pretrained on SMILES strings via
    masked language modelling. Each molecule's SMILES string is standardised and
    canonicalised before being tokenised and passed through the model. As the
    masked-language-model checkpoint has an untrained pooler head, a single
    fixed-length embedding is produced per molecule by mean-pooling the token
    embeddings (ignoring padding), mirroring `eda/chemberta_experiment.py`.
    """
    def __init__(self) -> None:
        """
        Initialises the ChemBERTaDatasetCreator class.
        """
        self._logger = logging.getLogger(__name__)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @override
    def create(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Creates a ChemBERTa-embedding dataset from a DataFrame of SMILES strings.

        Each SMILES string is standardised and canonicalised, tokenised in
        batches, and passed through the ChemBERTa model. The mean-pooled token
        embedding is stored as a new 'mol' column in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'smiles' column with
                               SMILES strings representing molecular structures.
            **kwargs: Additional keyword arguments required by the base class interface.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'mol' column
                          containing the ChemBERTa embedding as a NumPy float32
                          array. Molecules whose SMILES string could not be parsed
                          are dropped.
        """
        self._logger.info(f"Loading ChemBERTa model '{_MODEL_NAME}' onto device '{self._device}'")
        model = AutoModel.from_pretrained(_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model.to(self._device)
        model.eval()

        self._logger.info("Standardising and canonicalising SMILES strings")
        canonical_smiles = df['smiles'].apply(self._canonicalise_smiles).tolist()

        self._logger.info("Featurising SMILES strings into ChemBERTa embeddings. Please wait, this process might take a while")
        df['mol'] = self._embed(canonical_smiles, model, tokenizer)
        df = self._drop_unfeaturised(df)

        self._logger.info(f"Featurisation complete. '{len(df)}' molecules processed")
        return df

    def _canonicalise_smiles(self, smiles: str) -> str | None:
        """
        Standardises a SMILES string and returns its canonical form.

        The SMILES string is parsed and standardised via the shared base-class
        pipeline, then serialised back to a canonical SMILES string so that the
        same molecule is always presented to the model in a single, consistent
        way.

        Args:
            smiles (str): A SMILES string representing a molecular structure.

        Returns:
            str | None: The canonical SMILES string, or None if the input SMILES
                string could not be parsed.
        """
        mol = self._parse_smiles(smiles)
        if mol is None:
            return None

        return Chem.MolToSmiles(mol)

    def _embed(self, smiles_list: list[str | None], model, tokenizer) -> list[np.ndarray | None]:
        """
        Computes mean-pooled ChemBERTa embeddings for a list of SMILES strings.

        Valid SMILES strings are embedded in batches for efficiency; missing
        (non-string) entries, i.e. molecules that could not be parsed, are
        skipped and returned as None so that the calling strategy can drop them.
        For each batch, token embeddings are mean-pooled over the sequence
        dimension, ignoring padding tokens via the attention mask.

        Args:
            smiles_list (list[str | None]): Canonical SMILES strings, with a
                missing value for molecules that could not be parsed.
            model: The loaded ChemBERTa model, already placed on the target device
                and in evaluation mode.
            tokenizer: The tokenizer corresponding to the ChemBERTa model.

        Returns:
            list[np.ndarray | None]: One NumPy float32 embedding per input SMILES
                string, with None in the positions of unparseable molecules.
        """
        embeddings: list[np.ndarray | None] = [None] * len(smiles_list)
        valid_indices = [i for i, smiles in enumerate(smiles_list) if isinstance(smiles, str)]

        for start in range(0, len(valid_indices), _BATCH_SIZE):
            batch_indices = valid_indices[start:start + _BATCH_SIZE]
            batch_smiles = [smiles_list[i] for i in batch_indices]

            inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = model(**inputs)

            # mean-pool the token embeddings, ignoring padding, to get one vector per molecule
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            pooled = pooled.cpu().numpy().astype(np.float32)

            for position, index in enumerate(batch_indices):
                embeddings[index] = pooled[position]

        return embeddings
