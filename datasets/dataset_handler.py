import logging
from pathlib import Path

import pandas as pd

from constants import DATA_RAW_DIR, DATA_PROCESSED_DIR
from datasets.enums import FeatureType, DatasetSource
from datasets.creation_strategies.base_strategy import DatasetCreationStrategy
from datasets.creation_strategies.ecfp import ECFPDatasetCreator
from datasets.creation_strategies.dgl import DGLDatasetCreator
from datasets.creation_strategies.dgl_with_bonds import DGLBondsDatasetCreator
from datasets.train_test_strategies.base_strategy import TrainTestStrategy, TrainTestData
from datasets.train_test_strategies.ecfp import ECFPTrainTestStrategy
from datasets.train_test_strategies.dgl import DGLTrainTestStrategy
from datasets.train_test_strategies.dgl_with_bonds import DGLBondsTrainTestStrategy

# maps each feature type to the dataset creation strategy responsible for it
_DATASET_CREATION_STRATEGIES: dict[FeatureType, type[DatasetCreationStrategy]] = {
    FeatureType.ECFP: ECFPDatasetCreator,
    FeatureType.DGL: DGLDatasetCreator,
    FeatureType.DGL_WITH_BONDS: DGLBondsDatasetCreator,
}

# maps each feature type to the train test retrieval method
_TRAIN_TEST_STRATEGIES: dict[FeatureType, type[TrainTestStrategy]] = {
    FeatureType.ECFP: ECFPTrainTestStrategy,
    FeatureType.DGL: DGLTrainTestStrategy,
    FeatureType.DGL_WITH_BONDS: DGLBondsTrainTestStrategy,
}


class DatasetHandler:
    """
    Central interface for any dataset handling related processes.

    Provides a single entry point for loading raw datasets from disk and for
    creating processed datasets using the various dataset creation strategies
    (ECFP, DGL and DGL with bonds).
    """
    def __init__(self) -> None:
        """
        Initialises the DatasetHandler by setting up a logger instance.
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initialised DatasetHandler class")

    def load_raw_dataset(self, dataset_source: DatasetSource) -> pd.DataFrame:
        """
        Loads a raw dataset from a CSV file into a pandas DataFrame.

        Args:
            dataset_source (DatasetSource): The dataset source, used to resolve
                the raw CSV file path.

        Returns:
            pd.DataFrame: The raw dataset loaded from the CSV file.
        """
        raw_path = self.__get_raw_dataset_path(dataset_source=dataset_source)
        self._logger.info(f"Raw dataset path: '{raw_path}'")

        self._logger.info("Loading dataset into a pandas DataFrame")
        pdf: pd.DataFrame = pd.read_csv(raw_path)

        return pdf

    def load_processed_dataset(self, dataset_source: DatasetSource, feature_type: FeatureType) -> pd.DataFrame:
            """
            Loads a processed dataset from a PKL file into a pandas DataFrame.
    
            Args:
                dataset_source (DatasetSource): The dataset source, used to resolve
                    the pro CSV file path.
                feature_type (FeatureType): Feature representation to generate.
                    Accepts a FeatureType member.
    
            Returns:
                pd.DataFrame: The processed dataset loaded from the PKL file.
            """
            processed_path = self.__get_processed_dataset_path(dataset_source=dataset_source, feature_type=feature_type)
            self._logger.info(f"Processed dataset path: '{processed_path}'")
    
            self._logger.info("Loading dataset into a pandas DataFrame")
            pdf: pd.DataFrame = pd.read_pickle(processed_path)
    
            return pdf

    def load_train_test_set(self, dataset_source: DatasetSource, feature_type: FeatureType) -> TrainTestData:
        """
        Loads a processed dataset and splits it into per-assay train and test sets.

        A fixed set of assays is held out as the test set for each dataset
        source, with the remaining assays (excluding identifier and structure
        columns) used for training. The appropriate train test strategy for the
        requested feature type is then used to build a per-assay DataFrame
        (target and features, with missing values dropped) for each split.

        Args:
            dataset_source (DatasetSource): The dataset source, used to resolve
                the processed dataset path and its held-out test assays.
            feature_type (FeatureType): Feature representation of the processed
                dataset. Accepts a FeatureType member.

        Returns:
            TrainTestData: A tuple of (train assays dict, test assays dict),
                each mapping an assay name to a DataFrame with 'y' and 'mol' columns.
        """
        test_assays_map = {
            DatasetSource.TOX21: ["SR-HSE", "SR-MMP", "SR-p53"],
            DatasetSource.MUV: ["MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"],
        }

        pdf = self.load_processed_dataset(dataset_source=dataset_source, feature_type=feature_type)

        test_assays = test_assays_map.get(dataset_source)
        self._logger.info(f"Test assays for the '{dataset_source.value}' dataset source: {test_assays}")

        non_train_assays = test_assays.copy()
        non_train_assays.extend(["mol_id", "smiles", "mol"])
        train_assays = [x for x in list(pdf.columns) if x not in non_train_assays]
        self._logger.info(f"Train assays for the '{dataset_source.value}' dataset source: {train_assays}")

        train_test_creator = _TRAIN_TEST_STRATEGIES[feature_type]()

        return train_test_creator.get_train_test(
            pdf=pdf,
            train_assays=train_assays,
            test_assays=test_assays
        )

    def create_dataset(self, dataset_source: DatasetSource, feature_type: FeatureType, force_refresh: bool) -> None:
        """
        Creates a processed dataset for a given source and feature type.

        Loads the raw dataset, selects the appropriate dataset creation strategy
        based on the requested feature type, and persists the resulting dataset
        as a pickle file. If a processed dataset already exists, creation is
        skipped unless a refresh is forced.

        Args:
            dataset_source (DatasetSource): The dataset source, used to resolve
                the raw and processed dataset paths.
            feature_type (FeatureType): Feature representation to generate.
                Accepts a FeatureType member.
            force_refresh (bool): If True, recreates and overwrites the processed
                dataset even if it already exists.

        Raises:
            ValueError: If dataset_source or feature_type does not correspond to
                a valid enum member.
        """
        try:
            dataset_source = DatasetSource(dataset_source)
        except ValueError:
            raise ValueError(
                f"Dataset source '{dataset_source}' not available. "
                f"Possible dataset sources: '{[ds.value for ds in DatasetSource]}'"
            )

        try:
            feature_type = FeatureType(feature_type)
        except ValueError:
            raise ValueError(
                f"Feature type '{feature_type}' not available. "
                f"Possible feature types: '{[ft.value for ft in FeatureType]}'"
            )

        pdf = self.load_raw_dataset(dataset_source=dataset_source)

        processed_path = self.__get_processed_dataset_path(dataset_source=dataset_source, feature_type=feature_type)
        self._logger.info(f"Processed dataset path: '{processed_path}'")

        dataset_creator: DatasetCreationStrategy = _DATASET_CREATION_STRATEGIES[feature_type]()

        if not processed_path.exists() or force_refresh:
            self._logger.info(f"Creating dataset for '{dataset_source}' source, '{feature_type}' feature type and saving to '{processed_path}' location...")
            processed_path.parent.mkdir(parents=True, exist_ok=True)

            dataset = dataset_creator.create(df=pdf)
            dataset.to_pickle(processed_path)

            self._logger.info(f"Dataset saved to '{processed_path}'")
        else:
            self._logger.info(f"Dataset already exists at '{processed_path}', skipping creation")

    def __get_raw_dataset_path(self, dataset_source: DatasetSource) -> Path:
        """
        Resolves the filesystem path to a raw dataset CSV file.

        Args:
            dataset_source (DatasetSource): The dataset source.

        Returns:
            Path: Path to the raw dataset CSV file.
        """
        self._logger.info(f"Generating (raw) path for '{dataset_source.value}' dataset")
        return Path(DATA_RAW_DIR + f"/{dataset_source.value}.csv")

    def __get_processed_dataset_path(self, dataset_source: DatasetSource, feature_type: FeatureType) -> Path:
        """
        Resolves the filesystem path to a processed dataset pickle file.

        Args:
            dataset_source (DatasetSource): The dataset source.
            feature_type (FeatureType): Feature representation of the processed dataset.

        Returns:
            Path: Path to the processed dataset pickle file.
        """
        self._logger.info(f"Generating (processed) path for '{dataset_source.value}' dataset '{feature_type.value}' feature type")
        return Path(DATA_PROCESSED_DIR + f"/{dataset_source.value}/{feature_type.value}.pkl")