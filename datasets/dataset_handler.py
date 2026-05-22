import logging

import pandas as pd

from constants import DATA_RAW_DIR, DATA_PROCESSED_DIR
from datasets.creation_strategies.base_strategy import DatasetCreationStrategy
from datasets.creation_strategies.ecfp import ECFPDatasetCreator
from datasets.creation_strategies.dgl import DGLDatasetCreator
from datasets.creation_strategies.dgl_with_bonds import DGLBondsDatasetCreator

class DatasetHandler:
    """
    Central interface for any dataset handling related processes.

    todo
    """
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initialised DatasetHandler class")

    def load_raw_dataset(self, dataset_source: str) -> pd.DataFrame:
        """
        todo
        """
        raw_path = self.__get_raw_dataset_path(dataset_source=dataset_source)
        self._logger.info(f"Raw dataset path: '{raw_path}'")

        self._logger.info("Loading dataset into a pandas DataFrame")
        pdf: pd.DataFrame = pd.read_csv(raw_path)

        return pdf

    def create_dataset(self, dataset_source: str, feature_type: str, force_refresh: bool) -> None:
        """
        todo
        """
        ecfp_feature_type = "ecfp"
        dgl_feature_type = "dgl"
        dgl_with_bonds_feature_type = "dgl_with_bonds"
        possible_feature_types = {ecfp_feature_type, dgl_feature_type, dgl_with_bonds_feature_type}
        assert feature_type in possible_feature_types, f"Feature type '{feature_type}' not available. Possible feature types: '{possible_feature_types}'"

        pdf = self.load_raw_dataset(dataset_source=dataset_source)

        processed_path = self.__get_processed_dataset_path(dataset_source=dataset_source, feature_type=feature_type)
        self._logger.info(f"Processed dataset path: '{processed_path}'")

        # by default we use the ECFP dataset creation strategy
        # if a different `feature_type` that is no 'ecfp' is injected
        # the dataset creation strategy is overridden
        dataset_creator: DatasetCreationStrategy = ECFPDatasetCreator()
        if feature_type == dgl_feature_type:
            dataset_creator = DGLDatasetCreator()
        if feature_type == dgl_with_bonds_feature_type:
            dataset_creator = DGLBondsDatasetCreator()

        # todo:
        # check if dataset already exists or (fore_refresh is not true)
        # if it does not exists, create folder directory + dataset pkl file
        # log success (or that dataset already exists)

        # dataset_creator.create(df=pdf)
        # self._logger.info("Succe")

    def __get_raw_dataset_path(self, dataset_source: str) -> str:
        """
        todo
        """
        self._logger.info(f"Generating (raw) path for '{dataset_source}' dataset")
        return DATA_RAW_DIR + f"/{dataset_source}.csv"

    def __get_processed_dataset_path(self, dataset_source: str, feature_type: str) -> str:
        """
        todo
        """
        self._logger.info(f"Generating (processed) path for '{dataset_source}' dataset '{feature_type}' feature type")
        return DATA_PROCESSED_DIR + f"/{dataset_source}/{feature_type}.pkl"