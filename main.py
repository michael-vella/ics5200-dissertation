import logging

from datasets.enums import FeatureType, DatasetSource
from datasets.dataset_handler import DatasetHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

handler = DatasetHandler()
df = handler.create_dataset(dataset_source=DatasetSource.MUV, feature_type=FeatureType.ECFP, force_refresh=False)
