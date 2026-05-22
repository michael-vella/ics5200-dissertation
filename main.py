import logging

from datasets.dataset_handler import DatasetHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

handler = DatasetHandler()
df = handler.create_dataset(dataset_source="muv", feature_type="ecfp", force_refresh=False)
