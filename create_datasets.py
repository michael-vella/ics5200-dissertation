import logging

from dotenv import load_dotenv

from datasets.enums import FeatureType, DatasetSource
from datasets.dataset_handler import DatasetHandler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

handler = DatasetHandler()
df = handler.create_dataset(dataset_source=DatasetSource.MUV, feature_type=FeatureType.ECFP, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.MUV, feature_type=FeatureType.PYG_WITH_BONDS, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.MUV, feature_type=FeatureType.CHEMBERTA, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.MUV, feature_type=FeatureType.MOLFORMER, force_refresh=False)

df = handler.create_dataset(dataset_source=DatasetSource.TOX21, feature_type=FeatureType.ECFP, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.TOX21, feature_type=FeatureType.PYG_WITH_BONDS, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.TOX21, feature_type=FeatureType.CHEMBERTA, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.TOX21, feature_type=FeatureType.MOLFORMER, force_refresh=False)

df = handler.create_dataset(dataset_source=DatasetSource.DUDE_GPCR, feature_type=FeatureType.ECFP, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.DUDE_GPCR, feature_type=FeatureType.PYG_WITH_BONDS, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.DUDE_GPCR, feature_type=FeatureType.CHEMBERTA, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.DUDE_GPCR, feature_type=FeatureType.MOLFORMER, force_refresh=False)

df = handler.create_dataset(dataset_source=DatasetSource.LIT_PCBA, feature_type=FeatureType.ECFP, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.LIT_PCBA, feature_type=FeatureType.PYG_WITH_BONDS, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.LIT_PCBA, feature_type=FeatureType.CHEMBERTA, force_refresh=False)
df = handler.create_dataset(dataset_source=DatasetSource.LIT_PCBA, feature_type=FeatureType.MOLFORMER, force_refresh=False)