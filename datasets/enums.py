from enum import StrEnum


class FeatureType(StrEnum):
    """
    Enumeration of the feature representations supported when creating datasets.
    """
    ECFP = "ecfp"
    DGL = "dgl"
    DGL_WITH_BONDS = "dgl_with_bonds"


class DatasetSource(StrEnum):
    """
    Enumeration of the raw dataset sources supported by the handler.
    """
    TOX21 = "tox21"
    MUV = "muv"
    DUDE_GPCR = "dude_gpcr"