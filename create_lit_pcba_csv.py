import logging
import os

import pandas as pd

from constants import (
    DATA_LIT_PCBA_RAW_DIR,
    DATA_RAW_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

logging.info("Creating LIT-PCBA CSV file from raw (.smi) data")

final_df = pd.DataFrame()
assays_with_duplicates = []

for root, subdirectories, _ in os.walk(DATA_LIT_PCBA_RAW_DIR):
    for subdirectory in subdirectories:
        logging.info(f"Inside subdirectory: '{subdirectory}'")

        actives_ism_file = os.path.join(root, subdirectory) + "/actives.smi"
        decoys_ism_file = os.path.join(root, subdirectory) + "/inactives.smi"

        logging.info(f"Actives (.smi) file: '{actives_ism_file}'")
        logging.info(f"Decoys (.smi) file: '{decoys_ism_file}'")

        actives_df = pd.read_csv(actives_ism_file, sep=" ", header=None)
        decoys_df = pd.read_csv(decoys_ism_file, sep=" ", header=None)

        actives_df.columns = ["smiles", "id"]
        decoys_df.columns = ["smiles", "id"]

        actives_df = actives_df.drop(["id"], axis=1)
        decoys_df = decoys_df.drop(["id"], axis=1)

        actives_dupes = actives_df[actives_df.duplicated(keep=False)]
        decoys_dupes = decoys_df[decoys_df.duplicated(keep=False)]
        logging.info(
            f"Duplicate actives: {actives_df.duplicated().sum()} extra rows "
            f"({len(actives_dupes)} rows involved). Duplicates (top 1): "
            f"{sorted(actives_dupes['smiles'].unique())[:1]}"
        )
        logging.info(
            f"Duplicate decoys: {decoys_df.duplicated().sum()} extra rows "
            f"({len(decoys_dupes)} rows involved). Duplicates (top 1): "
            f"{sorted(decoys_dupes['smiles'].unique())[:1]}"
        )

        actives_df = actives_df.drop_duplicates()
        decoys_df = decoys_df.drop_duplicates()

        actives_df[subdirectory] = 1
        decoys_df[subdirectory] = 0

        subset_df = pd.concat([actives_df, decoys_df])

        concat_dupes = subset_df[subset_df.duplicated(subset="smiles", keep=False)]
        logging.error(f"Duplicates (top 1) after concatenating actives & decoys detected: {sorted(concat_dupes['smiles'].unique())[:1]}")

        if not concat_dupes.empty:
            assays_with_duplicates.append(subdirectory)

        if len(final_df) == 0:
            final_df = subset_df
        else:
            final_df = pd.merge(final_df, subset_df, how="outer")

logging.info(f"Assays that contain duplicates: '{assays_with_duplicates}'")
save_dir = DATA_RAW_DIR + "/lit_pcba.csv"
# logging.info(f"Saving CSV to '{save_dir}'")
# final_df.to_csv(save_dir, index=False)

# GBA DATASET:

# Duplicates inside actives.smi
# - Oc1nc2cc(c(cc2nc1O)[N+](=O)[O-])[N+](=O)[O-]
# - [O-][N+](=O)c1cc2NC(=O)C(=O)Nc2cc1[N+](=O)[O-]

# Duplicates inside inactives.smi (limit 5):
# - Brc1c(Br)c(Br)c2[nH]nnc2c1Br
# - Brc1c(NC2=[NH+]CCN2)ccc3nccnc13
# - Brc1ccc(SCCN2CC[NH+](CCc3ccccc3)CCC2=O)cc1
# - Brc1ccc(\\C=C\\C[NH2+]CCNS(=O)(=O)c2cccc3cnccc23)cc1
# - Brc1ccc2[nH]c3c(CC(=O)Nc4ccccc34)c2c1

# SMILES found in both actives.smi & inactives.smi files:
# - C[C@H](\\C=C(/C)\\C=C\\C(=O)NO)C(=O)c1ccc(cc1)N(C)C
# - Oc1ccc(CCC(=O)c2c(O)cc(O)cc2O)cc1
# - Oc1ccc(\\C=C\\C(=O)c2ccc(O)cc2O)cc1
# - Oc1ccc(cc1)C2=COc3cc(O)cc([O-])c3C2=O

# Assays that contain duplicates: '['GBA', 'FEN1', 'ESR1_ago', 'PPARG', 'ALDH1', 'ESR1_ant', 'MAPK1', 'VDR', 'KAT2A']'