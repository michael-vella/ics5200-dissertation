import logging
import os

import pandas as pd

from constants import (
    DATA_LIT_PCBA_RAW_DIR,
    DATA_RAW_DIR,
    LOGS_DIR
)

os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, "create_lit_pcba_csv.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler()
    ]
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

        # Select distinct within actives and within decoys, logging the
        # duplicates (and their counts) that were collapsed.
        actives_dupe_smiles = sorted(
            actives_df[actives_df.duplicated()]["smiles"].unique()
        )
        decoys_dupe_smiles = sorted(
            decoys_df[decoys_df.duplicated()]["smiles"].unique()
        )
        logging.info(
            f"[{subdirectory}] Duplicate actives: "
            f"{actives_df.duplicated().sum()} extra rows across "
            f"{len(actives_dupe_smiles)} distinct SMILES"
        )
        # logging.info(f"Duplicates: {actives_dupe_smiles}")
        logging.info(
            f"[{subdirectory}] Duplicate decoys: "
            f"{decoys_df.duplicated().sum()} extra rows across "
            f"{len(decoys_dupe_smiles)} distinct SMILES"
        )
        # logging.info(f"Duplicates: {decoys_dupe_smiles}")

        actives_df = actives_df.drop_duplicates()
        decoys_df = decoys_df.drop_duplicates()

        actives_df[subdirectory] = 1
        decoys_df[subdirectory] = 0

        subset_df = pd.concat([actives_df, decoys_df])

        # A SMILES present in both actives and decoys is contradictory, so
        # drop both occurrences entirely (keep=False removes all of them).
        overlap_smiles = sorted(
            subset_df[subset_df.duplicated(subset="smiles", keep=False)][
                "smiles"
            ].unique()
        )
        if overlap_smiles:
            assays_with_duplicates.append(subdirectory)
            logging.warning(
                f"[{subdirectory}] SMILES found in BOTH actives & decoys: "
                f"{len(overlap_smiles)} distinct SMILES dropped from both "
                f"({2 * len(overlap_smiles)} rows removed)"
            )
            # logging.warning(f"SMILES removed: {overlap_smiles}")
            subset_df = subset_df.drop_duplicates(subset="smiles", keep=False)
        else:
            logging.info(
                f"[{subdirectory}] No SMILES overlap between actives & decoys"
            )

        if len(final_df) == 0:
            final_df = subset_df
        else:
            final_df = pd.merge(final_df, subset_df, how="outer")

logging.info(
    f"Assays that contained actives/decoys overlap: "
    f"'{assays_with_duplicates}'"
)
save_dir = DATA_RAW_DIR + "/lit_pcba.csv"
logging.info(f"Saving CSV to '{save_dir}'")
final_df.to_csv(save_dir, index=False)
