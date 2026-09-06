import logging
import os

import pandas as pd

from constants import (
    DATA_DUDE_GPCR_RAW_DIR,
    DATA_RAW_DIR
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s"
)

logging.info("Creating DUD-E GPCR CSV file from raw (.ism) data")

final_df = pd.DataFrame()

for root, subdirectories, _ in os.walk(DATA_DUDE_GPCR_RAW_DIR):
    for subdirectory in subdirectories:
        logging.info(f"Inside subdirectory: '{subdirectory}'")

        actives_ism_file = os.path.join(root, subdirectory) + "/actives_final.ism"
        decoys_ism_file = os.path.join(root, subdirectory) + "/decoys_final.ism"

        logging.info(f"Actives (.ism) file: '{actives_ism_file}'")
        logging.info(f"Decoys (.ism) file: '{decoys_ism_file}'")

        actives_df = pd.read_csv(actives_ism_file, sep=" ", header=None)
        decoys_df = pd.read_csv(decoys_ism_file, sep=" ", header=None)

        actives_df[subdirectory] = 1
        decoys_df[subdirectory] = 0

        subset_df = pd.concat([actives_df, decoys_df])
        subset_df.columns = ["smiles", "id", "chem_id", subdirectory]
        subset_df = subset_df.drop(["id", "chem_id"], axis=1)

        if len(final_df) == 0:
            final_df = subset_df
        else:
            final_df = pd.merge(final_df, subset_df, how="outer")

save_dir = DATA_RAW_DIR + "/dude_gpcr.csv"
logging.info(f"Saving CSV to '{save_dir}'")
final_df.to_csv(save_dir, index=False)