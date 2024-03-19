from xgb import xgb_pipeline
from util import logger_setup
from gfp import gfp_enrichment
import pandas as pd
import logging
import time

if __name__ == '__main__':
    # Setup logging
    logger_setup()

    # #XGB
    # logging.info("Running XGB pipeline (without GFP enrichment)")
    # df = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    # run_name = f"XGB-{time.strftime('%Y-%m-%d-%H-%M')}-execute"
    # xgb_pipeline(df, run_name, fhe_mode="execute")

   # GFP + XGB
    logging.info("Running XGB pipeline (with GFP enrichment)")
    df_enriched = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"GFP-XGB-Balanced-{time.strftime('%Y-%m-%d-%H-%M')}"
    xgb_pipeline(df_enriched, run_name, gfp=True)

