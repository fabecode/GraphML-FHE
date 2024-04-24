from xgb import xgb_pipeline, xgb_batch_pipeline, xgb_experiment_n_estimators, xgb_experiment_max_depth, xgb_experiment_n_bits
from util import logger_setup
from gfp import gfp_enrichment
import pandas as pd
import logging
import time

if __name__ == '__main__':
    # Setup logging
    logger_setup()

    #XGB
    logging.info("Running XGB pipeline (without GFP enrichment)")
    df = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"XGB-{time.strftime('%Y-%m-%d-%H-%M')}-simulate"
    xgb_pipeline(df, run_name, gfp=False, fhe_mode="simulate")

    #GFP + XGB - run in batches
    logging.info("Running XGB pipeline (with GFP enrichment)")
    df_enriched = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"GFP-XGB-Balanced-{time.strftime('%Y-%m-%d-%H-%M')}-vertex-stats-execute"
    xgb_batch_pipeline(df_enriched, run_name, fhe_mode="execute", batch_size=128)

    #GFP + XGB on smaller dataset
    logging.info("Running XGB pipeline (GFP enrichment)")
    df_enriched = pd.read_csv("../data/HI-Small_Sampled_5491.csv")
    run_name = f"GFP-XGB-Balanced-{time.strftime('%Y-%m-%d-%H-%M')}-sample-5491-vertex-stats-execute"
    xgb_batch_pipeline(df_enriched, run_name, gfp=True, fhe_mode="execute", batch_size=128)


    ############################# EXPERIMENTS #############################
    # Running XGB experiments for different number of estimators
    logging.info("Running XGB experiments for different number of estimators")
    df = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"XGB-Experiment-{time.strftime('%Y-%m-%d-%H-%M')}-n_estimators-execute"
    xgb_experiment_n_estimators(df, run_name, gfp=False, fhe_mode="execute")


    # Running XGB experiments for different max depth
    logging.info("Running XGB experiments for different max depth")
    df = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"XGB-Experiment-{time.strftime('%Y-%m-%d-%H-%M')}-max_depth-execute"
    xgb_experiment_max_depth(df, run_name, fhe_mode="execute")


    # Running XGB experiments for different number of bits
    logging.info("Running XGB experiments for different number of bits")
    df = pd.read_csv("../data/HI-Small_Balanced_Formatted.csv")
    run_name = f"XGB-Experiment-{time.strftime('%Y-%m-%d-%H-%M')}-n_bits-8_16-execute"
    xgb_experiment_n_bits(df, run_name, fhe_mode="execute")