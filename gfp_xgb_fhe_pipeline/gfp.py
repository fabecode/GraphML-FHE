from snapml import GraphFeaturePreprocessor
import numpy as np
import time
import json
import logging
import pandas as pd
from IPython.display import display
import wandb

######################### PARAMETERS #########################
params = {
    "num_threads": 4,             # number of software threads to be used (important for performance)
    "time_window": 86400,            # time window used if no pattern was specified
    
    "vertex_stats": True,         # produce vertex statistics
    "vertex_stats_cols": [4, 5],     # produce vertex statistics using the selected input columns
    
    # features: 0:fan,1:deg,2:ratio,3:avg,4:sum,5:min,6:max,7:median,8:var,9:skew,10:kurtosis
   "vertex_stats_feats": [4, 8, 9],  # sum, var, skew
    
    # fan in/out parameters
    "fan": True,
    "fan_tw": 86400, #1 day
    "fan_bins": [2, 4],
    
    # in/out degree parameters
    "degree": True,
    "degree_tw": 86400, 
    "degree_bins": [2, 4],
    
    # scatter gather parameters
    "scatter-gather": True,
    "scatter-gather_tw": 86400,
    "scatter-gather_bins": [2, 4],
    
    # temporal cycle parameters
    "temp-cycle": True,
    "temp-cycle_tw": 86400,
    "temp-cycle_bins": [2, 4],
    
    # length-constrained simple cycle parameters
    "lc-cycle": True,
    "lc-cycle_tw": 86400,
    "lc-cycle_len": 8,
    "lc-cycle_bins": [2, 4],
}

######################### FUNCTIONS #########################

def gfp_train_test_enrichment(X_train_df, X_test_df):

    X_train_simple = X_train_df[["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp", "Amount Sent", "Amount Received"]]
    X_test_simple = X_test_df[["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp", "Amount Sent", "Amount Received"]]

    colnames_original = X_train_simple .columns.tolist()

    #convert df to numpy
    X_train = X_train_simple.to_numpy()
    X_test = X_test_simple.to_numpy()

    # Create a Graph Feature Preprocessor, set its configuration using the above dictionary and verify it
    logging.info(f"Creating a graph feature preprocessor with {params}")
    wandb.config.update(params)
    gp = GraphFeaturePreprocessor()
    gp.set_params(params)

    print("Enriching the transactions with new graph features ")
    X_train_enriched = gp.fit(X_train.astype("float64")) 
    X_train_enriched = gp.transform(X_train.astype("float64"))
    X_test_enriched = gp.transform(X_test.astype("float64"))

    logging.info(f"Simple X_train shape: {X_train.shape}")
    logging.info(f"Enriched X_train dataset shape: {X_train_enriched.shape}")
    logging.info(f"Simple X_test shape: {X_test.shape}")
    logging.info(f"Enriched X_test dataset shape: {X_test_enriched.shape}")

    colnames = generate_enriched_df_colnames(gp.get_params(), colnames_original)
    X_train_enriched = pd.DataFrame(X_train_enriched, columns=colnames)
    X_test_enriched = pd.DataFrame(X_test_enriched, columns=colnames)

    #drop EdgeID, SourceAccountId, TargetAccountId, Timestamp from df
    X_train_enriched = X_train_enriched.drop(["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp"], axis=1)
    X_test_enriched = X_test_enriched.drop(["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp"], axis=1)

    #concat with original df
    X_train_enriched = pd.concat([X_train_df, X_train_enriched], axis=1)
    X_test_enriched = pd.concat([X_test_df, X_test_enriched], axis=1)

    return X_train_enriched, X_test_enriched

def gfp_enrichment(df):
    df_simple = df[["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp", "Amount Sent", "Amount Received"]]

    colnames_original = ["EdgeID", "SourceAccountId", "TargetAccountId", "Timestamp", "Amount Sent", "Amount Received"]

    #convert df to numpy
    data = df_simple.to_numpy()

    # Create a Graph Feature Preprocessor, set its configuration using the above dictionary and verify it
    logging.info(f"Creating a graph feature preprocessor with {params}")
    gp = GraphFeaturePreprocessor()
    gp.set_params(params)


    print("Enriching the transactions with new graph features ")
    # the fit_transform and transform functions are equivalent
    # these functions can run on single transactions or on batches of transactions
    data = np.ascontiguousarray(data)
    data_enriched = gp.fit_transform(data.astype("float64")) 

    logging.info(f"Raw dataset shape: {data.shape}")
    logging.info(f"Enriched dataset shape: {data_enriched.shape}")

    colnames_generated = generate_enriched_df_colnames(gp.get_params(), colnames_original)
    df_enriched = pd.DataFrame(data_enriched, columns=colnames_generated)

    return df_enriched

def generate_enriched_df_colnames(params, colnames):
        '''
        Input: 
        - transaction: enriched data with graph features (in the form of a numpy array)
        - params: dictionary with the configuration parameters of the Graph Feature Preprocessor
        - colnames: list of column names of the enriched data
        '''
        
        # add features names for the graph patterns
        for pattern in ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle']:
            if pattern in params:
                if params[pattern]: #if the pattern is enabled
                    bins = len(params[pattern +'_bins'])
                    # construct column names based on pattern type and bin ranges
                    if pattern in ['fan', 'degree']:
                        for i in range(bins-1):
                            colnames.append(pattern+"_in_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                        colnames.append(pattern+"_in_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")
                        for i in range(bins-1):
                            colnames.append(pattern+"_out_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                        colnames.append(pattern+"_out_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")
                    else:
                        for i in range(bins-1):
                            colnames.append(pattern+"_bins_"+str(params[pattern +'_bins'][i])+"-"+str(params[pattern +'_bins'][i+1]))
                        colnames.append(pattern+"_bins_"+str(params[pattern +'_bins'][i+1])+"-inf")

        vert_feat_names = ["fan","deg","ratio","avg","sum","min","max","median","var","skew","kurtosis"]

        # add features names for the vertex statistics
        if params['vertex_stats'] == True:
            for orig in ['source', 'dest']:
                for direction in ['out', 'in']:
                    # add fan, deg, and ratio features
                    for k in [0, 1, 2]:
                        if k in params["vertex_stats_feats"]:
                            feat_name = orig + "_" + vert_feat_names[k] + "_" + direction
                            colnames.append(feat_name)
                    for col in params["vertex_stats_cols"]:
                        # add avg, sum, min, max, median, var, skew, and kurtosis features
                        for k in [3, 4, 5, 6, 7, 8, 9, 10]:
                            if k in params["vertex_stats_feats"]:
                                feat_name = orig + "_" + vert_feat_names[k] + "_col" + str(col) + "_" + direction
                                colnames.append(feat_name)

        return colnames
