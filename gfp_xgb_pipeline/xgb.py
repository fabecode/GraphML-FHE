from gfp import gfp_enrichment, gfp_train_test_enrichment
from data_split import split_data
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from skopt import BayesSearchCV
from concrete.ml.sklearn.xgb import XGBClassifier
import pandas as pd
import time
import logging
import wandb

def xgb_pipeline(df, run_name, fhe_mode="execute", gfp=False):
    ##################### WANDB INITIALIZATION #####################
    wandb.init(
            mode="online",
            project="GFP_XGBoost", #replace this with your wandb project name if you want to use wandb logging
            name=run_name,
    )
    logging.info(f"Wandb run ({wandb.run.name}) initialized")

    ####################### TRAIN TEST SPLIT #######################

    #Split df
    df_train, df_test = split_data(df, test_size=0.2)

    #x and y
    X_train = df_train.drop(["Is Laundering"], axis=1)
    y_train = df_train["Is Laundering"]

    X_test = df_test.drop(["Is Laundering"], axis=1)
    y_test = df_test["Is Laundering"]


    ####################### GFP ENRICHMENT #######################
    
    if gfp==True:
      #  X_train, X_test = gfp_train_test_enrichment(X_train, X_test)
        X_train = gfp_enrichment(X_train)
        X_test = gfp_enrichment(X_test)
        
        X_train.to_csv(f"../data/{run_name}_X_train.csv", index=False)
        X_test.to_csv(f"../data/{run_name}_X_test.csv", index=False)

    ####################### XGBOOST PIPELINE #######################
    logging.info("Building XGBoost pipeline")
    # Define our model
    model = XGBClassifier(n_jobs=1, n_bits=3)

    # Define the pipeline
    # We normalize the data and apply a PCA before fitting the model
    pipeline = Pipeline(
        [("standard_scaler", StandardScaler()), ("pca", PCA(random_state=0)), ("model", model)]
    )

    #################### BAYESIAN OPTIMIZATION ####################
    # Define the parameters to tune
    param_space = {
        "model__n_estimators": (5, 500),
        "model__max_depth": (2, 15),
        "model__learning_rate": (0.003, 0.1),
        "model__colsample_bytree": (0.5, 1)
    }

    bayes_search = BayesSearchCV(
    pipeline,
    param_space,
    n_iter=50,  # Number of parameter settings that are sampled
    cv=3,       # Number of cross-validation folds
    n_jobs=-1,  # Use all available cores
    scoring="accuracy",
    random_state=42,
   )

    # Launch the Bayesian optimization
    bayes_search.fit(X_train, y_train)

    # Save the best parameters found
    logging.info(f"Best XGB parameters found: {bayes_search.best_params_}")

    best_pipeline = bayes_search.best_estimator_

    data_transformation_pipeline = best_pipeline[:-1]
    model = best_pipeline[-1]

    # Transform test set
    X_test_transformed = data_transformation_pipeline.transform(X_test)
    ground_truth = y_test

    ####################### CLEAR INFERENCE #######################
    logging.info("Starting model evaluation on the test set in clear")
    # Evaluate the model on the test set in clear
    start_time_clear = time.time()
    y_pred_clear = model.predict(X_test_transformed, fhe="disable")
    end_time_clear = time.time()

    ########### CLEAR RESULTS ###########
    elapsed_time_clear = end_time_clear - start_time_clear
    accuracy_clear = (y_pred_clear == ground_truth).mean()
    f1_score_clear = f1_score(ground_truth, y_pred_clear)
    precision_clear = precision_score(ground_truth, y_pred_clear)
    recall_clear = recall_score(ground_truth, y_pred_clear)

    wandb.run.summary["time/unencrypted_eval_time"] = elapsed_time_clear
    wandb.run.summary["accuracy/unencrypted"] = accuracy_clear
    wandb.run.summary["f1_score/unencrypted"] = f1_score_clear
    wandb.run.summary["precision/unencrypted"] = precision_clear
    wandb.run.summary["recall/unencrypted"] = recall_clear
    logging.info("Prediction time for unencrypted: {:.6f}s".format(elapsed_time_clear))
    logging.info("Accuracy for unencrypted: {:.4f}".format(accuracy_clear))
    logging.info("F1 score for unencrypted: {:.4f}".format(f1_score_clear))
    logging.info("Precision for unencrypted: {:.4f}".format(precision_clear))
    logging.info("Recall for unencrypted: {:.4f}".format(recall_clear))

    ###################### FHE INFERENCE #######################
    # Compile the model to FHE
    model.compile(X_test_transformed)

    logging.info("Starting model evaluation on the test set in FHE")
    # Perform the inference in FHE and run on encrypted inputs
    start_time_fhe = time.time() 
    y_pred_fhe = model.predict(X_test_transformed, fhe=fhe_mode)
    end_time_fhe = time.time()

    ########### FHE RESULTS ###########
    elapsed_time_fhe = end_time_fhe - start_time_fhe
    accuracy_fhe = (y_pred_fhe == ground_truth).mean()
    f1_score_fhe = f1_score(ground_truth, y_pred_fhe)
    precision_fhe = precision_score(ground_truth, y_pred_fhe)
    recall_fhe = recall_score(ground_truth, y_pred_fhe)

    wandb.run.summary["time/fhe_eval_time"] = elapsed_time_fhe
    wandb.run.summary["accuracy/fhe"] = accuracy_fhe
    wandb.run.summary["f1_score/fhe"] = f1_score_fhe
    wandb.run.summary["precision/fhe"] = precision_fhe
    wandb.run.summary["recall/fhe"] = recall_fhe
    logging.info("Prediction time in FHE: {:.6f}s".format(elapsed_time_fhe))
    logging.info("Accuracy in FHE: {:.4f}".format(accuracy_fhe))
    logging.info("F1 score in FHE: {:.4f}".format(f1_score_fhe))
    logging.info("Precision in FHE: {:.4f}".format(precision_fhe))
    logging.info("Recall in FHE: {:.4f}".format(recall_fhe))

    ##################### OVERALL RESULTS #####################
    ratio_elapsed_time = elapsed_time_fhe / elapsed_time_clear
    logging.info("Prediction time of FHE / unencrypted: {:.2f}x".format(ratio_elapsed_time))
    wandb.run.summary["time/fhe_to_unencrypted_time_ratio"] = ratio_elapsed_time
    logging.info(f"Results similarity between FHE and unencrypted: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")

    wandb.finish()
