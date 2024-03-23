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

    print("Best pipeline found: ", best_pipeline)
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


def xgb_experiment_n_estimators(df, run_name, fhe_mode="execute", gfp=False):
    ##################### WANDB INITIALIZATION #####################
    wandb.init(
            mode="online",
            project="GFP_XGBoost_Experiments_2", #replace this with your wandb project name if you want to use wandb logging
            name=run_name,
    )
    try:
        logging.info(f"Wandb run ({wandb.run.name}) initialized")
        wandb.run.tags = ["n_estimators", fhe_mode, "GFP" if gfp else "No GFP"]
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

        config = {
            "n_jobs": 1,
            "n_bits": 3,
            "learning_rate": 0.07,
            "colsample_bytree": 0.98,
            "max_depth": 3
        }

        wandb.config = config
        
        # Define our model 
        model = XGBClassifier(**config)

        # Define lists to store results
        elapsed_times_fhe = []
        accuracies_fhe = []
        f1_scores_fhe = []
        precisions_fhe = []
        recalls_fhe = []
        ratio_elapsed_times = []
        n_estimators_values = [5, 10, 50, 100, 200]
        #n_estimators_values = [5] +list(range(20, 501, 20))
        wandb.run.summary["n_estimators_values"] = n_estimators_values
        logging.info("n_estimators values: {}".format(n_estimators_values))

        # Fit the model for each value of n_estimators
        for n_estimators in n_estimators_values:
            model.n_estimators = n_estimators    
            
            logging.info("")
            logging.info("n_estimator: {}".format(n_estimators))

            pipeline = Pipeline(
                [("standard_scaler", StandardScaler()), ("pca", PCA(random_state=0)), ("model", model)]
            )

            pipeline.fit(X_train, y_train)

            ground_truth = y_test

            ####################### XGBOOST PIPELINE #######################
            # Evaluate the model on the test set
            start_time_clear = time.time()
            y_pred_clear = model.predict(X_test)
            end_time_clear = time.time()

            ########### CLEAR RESULTS ###########
            elapsed_time_clear = end_time_clear - start_time_clear
            accuracy_clear = (y_pred_clear == ground_truth).mean()
            f1_score_clear = f1_score(ground_truth, y_pred_clear)
            precision_clear = precision_score(ground_truth, y_pred_clear)
            recall_clear = recall_score(ground_truth, y_pred_clear)


            # Log metrics to W&B with step=n_estimators
            wandb.log({"time/unencrypted_eval_time": elapsed_time_clear}, step=n_estimators)
            wandb.log({"accuracy/unencrypted": accuracy_clear}, step=n_estimators)
            wandb.log({"f1_score/unencrypted": f1_score_clear}, step=n_estimators)
            wandb.log({"precision/unencrypted": precision_clear}, step=n_estimators)
            wandb.log({"recall/unencrypted": recall_clear}, step=n_estimators)

            logging.info(f"Results for n_estimators={n_estimators}")
            logging.info("Prediction time for unencrypted: {:.6f}s".format(elapsed_time_clear))
            logging.info("Accuracy for unencrypted: {:.4f}".format(accuracy_clear))
            logging.info("F1 score for unencrypted: {:.4f}".format(f1_score_clear))
            logging.info("Precision for unencrypted: {:.4f}".format(precision_clear))
            logging.info("Recall for unencrypted: {:.4f}".format(recall_clear))

            ###################### FHE INFERENCE #######################
            # Compile the model to FHE
            model.compile(X_test)

            logging.info("Starting model evaluation on the test set in FHE")
            # Perform the inference in FHE and run on encrypted inputs
            start_time_fhe = time.time() 
            y_pred_fhe = model.predict(X_test, fhe=fhe_mode)
            end_time_fhe = time.time()

            ########### FHE RESULTS ###########
            elapsed_time_fhe = end_time_fhe - start_time_fhe
            accuracy_fhe = (y_pred_fhe == ground_truth).mean()
            f1_score_fhe = f1_score(ground_truth, y_pred_fhe)
            precision_fhe = precision_score(ground_truth, y_pred_fhe)
            recall_fhe = recall_score(ground_truth, y_pred_fhe)

            wandb.log({"time/fhe_eval_time": elapsed_time_fhe}, step=n_estimators)
            wandb.log({"accuracy/fhe": accuracy_fhe}, step=n_estimators)
            wandb.log({"f1_score/fhe": f1_score_fhe}, step=n_estimators)
            wandb.log({"precision/fhe": precision_fhe}, step=n_estimators)
            wandb.log({"recall/fhe": recall_fhe}, step=n_estimators)

            logging.info("Prediction time in FHE: {:.6f}s".format(elapsed_time_fhe))
            logging.info("Accuracy in FHE: {:.4f}".format(accuracy_fhe))
            logging.info("F1 score in FHE: {:.4f}".format(f1_score_fhe))
            logging.info("Precision in FHE: {:.4f}".format(precision_fhe))
            logging.info("Recall in FHE: {:.4f}".format(recall_fhe))

            ##################### OVERALL RESULTS #####################
            ratio_elapsed_time = elapsed_time_fhe / elapsed_time_clear
            logging.info("Prediction time of FHE / unencrypted: {:.2f}x".format(ratio_elapsed_time))
            wandb.log({"time/fhe_to_unencrypted_time_ratio": ratio_elapsed_time}, step=n_estimators)
            logging.info(f"Results similarity between FHE and unencrypted: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")
            wandb.log({"results_similarity": int((y_pred_fhe == y_pred_clear).mean()*100)}, step=n_estimators)

            # Append results to lists
            elapsed_times_fhe.append(elapsed_time_fhe)
            accuracies_fhe.append(accuracy_fhe)
            f1_scores_fhe.append(f1_score_fhe)
            precisions_fhe.append(precision_fhe)
            recalls_fhe.append(recall_fhe)
            ratio_elapsed_times.append(ratio_elapsed_time)

        # Log lists to W&B
        wandb.run.summary["elapsed_times_fhe"] = elapsed_times_fhe
        wandb.run.summary["accuracies_fhe"] = accuracies_fhe
        wandb.run.summary["f1_scores_fhe"] = f1_scores_fhe
        wandb.run.summary["precisions_fhe"] = precisions_fhe
        wandb.run.summary["recalls_fhe"] = recalls_fhe
        wandb.run.summary["ratio_eval_times"] = ratio_elapsed_times

        ######################### CREATING CUSTOM GRAPHS #########################
        #graph of Elapsed Time for FHE (elapsed_time_fhe) vs Number of Gradient Boosting Rounds (n_estimators)
        wandb.log({"time/fhe_eval_time_vs_n_estimators": 
                wandb.plot.line_series(xs=n_estimators_values, ys=[elapsed_times_fhe], 
                                        title="Elapsed Time for FHE vs Number of Gradient Boosting Rounds", 
                                        xname="Number of Gradient Boosting Rounds", 
                                        keys=["Elapsed Time for FHE"]
                                        )})
        
        #graph of Ratio of Elapsed Time for FHE to Unencrypted (ratio_elapsed_time) vs Number of Gradient Boosting Rounds (n_estimators)
        wandb.log({"graph/time_ratio_vs_n_estimators": 
                wandb.plot.line_series(xs=n_estimators_values, ys=[ratio_elapsed_times], 
                                        title="Ratio of FHE to Unencrypted Prediction Time against Number of Gradient Boosting Rounds", 
                                        xname="Number of Gradient Boosting Rounds", 
                                        keys=["Ratio of FHE to Unencrypted Prediction Time"]
                                        )})
        
        #graph Trade-Off between Accuracy, F1 Score, Precision and Recall vs Number of Gradient Boosting Rounds (n_estimators)
        wandb.log({"graph/fhe_metrics_vs_n_estimators": 
                wandb.plot.line_series(xs=n_estimators_values, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Performance Metrics against Number of Gradient Boosting Rounds", 
                                        xname="Number of Gradient Boosting Rounds", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})
        
        #graph Trade-off between Accuracy, F1 Score, Precision and Recall vs Elapsed Time for FHE
        wandb.log({"graph/fhe_metrics_vs_fhe_elapsed_time": 
                wandb.plot.line_series(xs=elapsed_times_fhe, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Trade-off between Performance Metrics and Prediction Time for FHE", 
                                        xname="FHE Prediction Time (s)", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})

        wandb.finish()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        wandb.finish()
        return
    
def xgb_experiment_max_depth(df, run_name, fhe_mode="execute", gfp=False):
    ##################### WANDB INITIALIZATION #####################
    wandb.init(
            mode="online",
            project="GFP_XGBoost_Experiments_2", #replace this with your wandb project name if you want to use wandb logging
            name=run_name,
    )
    logging.info(f"Wandb run ({wandb.run.name}) initialized")
    wandb.run.tags = ["max_depth", fhe_mode, "GFP" if gfp else "No GFP"]
    
    try:
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

        config = {
            "n_jobs": 1,
            "n_bits": 3,
            "learning_rate": 0.07,
            "colsample_bytree": 0.98,
            "n_estimators": 20
        }

        wandb.config = config
        
        # Define our model 
        model = XGBClassifier(**config)

        # Define lists to store results
        elapsed_times_fhe = []
        accuracies_fhe = []
        f1_scores_fhe = []
        precisions_fhe = []
        recalls_fhe = []
        ratio_elapsed_times = []
        max_depth = list(range(1,16))
        wandb.run.summary["max_depth"] = max_depth
        logging.info("max_depth: {}".format(max_depth))

        # Fit the model for each value of cur_max_depth
        for cur_max_depth in max_depth:
            model.max_depth = cur_max_depth    
            
            logging.info("")
            logging.info("cur_max_depth: {}".format(cur_max_depth))

            pipeline = Pipeline(
                [("standard_scaler", StandardScaler()), ("pca", PCA(random_state=0)), ("model", model)]
            )

            pipeline.fit(X_train, y_train)

            ground_truth = y_test

            ####################### XGBOOST PIPELINE #######################
            # Evaluate the model on the test set
            start_time_clear = time.time()
            y_pred_clear = model.predict(X_test)
            end_time_clear = time.time()

            ########### CLEAR RESULTS ###########
            elapsed_time_clear = end_time_clear - start_time_clear
            accuracy_clear = (y_pred_clear == ground_truth).mean()
            f1_score_clear = f1_score(ground_truth, y_pred_clear)
            precision_clear = precision_score(ground_truth, y_pred_clear)
            recall_clear = recall_score(ground_truth, y_pred_clear)


            # Log metrics to W&B with step=cur_max_depth
            wandb.log({"time/unencrypted_eval_time": elapsed_time_clear}, step=cur_max_depth)
            wandb.log({"accuracy/unencrypted": accuracy_clear}, step=cur_max_depth)
            wandb.log({"f1_score/unencrypted": f1_score_clear}, step=cur_max_depth)
            wandb.log({"precision/unencrypted": precision_clear}, step=cur_max_depth)
            wandb.log({"recall/unencrypted": recall_clear}, step=cur_max_depth)

            logging.info(f"Results for cur_max_depth={cur_max_depth}")
            logging.info("Prediction time for unencrypted: {:.6f}s".format(elapsed_time_clear))
            logging.info("Accuracy for unencrypted: {:.4f}".format(accuracy_clear))
            logging.info("F1 score for unencrypted: {:.4f}".format(f1_score_clear))
            logging.info("Precision for unencrypted: {:.4f}".format(precision_clear))
            logging.info("Recall for unencrypted: {:.4f}".format(recall_clear))

            ###################### FHE INFERENCE #######################
            # Compile the model to FHE
            model.compile(X_test)

            logging.info("Starting model evaluation on the test set in FHE")
            # Perform the inference in FHE and run on encrypted inputs
            start_time_fhe = time.time() 
            y_pred_fhe = model.predict(X_test, fhe=fhe_mode)
            end_time_fhe = time.time()

            ########### FHE RESULTS ###########
            elapsed_time_fhe = end_time_fhe - start_time_fhe
            accuracy_fhe = (y_pred_fhe == ground_truth).mean()
            f1_score_fhe = f1_score(ground_truth, y_pred_fhe)
            precision_fhe = precision_score(ground_truth, y_pred_fhe)
            recall_fhe = recall_score(ground_truth, y_pred_fhe)

            wandb.log({"time/fhe_eval_time": elapsed_time_fhe}, step=cur_max_depth)
            wandb.log({"accuracy/fhe": accuracy_fhe}, step=cur_max_depth)
            wandb.log({"f1_score/fhe": f1_score_fhe}, step=cur_max_depth)
            wandb.log({"precision/fhe": precision_fhe}, step=cur_max_depth)
            wandb.log({"recall/fhe": recall_fhe}, step=cur_max_depth)

            logging.info("Prediction time in FHE: {:.6f}s".format(elapsed_time_fhe))
            logging.info("Accuracy in FHE: {:.4f}".format(accuracy_fhe))
            logging.info("F1 score in FHE: {:.4f}".format(f1_score_fhe))
            logging.info("Precision in FHE: {:.4f}".format(precision_fhe))
            logging.info("Recall in FHE: {:.4f}".format(recall_fhe))

            ##################### OVERALL RESULTS #####################
            ratio_elapsed_time = elapsed_time_fhe / elapsed_time_clear
            logging.info("Prediction time of FHE / unencrypted: {:.2f}x".format(ratio_elapsed_time))
            wandb.log({"time/fhe_to_unencrypted_time_ratio": ratio_elapsed_time}, step=cur_max_depth)
            logging.info(f"Results similarity between FHE and unencrypted: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")
            wandb.log({"results_similarity": int((y_pred_fhe == y_pred_clear).mean()*100)}, step=cur_max_depth)

            # Append results to lists
            elapsed_times_fhe.append(elapsed_time_fhe)
            accuracies_fhe.append(accuracy_fhe)
            f1_scores_fhe.append(f1_score_fhe)
            precisions_fhe.append(precision_fhe)
            recalls_fhe.append(recall_fhe)
            ratio_elapsed_times.append(ratio_elapsed_time)

        # Log lists to W&B
        wandb.run.summary["elapsed_times_fhe"] = elapsed_times_fhe
        wandb.run.summary["accuracies_fhe"] = accuracies_fhe
        wandb.run.summary["f1_scores_fhe"] = f1_scores_fhe
        wandb.run.summary["precisions_fhe"] = precisions_fhe
        wandb.run.summary["recalls_fhe"] = recalls_fhe
        wandb.run.summary["ratio_eval_times"] = ratio_elapsed_times

        ######################### CREATING CUSTOM GRAPHS #########################
        #graph of Elapsed Time for FHE (elapsed_time_fhe) vs Max Depth (cur_max_depth)
        wandb.log({"time/fhe_eval_time_vs_cur_max_depth": 
                wandb.plot.line_series(xs=max_depth, ys=[elapsed_times_fhe], 
                                        title="Elapsed Time for FHE vs Max Depth", 
                                        xname="Max Depth", 
                                        keys=["Elapsed Time for FHE"]
                                        )})
        
        #graph of Ratio of Elapsed Time for FHE to Unencrypted (ratio_elapsed_time) vs Max Depth (cur_max_depth)
        wandb.log({"graph/time_ratio_vs_cur_max_depth": 
                wandb.plot.line_series(xs=max_depth, ys=[ratio_elapsed_times], 
                                        title="Ratio of FHE to Unencrypted Prediction Time against Max Depth", 
                                        xname="Max Depth", 
                                        keys=["Ratio of FHE to Unencrypted Prediction Time"]
                                        )})
        
        #graph Trade-Off between Accuracy, F1 Score, Precision and Recall vs Max Depth (cur_max_depth)
        wandb.log({"graph/fhe_metrics_vs_cur_max_depth": 
                wandb.plot.line_series(xs=max_depth, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Performance Metrics against Max Depth", 
                                        xname="Max Depth", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})
        
        #graph Trade-off between Accuracy, F1 Score, Precision and Recall vs Elapsed Time for FHE
        wandb.log({"graph/fhe_metrics_vs_fhe_elapsed_time": 
                wandb.plot.line_series(xs=elapsed_times_fhe, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Trade-off between Performance Metrics and Prediction Time for FHE", 
                                        xname="FHE Prediction Time (s)", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})

        wandb.finish()
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        wandb.finish()
        return
    
def xgb_experiment_n_bits(df, run_name, fhe_mode="execute", gfp=False):
    ##################### WANDB INITIALIZATION #####################
    wandb.init(
            mode="online",
            project="GFP_XGBoost_Experiments_2", #replace this with your wandb project name if you want to use wandb logging
            name=run_name,
    )
    try:
        logging.info(f"Wandb run ({wandb.run.name}) initialized")
        wandb.run.tags = ["n_bits", fhe_mode, "GFP" if gfp else "No GFP"]
    
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

        config = {
            "n_jobs": 1,
            "learning_rate": 0.07,
            "colsample_bytree": 0.98,
            "n_estimators": 20,
            "n_bits": 3,
        }

        wandb.config = config
        
        # Define our model 
        model = XGBClassifier(**config)

        # Define lists to store results
        elapsed_times_fhe = []
        accuracies_fhe = []
        f1_scores_fhe = []
        precisions_fhe = []
        recalls_fhe = []
        ratio_elapsed_times = []
        n_bits = list(range(1,16))
        wandb.run.summary["n_bits"] = n_bits
        logging.info("n_bits: {}".format(n_bits))

        # Fit the model for each value of cur_n_bits
        for cur_n_bits in n_bits:
            model.n_bits = cur_n_bits    
            
            logging.info("")
            logging.info("n_bits: {}".format(cur_n_bits))

            pipeline = Pipeline(
                [("standard_scaler", StandardScaler()), ("pca", PCA(random_state=0)), ("model", model)]
            )

            pipeline.fit(X_train, y_train)

            ground_truth = y_test

            ####################### XGBOOST PIPELINE #######################
            # Evaluate the model on the test set
            start_time_clear = time.time()
            y_pred_clear = model.predict(X_test)
            end_time_clear = time.time()

            ########### CLEAR RESULTS ###########
            elapsed_time_clear = end_time_clear - start_time_clear
            accuracy_clear = (y_pred_clear == ground_truth).mean()
            f1_score_clear = f1_score(ground_truth, y_pred_clear)
            precision_clear = precision_score(ground_truth, y_pred_clear)
            recall_clear = recall_score(ground_truth, y_pred_clear)


            # Log metrics to W&B with step=cur_n_bits
            wandb.log({"time/unencrypted_eval_time": elapsed_time_clear}, step=cur_n_bits)
            wandb.log({"accuracy/unencrypted": accuracy_clear}, step=cur_n_bits)
            wandb.log({"f1_score/unencrypted": f1_score_clear}, step=cur_n_bits)
            wandb.log({"precision/unencrypted": precision_clear}, step=cur_n_bits)
            wandb.log({"recall/unencrypted": recall_clear}, step=cur_n_bits)

            logging.info(f"Results for cur_n_bits={cur_n_bits}")
            logging.info("Prediction time for unencrypted: {:.6f}s".format(elapsed_time_clear))
            logging.info("Accuracy for unencrypted: {:.4f}".format(accuracy_clear))
            logging.info("F1 score for unencrypted: {:.4f}".format(f1_score_clear))
            logging.info("Precision for unencrypted: {:.4f}".format(precision_clear))
            logging.info("Recall for unencrypted: {:.4f}".format(recall_clear))

            ###################### FHE INFERENCE #######################
            # Compile the model to FHE
            model.compile(X_test)

            logging.info("Starting model evaluation on the test set in FHE")
            # Perform the inference in FHE and run on encrypted inputs
            start_time_fhe = time.time() 
            y_pred_fhe = model.predict(X_test, fhe=fhe_mode)
            end_time_fhe = time.time()

            ########### FHE RESULTS ###########
            elapsed_time_fhe = end_time_fhe - start_time_fhe
            accuracy_fhe = (y_pred_fhe == ground_truth).mean()
            f1_score_fhe = f1_score(ground_truth, y_pred_fhe)
            precision_fhe = precision_score(ground_truth, y_pred_fhe)
            recall_fhe = recall_score(ground_truth, y_pred_fhe)

            wandb.log({"time/fhe_eval_time": elapsed_time_fhe}, step=cur_n_bits)
            wandb.log({"accuracy/fhe": accuracy_fhe}, step=cur_n_bits)
            wandb.log({"f1_score/fhe": f1_score_fhe}, step=cur_n_bits)
            wandb.log({"precision/fhe": precision_fhe}, step=cur_n_bits)
            wandb.log({"recall/fhe": recall_fhe}, step=cur_n_bits)

            logging.info("Prediction time in FHE: {:.6f}s".format(elapsed_time_fhe))
            logging.info("Accuracy in FHE: {:.4f}".format(accuracy_fhe))
            logging.info("F1 score in FHE: {:.4f}".format(f1_score_fhe))
            logging.info("Precision in FHE: {:.4f}".format(precision_fhe))
            logging.info("Recall in FHE: {:.4f}".format(recall_fhe))

            ##################### OVERALL RESULTS #####################
            ratio_elapsed_time = elapsed_time_fhe / elapsed_time_clear
            logging.info("Prediction time of FHE / unencrypted: {:.2f}x".format(ratio_elapsed_time))
            wandb.log({"time/fhe_to_unencrypted_time_ratio": ratio_elapsed_time}, step=cur_n_bits)
            logging.info(f"Results similarity between FHE and unencrypted: {int((y_pred_fhe == y_pred_clear).mean()*100)}%")
            wandb.log({"results_similarity": int((y_pred_fhe == y_pred_clear).mean()*100)}, step=cur_n_bits)

            # Append results to lists
            elapsed_times_fhe.append(elapsed_time_fhe)
            accuracies_fhe.append(accuracy_fhe)
            f1_scores_fhe.append(f1_score_fhe)
            precisions_fhe.append(precision_fhe)
            recalls_fhe.append(recall_fhe)
            ratio_elapsed_times.append(ratio_elapsed_time)

        # Log lists to W&B
        wandb.run.summary["elapsed_times_fhe"] = elapsed_times_fhe
        wandb.run.summary["accuracies_fhe"] = accuracies_fhe
        wandb.run.summary["f1_scores_fhe"] = f1_scores_fhe
        wandb.run.summary["precisions_fhe"] = precisions_fhe
        wandb.run.summary["recalls_fhe"] = recalls_fhe
        wandb.run.summary["ratio_eval_times"] = ratio_elapsed_times

        ######################### CREATING CUSTOM GRAPHS #########################
        #graph of Elapsed Time for FHE (elapsed_time_fhe) vs Number of Bits (cur_n_bits)
        wandb.log({"time/fhe_eval_time_vs_cur_n_bits": 
                wandb.plot.line_series(xs=n_bits, ys=[elapsed_times_fhe], 
                                        title="Elapsed Time for FHE vs Number of Bits", 
                                        xname="Number of Bits", 
                                        keys=["Elapsed Time for FHE"]
                                        )})
        
        #graph of Ratio of Elapsed Time for FHE to Unencrypted (ratio_elapsed_time) vs Number of Bits (cur_n_bits)
        wandb.log({"graph/time_ratio_vs_cur_n_bits": 
                wandb.plot.line_series(xs=n_bits, ys=[ratio_elapsed_times], 
                                        title="Ratio of FHE to Unencrypted Prediction Time against Number of Bits", 
                                        xname="Number of Bits", 
                                        keys=["Ratio of FHE to Unencrypted Prediction Time"]
                                        )})
        
        #graph Trade-Off between Accuracy, F1 Score, Precision and Recall vs Number of Bits (cur_n_bits)
        wandb.log({"graph/fhe_metrics_vs_cur_n_bits": 
                wandb.plot.line_series(xs=n_bits, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Performance Metrics against Number of Bits", 
                                        xname="Number of Bits", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})
        
        #graph Trade-off between Accuracy, F1 Score, Precision and Recall vs Elapsed Time for FHE
        wandb.log({"graph/fhe_metrics_vs_fhe_elapsed_time": 
                wandb.plot.line_series(xs=elapsed_times_fhe, ys=[accuracies_fhe, f1_scores_fhe, precisions_fhe, recalls_fhe], 
                                        title="Trade-off between Performance Metrics and Prediction Time for FHE", 
                                        xname="FHE Prediction Time (s)", 
                                        keys=["Accuracy", "F1 Score", "Precision", "Recall"]
                                        )})

        wandb.finish()
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        wandb.finish()
        return