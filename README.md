# GraphML-FHE
This is the repository for my Nanyang Technological University Undergraduate Computer Science Final Year Project - "Privacy-Preserving Graph-Based Machine Learning with Fully Homomorphic Encryption for Collaborative Anti-Money Laundering". 

This research presents a novel privacy-preserving approach for collaborative AML machine learning, facilitating secure data sharing across institutions and borders while preserving data privacy and regulatory compliance. Leveraging Fully Homomorphic Encryption (FHE), computations can be performed on encrypted data without decryption, ensuring sensitive financial data remains confidential. 

The research delves into the integration of Fully Homomorphic Encryption over the Torus (TFHE) using [Concrete ML](https://github.com/zama-ai/concrete-ml) with graph-based machine learning techniques, which are divided into 2 pipelines.
1. Graph Neural Network (GNN) pipeline - in [gnn_fhe_pipeline](./gnn_fhe_pipeline) directory
2. Graph-based XGBoost pipline using Graph Feature Preprocessor - in [gfp_xgb_fhe_pipeline](./gfp_xgb_fhe_pipeline/) directory

Instructions for running each pipeline are detailed in the README.md of the respective directories.

## Licence
Apache License
Version 2.0, January 2004
