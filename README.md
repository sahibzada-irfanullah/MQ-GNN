

# MQ-GNN Module

This repository is part of the **Graph Database Management System (GDBMS)** project. It contains the implementation of the **MQ-GNN module**, which is designed for efficient processing of graph data.

The complete project code, including the final MQ-GNN implementation, will be released following the evaluation of the GDBMS project. Currently, this repository is private ([dke-lab/dgll](https://github.com/dke-lab/dgll)) but will be made public after the evaluation is completed.

## Project Support

This work is supported by the **Institute of Information & communications Technology Planning & Evaluation (IITP)** grant funded by the Korea government (MSIT) under the project:

> **No.2021-0-00859**: Development of a distributed graph DBMS for intelligent processing of big graphs.

## Repository Structure

- **`memory_profiling.py`**: A utility script for profiling GPU memory usage during training and inference.
- **`sampler.py`**: Contains implementations of various sampling strategies for GNN training.
- **`model/`**: Houses the implementations of the MQ-GNN module.
- **`MQ-GNN.py`**: The core implementation of the MQ-GNN module.
- **`SoTA_default.py`**: Scripts for running baseline models against the MQ-GNN for benchmarking.

## About MQ-GNN

**MQ-GNN** is a framework for scalable and efficient GNN training. It serves as an important component within the GDBMS project and demonstrates state-of-the-art performance on graph datasets.

