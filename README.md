

# MQ-GNN Module

This repository is part of the **Graph Database Management System (GDBMS)** project. It contains the implementation of the **MQ-GNN module**, which is designed for efficient processing of graph data.

The complete project code, including the final MQ-GNN implementation, will be soon released following the evaluation of the GDBMS project. Currently, this repository is private ([dke-lab/dgll](https://github.com/dke-lab/dgll)) but will be made public after the evaluation is completed. For reference, here are the repositories showcasing early development of the project:

- [GDLL 2023](https://github.com/ahj6377/GDLL2023)  
- [DGLL 2022](https://github.com/dke-lab/DGLL-2022)  



## Project Support

This work is supported by the **Institute of Information & communications Technology Planning & Evaluation (IITP)** grant funded by the Korea government (MSIT) under the project:

> **No.2021-0-00859**: Development of a distributed graph DBMS for intelligent processing of big graphs.

## Repository Structure

- **`memory_profiling.py`**: A script for profiling GPU memory usage during training.
- **`sampler.py`**: Contains implementations of various sampling strategies for GNN training.
- **`model`**: Contains GNN models.
- **`MQ-GNN.py`**: Contains the implementation of the MQ-GNN module.
- **`SoTA_default.py`**: Scripts for running baseline models against the MQ-GNN for evaluation.

## About MQ-GNN

**MQ-GNN** is a framework for scalable and efficient GNN training. It serves as an important component within the GDBMS project and demonstrates state-of-the-art performance on graph datasets.

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
* [Data and Knowlege Engineering Lab (DKE)](http://dke.khu.ac.kr/)
<p align="right">(<a href="#top">back to top</a>)</p>
