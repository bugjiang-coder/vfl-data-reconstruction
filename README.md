# UIFV: Data Reconstruction Attack in Vertical Federated Learning

This repository contains the code and configurations for the paper "UIFV: Data Reconstruction Attack in Vertical Federated Learning." The code is organized into various directories, each serving a specific purpose in the implementation of the data reconstruction attack and related experiments.

![img](img.png)

## Description of Directories

- **configs**: Contains configuration files for attacks and training.
  - **attack**: Configurations for various attack scenarios on different datasets (e.g., adult, bank, cifar10).
  - **train**: Configurations for training the models.
- **DataSize**: Contains data size-related files.
- **defense**: Code and configurations related to defense mechanisms against attacks.
- **experiment**: Scripts and files for running various experiments.
- **fedml_core**: Core library for federated learning.
  - **model**: Contains model definitions.
  - **preprocess**: Scripts for preprocessing datasets.
    - Subdirectories for specific datasets (e.g., adult, avazu, bank).
  - **trainer**: Training scripts for the models.
  - **utils**: Utility functions used throughout the project.
- **modelSize**: Contains files related to model sizes.
- **RatioExperiment**: Files related to ratio experiments.
- **train_base_model**: Scripts for training the base models.

## Getting Started

### Prerequisites

Creating an Environment Use the following command to create an environment, which will read the `environment_vfl.yml` file and install all the listed dependencies:

```
conda env create -f environment_vfl.yml
```



### Running Experiments

To run an experiment, you need to specify the configuration file. For example, to run an attack on the adult dataset:

```bash
python experiment/run_attack.py --c configs/attack/adult/config.yaml
```

### Training Models

To train a base model, specify the configuration file:

```bash
python train_base_model/train.py --c configs/train/config.yaml
```

### Defense Mechanisms

To test defense mechanisms, run the appropriate script with the desired configuration:

```bash
python defense/run_defense.py --c configs/defense/config.yaml
```



## Paper URL

Our paper can be accessed at: http://arxiv.org/abs/2406.12588





---

For any questions or issues, please open an issue on GitHub or contact the maintainer.
