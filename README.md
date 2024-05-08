[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ifbeTrPr)
# e6691-2024spring-project-cgnn-nhw2114

For general requiremenets, refer to the [project instruction document](https://docs.google.com/document/d/1IqkNFUTRoI8xk0a-xawlIzA_QHHk5pZcRZY-q-zey1Q/edit?usp=share_link).

## Deliverables
- Students’ report - [e6691-2024spring-project-cgnn-nhw2114](#deliverables)
- Students’ slides - [E6691.2024Spring.CGNN.nhw2114.presentationFinal](https://docs.google.com/presentation/d/1HNhcweCy0BCiZ24RSah8lsI6n9f_g-vdYaPZ-Pe3NV8/edit#slide=id.geb327816ab_0_0)
- Papers: [Examining COVID-19 Forecasting using Spatio-Temporal Graph Neural Networks (2020)](https://arxiv.org/abs/2007.03113)
- Github: [e6691-2024spring-project-cgnn-nhw2114](#deliverables)

## Installation
```
pip install -r requirements.txt
```

```
conda env create -f environment.yml
```
Other requirements to run on GCP:
- Linux
- NVIDIA GPU
- PyTorch 2.1.*
- CUDA 11.8+

## Directory Tree Structure
```
e6691-2024spring-project-cgnn-nhw2114/
│
├── assets/                                            # Plots of processed data and model results
|
├── data/
│   ├── raw/                                           # Raw safegraph and covid data
│   └── processed/                                     # Processed data for PyTorch-Geometric format
│
├── src/
│   ├── experiments/                                   # Result csv's from experiments
|   ├── *.yaml                                         # Experiment yaml files for hyperparameter tuning
|   ├── *_experiment.py                                # Scripts to run experiments
|   ├── colab_process_safegraph_mobility.py            # Jupyter notebook for processing raw safegraph data
|   ├── model.py                                       # GNN model definitions
|   ├── process_data.py                                # Script to process raw data into PyTorch-Geometric format
│   └── main.ipynb                                     # Core jupyter notebook with model runs and outputs                        
|
├── test/                                              # Unit tests for data processing
|
├── utils/   
│   ├── codebook.py                                    # mapping dicts for borough and FIPS code
│   └── utils.py                                       # util functions for graph node mapping
│
├── requirements.txt                                   # List of Python dependencies for pip
├── environment.yml                                    # List of Python dependencies for conda
├── .flake8                                            # flake8 codestyle
└── README.md                                          # Project README file with an overview and setup instructions
```

## Usage
To reproduce all plots and results in the presentation and report run the following Jupyter notebook
```
src/main.ipynb
```
To rerun experiments
```
python src/cgnn_experiment.py
```

```
python src/a3tgcn_experiment.py
```