## Project Brief
https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications

This project is an absolute mess. 
I need to organize the code and workflow of this project much better.

## Installation
To get all the required dependencies ...

pip install .

## Getting Started

This project requires the training data from the challenge to be added the data directory of this project.

Starting at the root of the project, follow the below steps:

- pip install -U aicrowd-cli==0.1
- API_KEY = "XXX" # Whatever your key is.
- aicrowd login --api-key $API_KEY
- mkdir data && cd data
- aicrowd dataset download --challenge mabe-task-1-classical-classification

## Prepping the data

From the root of the project run the following command:

python3 src/animal_behavior_detection/data_seq_aug_mars.py

This command restructures the data to make it more useful.

## Entry points

From the root of the project run the following commands:

- python3 src/animal_behavior_detection/class_imbalance.py

This module implements experiments that tackle the issue of class imbalance in the dataset. 
The experiments test the weighted sampling, focal loss, and class imbalance focal loss methods. 
Running this module will produce models and various performance plots for each of the methods. 

- python3 src/animal_behavior_detection/plot_data_stats.py

This module produces figures that visualization class imbalance problem of data set.

- python3 src/animal_behavior_detection/transformer.py

This module trains a transformer on the MARS dataset.
The training dataset has been augmented to ensure equal weighting across all classes.
The dataset has also been sequenced (more explanation required here), allowing us to take advantage of the temporal nature of the data.
Running this module will produce a model and various performance plots of the model. 

## TODO

- Add technical details.
