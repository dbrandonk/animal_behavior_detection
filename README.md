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

This command restructures the data to make useful.

## TODO

- Add technical details.
