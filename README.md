## Project Brief
https://www.aicrowd.com/challenges/multi-agent-behavior-representation-modeling-measurement-and-applications

As a disclaimer, to be put nicely, the organization and workflow of the existing codebase of this project has a lot of room for improvement. 

## Installation
To install all the necessary dependencies, please run the following command at the root of the project:

```
pip install .
```

## Getting Started
To begin working with this project, you need to obtain the training data from the challenge and add it to the data directory of this project. Follow the steps below:

1. Install the AICrowd CLI by running the command:
   ```
   pip install -U aicrowd-cli==0.1
   ```

2. Set your API key as an environment variable:
   ```
   API_KEY="XXX"  # Replace with your actual API key
   ```

3. Log in to the AICrowd platform using the CLI:
   ```
   aicrowd login --api-key $API_KEY
   ```

4. Create a directory named 'data' and navigate into it:
   ```
   mkdir data && cd data
   ```

5. Download the challenge dataset using the AICrowd CLI:
   ```
   aicrowd dataset download --challenge mabe-task-1-classical-classification
   ```

## Prepping the Data
The below command performs necessary data restructuring operations:

```
python3 src/animal_behavior_detection/data_seq_aug_mars.py
```

TL;DR: The dataset is sequenced allowing us to take advantage of the temporal nature of the data.

## Entry Points
The following commands can be executed from the root of the project:

  ```
  python3 src/animal_behavior_detection/plot_data_stats.py
  ```

  This module produces figures that visualization the class imbalance problem of data set.

  ```
  python3 src/animal_behavior_detection/class_imbalance.py
  ```

  This command implements experiments that tackle the issue of class imbalance in the dataset. 
  The experiments test the weighted sampling, focal loss, and class imbalance focal loss methods. 
  Running this module will produce models and various performance plots for each of the methods. 


  ```
  python3 src/animal_behavior_detection/transformer.py
  ```
  This command trains a transformer on the MARS dataset.
  The training dataset has been augmented to ensure equal weighting across all classes.
  The dataset has also been sequenced (more explanation required here), allowing us to take advantage of the temporal nature of the data.
  Running this command will produce a model and various performance plots of the model. 

## TODO
- Include more technical details.
