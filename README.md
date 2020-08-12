# TBQN
The code base for my Master thesis "Transformer based action sequence generation in reinforcement learning settings"


This repository is build on TensorFlow and TF-Agents.

It includes the following useful things: 

  - Fully modularized code to Run a DQN agent with a Transformer based architecture as its network  (TBQN). 
  - Simple scripts to run TBQN with a mountain of different Parameters and model variations.
  - Scripts to perform parameter optimization for TBQN using the Optuna Library
  - The code and the results of the experiments I conducted during my thesis work.
  - Notebooks that can be used to evaluate and display either single model performance or whole studies.
  
  
  
 Examples: 
 
  - To run an experiment simply run one of the experiment scripts like this: 
    "python experiment_script_3.py --output_dir Acrobot-v1 --env Acrobot-v1"
    parameters can be added and changed accordingly to the script.
