# AMLS_assignment23-24_SN23047341

UCL AMLS coursework by Qingyun Ran

To obtain the best model files, please download them from the following link and place them in their respective checkpoint directories:
- [Download best_model_A.pth](https://liveuclac-my.sharepoint.com/:u:/g/personal/uceeqra_ucl_ac_uk/EUK-wT4TdBBBu7V2tzwXIqMBdJYZYqn_vV0n-szI4_FUsw?e=tBx0o6)
- [Download best_model_B.pth](https://liveuclac-my.sharepoint.com/:u:/g/personal/uceeqra_ucl_ac_uk/EWejm_369QNHmIITCG4qooUBNeqjsAzkswfgmZKJR1ZEJw?e=zpUfbV)


## Project Structure

```plaintext
project/
│
├── main.py     
│
├── A/
│   ├── checkpoint/     
│   ├── image_output/     
│   ├── __init__.py        
│   ├── model.py           
│   ├── Solution_A.py      
│   └── Try.py            
│
├── B/
│   ├── checkpoint/      
│   ├── image_output/     
│   ├── __init__.py       
│   ├── model.py          
│   └── Solution_B.py      
│
├── Datasets/
│   ├── PneumoniaMNIST/pneumoniamnist.npz   
│   └── PathMNIST/pathmnist.npz        
│
└── utils/
├── __init__.py     
├── datasets.py       
└── run.py            
```

### File Descriptions

- `main.py`: This is the main script of the project. Run this file to start the process defined for either Task A or
  Task B.

#### Task A Files

- `A/checkpoint/`: This folder contains the saved model checkpoints which can be used to resume training or evaluate the
  model on the test set.
- `A/image_output/`: Here you will find visualizations of the training and validation loss and accuracy, which are
  helpful for analyzing the model's performance over time.
- `A/model.py`: Defines the neural network architecture used for solving Task A.
- `A/Solution_A.py`: Implements the specific solution, including the training and evaluation pipelines for Task A.
- `A/Try.py`: Contains initial attempts to apply traditional machine learning methods to Task A before switching to
  neural network-based solutions.Since neural networks were ultimately chosen as the solution for both tasks, there is
  no Try.py for Task B, and it is not included in main.py. If you need to see the attempts of the machine learning method
  for Task A, run Try.py separately.

#### Task B Files

- `B/checkpoint/`: Similarly, this folder contains model checkpoints for Task B.
- `B/image_output/`: Stores training and validation performance plots for Task B.
- `B/model.py`: The neural network architecture designed for Task B.
- `B/Solution_B.py`: The solution code for Task B, including training and evaluation procedures.

#### Dataset Files

- `Datasets/PneumoniaMNIST/`: Contains the Pneumonia MNIST dataset in `.npz` format for use with Task A.
- `Datasets/PathMNIST/`: Contains the Path MNIST dataset in `.npz` format, which may be used for Task B.

#### Utils Files

- `Utils/datasets.py`: This script includes functions for loading, preprocessing, and augmenting the data for both
  tasks.
- `Utils/run.py`: Contains common functions used to train and evaluate models for both Task A and Task B.


## Packages and Requirements

This program runs under Python version 3.11.5.
The project depends on several external libraries, which are listed in `requirements.txt` or `environment.yml`. To install these dependencies, run the command below:

```sh
pip3 install -r requirements.txt
```
or
```sh
conda env create -f environment.yml
```
```plaintext
numpy
optuna
matplotlib
torch==2.1.0+cu118
torchaudio==2.1.0+cu118
torchvision==0.16.0
scikit-learn==1.1.3
joblib
albumentations
```

## Running the Main Script

To run `main.py`, please be aware of the following key parameters:

- `seed`: The random seed for reproducibility.
- `num_epochs`: Number of epochs to train the model.
- `patience`: Number of epochs to wait for improvement before early stopping.
- `batch_size`: Number of samples per batch.
- `learning_rate`: Learning rate for the optimizer.
- `dropout_rate`: Dropout rate for the neural network.
- `weight_decay`: Weight decay for regularization.
- `clip_value`: Gradient clipping threshold.
- `retrain_flag`: If `True`, retrain the model from scratch.
- `params_search`: If `True`, execute hyperparameter optimization with Optuna.

Default values are provided for the training hyperparameters. When the `retrain_flag` is enabled, the model will retrain and save as `retrain_model.pth` in the task-specific `checkpoint` directory. If `params_search` is enabled, the model will undergo a comprehensive parameter search with Optuna and save the best model as `best_model.pth`, which may take an extended time to complete.

If both `retrain_flag` and `params_search` are set to `False`, the script will directly load the best-tuned model and display its accuracy on the test set.

Ensure that you have `best_model_A.pth` and `best_model_B.pth` in the checkpoint directories of their respective tasks. Otherwise, you will not be able to run the following commands to verify the best accuracy obtained, and you will have to retrain the models yourself.
```sh
python main.py --solution A
python main.py --solution B
```
Or you can just train the model using custom hyperparameters, specifically your own learning rate, you can run like the following command:

```sh
python main.py --solution A --retrain_flag --learning_rate 0.001
```

If you want to save time, you can directly use the default settings for training, but they may not be the optimal parameters for your chosen task.
```sh
python main.py --solution A --retrain_flag
python main.py --solution B --retrain_flag
```
