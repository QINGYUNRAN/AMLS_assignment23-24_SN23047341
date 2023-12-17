import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from utils.datasets import get_loader, get_data
from utils.run import train, test
from A.model import my_net
import random
import optuna
import matplotlib.pyplot as plt


class SolutionA:
    """
        A class to encapsulate the complete pipeline for training, validating, and testing a neural network model.
        It handles data loading, model setup, training, hyperparameter tuning, and testing.
    """
    def __init__(self, config):
        """
            Initializes the SolutionA class with the given configuration.

            Args:
                config (dict): Configuration parameters including seeds, batch size, learning rate, etc.
        """
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.set_global_seed(self.config['seed'])
        base_dir = os.path.dirname(__file__)
        self.data_path_A = os.path.join(base_dir, '..', 'Datasets', 'PneumoniaMNIST', 'pneumoniamnist.npz')
        self.checkpoint_path = os.path.join(base_dir, "checkpoint")
        self.image_output_path = os.path.join(base_dir, "image_output")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.image_output_path, exist_ok=True)
        self.setup_dataloader()
        self.setup_model()
        # if self.config["check_balance"]:
        #     self.check_balance()

        if not self.config["params_search"]:
            self.run()
        else:
            self.params_search()
            self.config["retrain_flag"] = False
            self.run()

    def set_global_seed(self, seed_value):
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed_value)

    def calc_dataset_stats(self, x_train):
        calculated_x = x_train / 255.0
        mean = np.mean(calculated_x)
        std = np.std(calculated_x)
        return mean, std

    def setup_dataloader(self):
        data_a = np.load(self.data_path_A)
        batch_size = self.config['batch_size']
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = get_data(data_a)
        self.mean, self.std = self.calc_dataset_stats(self.x_train)
        self.train_loader = get_loader(self.x_train, self.y_train, batch_size=batch_size, mean=self.mean, std=self.std,
                                       flag='Train')
        self.val_loader = get_loader(self.x_val, self.y_val, batch_size=batch_size, mean=self.mean, std=self.std,
                                     flag='Val')
        self.test_loader = get_loader(self.x_test, self.y_test, batch_size=batch_size, mean=self.mean, std=self.std,
                                      flag='Test')

    def setup_model(self):
        self.model = my_net(input_dim=1, num_classes=2, dropout_rate=self.config['dropout_rate']).to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'],
                                           weight_decay=self.config['weight_decay'])

    def run(self):
        if self.config["retrain_flag"]:
            self.best_model, self.train_loss_history, self.valid_loss_history = train(
                self.model, self.train_loader, self.val_loader,
                self.loss_function, self.optimizer, self.device,
                num_epochs=self.config['num_epochs'], patience=self.config['patience'],
                clip_value=self.config["clip_value"]
            )
            torch.save(self.best_model.state_dict(), os.path.join(self.checkpoint_path, "retrained_model.pth"))
            self.test_accuracy, self.predictions = test(self.best_model, self.test_loader, self.device)
            print("Test accuracy:", self.test_accuracy)
            self.visualize_loss(self.train_loss_history, self.valid_loss_history)
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(self.checkpoint_path, "best_model_A.pth"), map_location=self.device))
            self.best_model = self.model
            self.test_accuracy, self.predictions = test(self.best_model, self.test_loader, self.device)
            print("Test accuracy:", self.test_accuracy)

    def params_search(self):
        def objective(trial):
            batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7)
            clip_value = trial.suggest_float('clip_value', 0.1, 10.0, log=True)

            train_loader = get_loader(self.x_train, self.y_train, batch_size, self.mean, self.std, flag='Train')
            val_loader = get_loader(self.x_val, self.y_val, batch_size, self.mean, self.std, flag='Val')
            model = my_net(input_dim=1, num_classes=2, dropout_rate=dropout_rate).to(self.device)
            loss_function = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            best_model, train_loss_history, valid_loss_history = train(
                model, train_loader, val_loader,
                loss_function, optimizer, self.device,
                num_epochs=self.config['num_epochs'], patience=self.config['patience'],
                clip_value=clip_value,
            )

            val_accuracy, _ = test(best_model, val_loader, self.device)
            if trial.number == 0 or val_accuracy > trial.study.best_value:
                torch.save(best_model.state_dict(), os.path.join(self.checkpoint_path, "best_model_A.pth"))
                # self.visualize_loss(train_loss_history, valid_loss_history)
            return val_accuracy

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        print(f'Best trial: {study.best_trial.number}')
        print(f'Best value: {study.best_trial.value}')
        print(f'Best hyperparameters: {study.best_trial.params}')

    def check_balance(self):
        def print_class_distribution(labels, dataset_name):
            unique, counts = np.unique(labels, return_counts=True)
            class_counts = dict(zip(unique, counts))
            print(f"{dataset_name} Class Distribution:")
            for class_id, count in class_counts.items():
                print(f"Class {class_id}: {count} samples")
            return class_counts

        def check_balance(class_counts, total_samples, threshold=0.1):
            for count in class_counts.values():
                if count / total_samples < threshold:
                    return False
            return True

        train_class_counts = print_class_distribution(self.y_train, "Training")
        val_class_counts = print_class_distribution(self.y_val, "Validation")
        test_class_counts = print_class_distribution(self.y_test, "Testing")

        train_balanced = check_balance(train_class_counts, len(self.y_train))
        val_balanced = check_balance(val_class_counts, len(self.y_val))
        test_balanced = check_balance(test_class_counts, len(self.y_test))

        print(f"Training set is {'balanced' if train_balanced else 'unbalanced'}")
        print(f"Validation set is {'balanced' if val_balanced else 'unbalanced'}")
        print(f"Testing set is {'balanced' if test_balanced else 'unbalanced'}")

    def visualize_loss(self, train_loss_history, valid_loss_history):
        if not train_loss_history or not valid_loss_history:
            print("Loss history is empty. Cannot visualize loss.")
            return

        epochs = range(1, len(train_loss_history) + 1)

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_loss_history, label='Training Loss')
        plt.plot(epochs, valid_loss_history, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.image_output_path, "task_A_loss.pdf"), format='pdf')


def main():
    config = {
        "seed": 75,
        "num_epochs": 100,
        "patience": 10,
        "batch_size": 32,
        "learning_rate": 6.545977883745346e-05,
        "dropout_rate": 0.4533425997939389,
        "weight_decay": 0.0007272720771299431,
        "clip_value": 2.309330019284563,
        "retrain_flag": True,
        "check_balance": False,
        "params_search": False,

    }
    SolutionA(config)


if __name__ == '__main__':
    main()
