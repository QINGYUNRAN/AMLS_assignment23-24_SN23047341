import argparse
from A.Solution_A import SolutionA
from B.Solution_B import SolutionB


def main():
    parser = argparse.ArgumentParser(description='Run Solution A or B with configurations.')

    parser.add_argument('--solution', type=str, choices=['A', 'B'], required=True, help='Choose which solution to run.')
    parser.add_argument('--seed', type=int, default=75, help='Set the random seed.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=6.130419127136388e-05,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--dropout_rate', type=float, default=0.6215811268016947, help='Dropout rate for the model.')
    parser.add_argument('--weight_decay', type=float, default=3.734186103804895e-07,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--clip_value', type=float, default=3.9099035958216297, help='Gradient clipping value.')
    parser.add_argument('--retrain', action='store_true', help='Flag to retrain the model.')
    parser.add_argument('--check_balance', action='store_true', help='Flag to check balance of the dataset.')
    parser.add_argument('--params_search', action='store_true', help='Flag to perform hyperparameter search.')

    args = parser.parse_args()


    config = {
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout_rate": args.dropout_rate,
        "weight_decay": args.weight_decay,
        "clip_value": args.clip_value,
        "retrain_flag": args.retrain,
        "params_search": args.params_search,
    }

    if args.solution == 'A':
        SolutionA(config)
    elif args.solution == 'B':
        SolutionB(config)


if __name__ == '__main__':
    main()
