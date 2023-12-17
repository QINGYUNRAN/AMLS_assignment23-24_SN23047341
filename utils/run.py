import torch
import copy


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, clip_value=None):
    """
       Trains a neural network model using given data loaders and updates the model parameters.

       Args:
           model (torch.nn.Module): The neural network model to train.
           train_loader (DataLoader): DataLoader for the training data.
           val_loader (DataLoader): DataLoader for the validation data.
           criterion (function): The loss function.
           optimizer (torch.optim.Optimizer): The optimization algorithm.
           device (torch.device): Device to run the training on (e.g., 'cuda', 'cpu').
           num_epochs (int): Number of epochs to train for.
           patience (int): Patience for early stopping (number of epochs to wait after last improvement).
           clip_value (float, optional): Max norm of the gradients; if specified, gradient clipping is applied.

       Returns:
           tuple: (model, train_loss_history, valid_loss_history)
               model - The trained model with the best weights.
               train_loss_history - List of training losses per epoch.
               valid_loss_history - List of validation losses per epoch.
       """
    train_loss_history = []
    valid_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # train
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.squeeze()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                labels = labels.squeeze()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        epoch_valid_loss = running_loss / len(val_loader)
        valid_loss_history.append(epoch_valid_loss)

        if epoch_valid_loss < best_loss:
            best_loss = epoch_valid_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print('Early stopping!')
            break

        # scheduler.step()

        print(
            f'Epoch {epoch + 1}/{num_epochs} train loss: {epoch_train_loss} valid loss: {epoch_valid_loss}')

    model.load_state_dict(best_model_wts)

    return model, train_loss_history, valid_loss_history


def test(model, test_loader, device):
    """
        Tests a trained neural network model using a test data loader.

        Args:
            model (torch.nn.Module): The trained neural network model.
            test_loader (DataLoader): DataLoader for the test data.
            device (torch.device): Device to run the test on (e.g., 'cuda', 'cpu').

        Returns:
            tuple: (test_accuracy, predictions)
                test_accuracy - The accuracy of the model on the test set.
                predictions - List of predicted labels for the test dataset.
        """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.squeeze()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # predicted = (torch.sigmoid(outputs).data > 0.5).float()
            labels = labels.float()
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy, predictions
