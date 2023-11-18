import torch
import copy


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience):
    train_loss_history = []
    valid_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0

    # train
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            # labels = labels.squeeze()
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_loss_history.append(epoch_train_loss)

        # validation
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                # labels = labels.squeeze()
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

        print(f'Epoch {epoch + 1}/{num_epochs} train loss: {epoch_train_loss} valid loss: {epoch_valid_loss}')

    model.load_state_dict(best_model_wts)

    return model, train_loss_history, valid_loss_history


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # labels = labels.squeeze()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = (torch.sigmoid(outputs).data > 0.5).float()
            labels = labels.float()
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy
