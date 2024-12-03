import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error


__all__ = ['train_val', 'train', 'model_test']


def train(model, train_loader, num_epochs=100, lr=0.001):
    """
    Trains a given model using the specified training data loader.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int, optional): Number of epochs to train for. Defaults to
        100.
        lr (float, optional): Learning rate for the optimizer. Defaults to
        0.001.

    Returns:
        nn.Module: The trained model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_loss = float('inf')
    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0

        for batch in train_loader:
            x_img, x_storm_id, x_relative_time, x_ocean, y = batch
            x_relative_time = x_relative_time.unsqueeze(2)
            optimizer.zero_grad()
            outputs = model(x_img, x_storm_id, x_relative_time, x_ocean)
            loss = criterion(outputs, y)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: \
               {avg_train_loss:.4f}')
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), './Final_best_Speed_{:03d}.pth')
            continue
        if (np.mod(epoch, 5) == 0):
            torch.save(model.state_dict(), "./Final_Speed_{:03d}.pth"
                       .format(epoch))
        return model


def train_val(model, train_loader, val_loader, num_epochs=100, lr=0.001):
    """
    Trains and validates a given model using specified training and validation
    data loaders.

    Args:
        model (nn.Module): The model to be trained and validated.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        num_epochs (int, optional): Number of epochs for training. Defaults to
        100.
        lr (float, optional): Learning rate for the optimizer. Defaults to
        0.001.

    Note:
        This function prints the training and validation loss for each epoch
        and
        saves the model state dict periodically and when the best validation
        loss is achieved.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_loss = 99999999999.
    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0

        for batch in train_loader:
            x_img, x_storm_id, x_relative_time, x_ocean, y = batch
            x_relative_time = x_relative_time.unsqueeze(2)
            optimizer.zero_grad()
            outputs = model(x_img, x_relative_time)
            loss = criterion(outputs, y)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: \
               {avg_train_loss:.4f}')

        model.eval()
        total_val_loss = 0
        predictions, actuals = [], []
        exp = []
        act = []
        with torch.no_grad():
            for batch in val_loader:
                x_img, x_storm_id, x_relative_time, x_ocean, y = batch
                x_relative_time = x_relative_time.unsqueeze(2)
                outputs = model(x_img, x_relative_time)
                exp.append(y)
                act.append(outputs)

                print("-------------------------------------------- \
                      --------------------------")
                exp_floats = ['{:.2f}'.format(x.item()) for x in y]
                print("GroundTruth:", exp_floats)
                act_floats = ['{:.2f}'.format(x.item()) for x in outputs]
                print("Prediction: ", act_floats)
                print("-------------------------------------------- \
                      --------------------------")
                loss = criterion(outputs, y)
                total_val_loss += loss.item()
                predictions.extend(outputs.squeeze(1).tolist())
                actuals.extend(y.squeeze(1).tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        val_mse = mean_squared_error(actuals, predictions)
        print(f'Validation Loss: {avg_val_loss:.4f}, MSE: {val_mse:.4f}')
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), './F_best_Speed.pth')
            continue
        if (np.mod(epoch, 5) == 0):
            torch.save(model.state_dict(), "./F_Speed_{:03d}.pth"
                       .format(epoch))


def train_model(model, dataloader, criterion, optimizer, num_epochs=20,
                device="cpu"):
    """
    Trains a model with a given dataset, criterion, and optimizer.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): The DataLoader providing the dataset.
        criterion (Loss): The loss function to use for training.
        optimizer (Optimizer): The optimizer to use for training.
        num_epochs (int): The number of epochs to train the model.
        device (str): The device to train the model on, 'cpu' or 'cuda'.

    Note:
        Prints loss every 10 epochs and saves the best model based on loss.
    """
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for image, relative_time, storm_id, ocean_id, wind_speed in dataloader:
            image = image.to(device).float()
            relative_time = relative_time.to(device).unsqueeze(0).float()
            storm_id = storm_id.to(device).float()
            ocean_id = ocean_id.to(device).float()
            wind_speed = wind_speed.to(device).float()

            outputs = model(image, storm_id, relative_time, ocean_id)
            loss = criterion(outputs, wind_speed)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), 'best_model.pth')


def model_test(model, test_loader, device='cpu'):
    """
    Tests the model with a given test data loader and calculates predictions.

    Args:
        model (nn.Module): The trained model to be tested.
        test_loader (DataLoader): DataLoader for test data.
        device (str, optional): The device to run the test on. Defaults to
        'cpu'.

    Returns:
        tuple: A tuple containing two lists, predicted values and actual
        values from the test set.
    """
    model.eval()
    y_pred_list = []
    actual = []

    with torch.no_grad():

        for img, relative_time, storm_id, ocean_id, wind_speed in test_loader:
            outputs = model(img, storm_id, relative_time, ocean_id)
            y_pred_list.append(outputs.squeeze(1).tolist())
            actual.append(wind_speed.squeeze(1).tolist())

    return y_pred_list, actual
