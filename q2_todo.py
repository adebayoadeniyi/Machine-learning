# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
import os

dir_working = "/media/adebayo/Windows/UofACanada/study/codes/CMPUT566_ML/coding_2/"

os.chdir(dir_working)


def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]


    return X_train, t_train, X_val, t_val, test_data / 256, test_labels


def softmax(z):
    # Subtract the max for numerical stability
    z -= np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y, t):
    # Cross-entropy loss for multi-class classification
    N = t.shape[0]
    return -np.sum(t * np.log(y + 1e-15)) / N  # Adding 1e-15 to prevent log(0)


def predict(X, W, t=None):
    z = np.dot(X, W)  # Compute the class scores
    y = softmax(z)    # Compute softmax probabilities
    t_hat = np.argmax(y, axis=1)  # Predicted class labels

    loss, acc = None, None
    if t is not None:
        # Convert t to one-hot encoding
        K = W.shape[1]  # Number of classes
        t_one_hot = np.eye(K)[t.flatten()]  # One-hot encode the labels
        loss = cross_entropy_loss(y, t_one_hot)

        # Calculate accuracy
        acc = np.mean(t_hat == t.flatten()) * 100  # Accuracy as percentage

    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    d_plus_1 = X_train.shape[1]
    K = len(np.unique(y_train))  # Number of classes
    W = np.random.randn(d_plus_1, K) * 0.01  # Initialize weights

    train_losses = []
    valid_accs = []

    acc_best = 0
    W_best = None
    epoch_best = 0

    for epoch in range(MaxEpoch):
        # Shuffle the data for SGD
        indices = np.random.permutation(N_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for i in range(0, N_train, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Convert y_batch to one-hot encoding
            t_one_hot_batch = np.eye(K)[y_batch.flatten()]

            # Forward pass
            z_batch = np.dot(X_batch, W)
            y_batch_pred = softmax(z_batch)

            # Gradient computation
            grad_W = np.dot(X_batch.T, (y_batch_pred - t_one_hot_batch)) / X_batch.shape[0]
            grad_W += decay * W  # Regularization term (L2 penalty)

            # Update weights
            W -= alpha * grad_W

        # Evaluate on the current mini-batch (train loss)
        _, _, train_loss, _ = predict(X_batch, W, y_batch)
        train_losses.append(train_loss)

        # Evaluate on validation data
        _, _, _, val_acc = predict(X_val, W, t_val)
        valid_accs.append(val_acc)

        # Track the best model based on validation accuracy
        if val_acc > acc_best:
            acc_best = val_acc
            W_best = W.copy()
            epoch_best = epoch

        print(f"Epoch {epoch+1}/{MaxEpoch} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_acc:.2f}%")

    return epoch_best, acc_best, W_best, train_losses, valid_accs


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    d_plus_1 = X_train.shape[1]
    K = len(np.unique(y_train))  # Number of classes
    W = np.random.randn(d_plus_1, K) * 0.01  # Initialize weights

    train_losses = []
    valid_accs = []

    acc_best = 0
    W_best = None
    epoch_best = 0

    for epoch in range(MaxEpoch):
        # Shuffle the data for SGD
        indices = np.random.permutation(N_train)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0  # Accumulate loss for the epoch
        batch_count = 0
        
        for i in range(0, N_train, batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # Convert y_batch to one-hot encoding
            t_one_hot_batch = np.eye(K)[y_batch.flatten()]

            # Forward pass
            z_batch = np.dot(X_batch, W)
            y_batch_pred = softmax(z_batch)

            # Compute loss for this batch
            batch_loss = cross_entropy_loss(y_batch_pred, t_one_hot_batch)
            epoch_loss += batch_loss
            batch_count += 1

            # Gradient computation
            grad_W = np.dot(X_batch.T, (y_batch_pred - t_one_hot_batch)) / X_batch.shape[0]
            grad_W += decay * W  # Regularization term (L2 penalty)

            # Update weights
            W -= alpha * grad_W

        # Evaluate on the current mini-batch (train loss)
        train_loss = epoch_loss / batch_count
        train_losses.append(train_loss)
        

        # Evaluate on validation data
        _, _, _, val_acc = predict(X_val, W, t_val)
        valid_accs.append(val_acc)

        # Track the best model based on validation accuracy
        if val_acc > acc_best:
            acc_best = val_acc
            W_best = W.copy()
            epoch_best = epoch

        print(f"Epoch {epoch+1}/{MaxEpoch} - Train Loss: {train_loss:.4f} - Val Accuracy: {val_acc:.2f}%")

    return epoch_best, acc_best, W_best, train_losses, valid_accs


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)

print(f"Best epoch {epoch_best}-Best Valid Accuracy {acc_best}-Best Test Accuracy {acc_test}")

# Define a plotting function to show the learning curves
def plot_learning_curves(losses_train, risks_val):
    # Ensure that the lengths of losses_train and risks_val are the same
    epochs = range(1, len(losses_train) + 1)
    
    # Check if lengths match
    if len(risks_val) != len(losses_train):
        print("Warning: The lengths of training losses and validation risks don't match.")
        # Adjust `risks_val` or `losses_train` accordingly if necessary
        # For now, you can plot the minimum length for both
        min_len = min(len(losses_train), len(risks_val))
        epochs = range(1, min_len + 1)
        losses_train = losses_train[:min_len]
        risks_val = risks_val[:min_len]

    # Plot the training loss
    plt.figure(figsize=(10, 3))

    # Training loss curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses_train, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Learning Curve: Training Loss')
    plt.grid(True)
    plt.legend()

    # Validation risk curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, risks_val, label='Validation Risk', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Risk (MAE)')
    plt.title('Learning Curve: Validation Risk')
    plt.grid(True)
    plt.legend()

    # Show both plots
    plt.tight_layout()
    plt.show()

# Plot the learning curves for training loss and validation risk
plot_learning_curves(train_losses, valid_accs)

