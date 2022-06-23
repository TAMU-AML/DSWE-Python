import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

from ._DNN_subroutine import *


class DNNPowerCurve(object):
    """
    Parameters
    ----------
    feats_scale: int
        It increases number of neurons in each layer by multiplying the number of neurons with its value.

    train_all: bool
        A boolean to specify whether to train model on the entire dataset and do not separate dataset for cross-validation).

    n_folds: int
        Number of folds for cross_validation.

    batch_size: int
        Number of batch_size in the training.

    lr: float
        Learning rate.

    n_epochs: int
        Number of epochs to the train the model.

    optimizer: string
        A string specifying which optimization algorithm to be used to update model weights.
        The best working optimizers are ['sgd', 'adam', 'rmsprop', 'adagrad'].

    loss_fn: string
        A string specifying which loss functions to be used to compute the errors.
        The common loss functions for continuous labels are ['l1_loss', 'l2_loss']. 

    momentum: float
        Momentum value for 'sgd' and 'rmsprop' optimizers. Default value is set to 0.

    print_loss: bool
        A boolean to specify whether to print loss value after each epoch during training.

    save_fig: bool
        A boolean to specify whether to plot and save loss values during training and validation.
        The plot(s) save in a pdf format.
    """

    def __init__(self, feats_scale=12,
                 train_all=False,
                 n_folds=5,
                 batch_size=64,
                 lr=0.1,
                 n_epochs=30,
                 optimizer='sgd',
                 loss_fn='l2_loss',
                 momentum=0.8,
                 print_log=False,
                 save_fig=True):

        if not isinstance(feats_scale, int):
            raise ValueError("The feats_scale must be an integer value.")

        if type(train_all) != type(True):
            raise ValueError("The train_all must be either True or False.")

        if not isinstance(n_folds, int):
            raise ValueError("The n_folds must be an integer value.")

        if not isinstance(batch_size, int):
            raise ValueError("The batch_size must be an integer value.")

        if not (isinstance(lr, int) or isinstance(lr, float)) and lr > 0:
            raise ValueError("The lr must be a numeric value greater than 0.")

        if not isinstance(n_epochs, int):
            raise ValueError("The n_epochs must be an integer value.")

        if optimizer not in ['sgd', 'adam', 'rmsprop', 'adagrad']:
            raise ValueError(
                "The optimizer can be 'sgb' or 'adam' or'rmsprop' or 'adagrad'.")

        if loss_fn not in ['l1_loss', 'l2_loss']:
            raise ValueError("The loss_fn must be 'L-BFGS-B' or 'BFGS'.")

        if not (isinstance(momentum, int) or isinstance(momentum, float)) and momentum > 0:
            raise ValueError(
                "The momentum must be a numeric value greater than 0.")

        if type(print_log) != type(True):
            raise ValueError("The print_log must be either True or False.")

        if type(save_fig) != type(True):
            raise ValueError("The save_fig must be either True or False.")

        self.feats_scale = feats_scale
        self.train_all = train_all
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.momentum = momentum
        self.print_log = print_log
        self.save_fig = save_fig

        self.kfold = KFold(n_splits=n_folds, shuffle=True)
        self.device = get_default_device()

    def train(self, X_train, y_train):
        """
        Parameters
        ----------
        X_train: np.ndarray or pd.DataFrame
            A matrix or dataframe of input variable values in the training dataset.

        y_train: np.array
            A numeric array for response values in the training dataset.
        """

        if not (isinstance(X_train, list) or isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series) or isinstance(X_train, np.ndarray)):
            raise ValueError(
                "The X_train should be either a list or numpy array or dataframe.")

        if not (isinstance(y_train, list) or isinstance(y_train, np.ndarray)) or isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
            raise ValueError(
                "The target data should be either a list or numpy array or dataframe.")

        if len(X_train) != len(y_train):
            raise ValueError(
                "The X_train and y_train should have same number of data points.")

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        if len(self.X_train.shape) == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        self.y_train = self.y_train.reshape(-1, 1)

        self.n_feats = self.X_train.shape[1]

        self.scaler_features = StandardScaler()
        self.scaler_features.fit(self.X_train)
        self.X_train = self.scaler_features.transform(self.X_train)

        self.scaler_target = StandardScaler()
        self.scaler_target.fit(self.y_train)
        self.y_train = self.scaler_target.transform(self.y_train)

        self.dataset = []
        for i in range(self.X_train.shape[0]):
            self.dataset.append(
                [Tensor(self.X_train[i]), Tensor(self.y_train[i])])

        if self.train_all:
            print("--Initiating training on the entire dataset--")
            train_sampler = SubsetRandomSampler(
                list(range(self.X_train.shape[0])))
            dl = DataLoader(self.dataset,
                            self.batch_size,
                            sampler=train_sampler)
            self.model = NeuralNets(
                n_feats=self.n_feats, feats_scale=self.feats_scale)
            to_device(self.model, self.device)

            history = train_model(self.n_epochs, self.model, dl, [], self.batch_size, self.device,
                                  self.loss_fn, self.optimizer, self.lr, self.momentum, print_log=self.print_log)
            self.model, self.train_losses, self.val_losses = history

            if self.save_fig:
                plot_losses([self.train_losses], [])

        else:
            self.cv_train_loss = []
            self.cv_val_loss = []
            cv_loss = 0
            for fold, (train_indices, val_indices) in enumerate(self.kfold.split(self.dataset)):
                print("--Initiating training for fold {}--".format(fold + 1))
                train_sampler = SubsetRandomSampler(train_indices)
                train_dl = DataLoader(self.dataset,
                                      self.batch_size,
                                      sampler=train_sampler)
                val_sampler = SubsetRandomSampler(val_indices)
                val_dl = DataLoader(self.dataset,
                                    self.batch_size,
                                    sampler=val_sampler)

                self.model = NeuralNets(
                    n_feats=self.n_feats, feats_scale=self.feats_scale)
                to_device(self.model, self.device)

                history = train_model(self.n_epochs, self.model, train_dl, val_dl, self.batch_size, self.device,
                                      self.loss_fn, self.optimizer, self.lr, self.momentum, print_log=self.print_log)
                self.model, self.train_losses, self.val_losses = history

                self.cv_train_loss.append(self.train_losses)
                self.cv_val_loss.append(self.val_losses)

                """calculating val loss on unscaled predictions and labels"""
                predictions = []
                labels = []
                for batch_idx, (data, target) in enumerate(val_dl):
                    data, target = data.to(self.device), target.to(self.device)
                    pred = self.scaler_target.inverse_transform(
                        self.model(data).cpu().detach().numpy())
                    label = self.scaler_target.inverse_transform(target)
                    predictions.extend(pred)
                    labels.extend(label)
                predictions = np.array(predictions)
                labels = np.array(labels)

                cv_loss += np.sqrt(np.square(predictions - labels).mean())

                print("Fold {} val loss: {:.3f}".format(
                    fold + 1, np.sqrt(np.square(predictions - labels).mean())))

            if self.save_fig:
                plot_losses(self.cv_train_loss, self.cv_val_loss)

            print(
                "Average loss on 5-fold cross-validation: {:.3f}".format(cv_loss / self.n_folds))

        print("Everything done!!")

    def predict(self, X_test):
        """
        Parameters
        ----------
        X_test: np.ndarray or pd.DataFrame
            A matrix or dataframe of test input variable values to compute predictions.

        Returns
        -------
        np.array
            A numeric array for predictions at the data points in X_test.
        """

        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        if len(self.X_train.shape) > 1:
            if X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    "The number of features in train and test set must be same.")

        X_test = self.scaler_features.transform(X_test)

        test_dataset = []
        for i in range(X_test.shape[0]):
            test_dataset.append(Tensor(X_test[i]))
        test_dl = DataLoader(test_dataset, self.batch_size)

        predictions = []
        for batch_idx, data in enumerate(test_dl):
            data = data.to(self.device)
            pred = self.scaler_target.inverse_transform(
                self.model(data).cpu().detach().numpy())
            predictions.extend(pred)
        predictions = np.array(predictions).squeeze()

        return predictions

    def calculate_rmse(self, X_test, y_test):
        """
        Parameters
        ----------
        X_test: np.ndarray or pd.DataFrame
            A matrix or dataframe of test input variable values to compute predictions.

        y_test: np.array
            A numeric array for response values in the testing dataset.

        Returns
        -------
        float
            A numeric Root Mean Square Error (RMSE) value calculated on the X_test.
        """
        if not (isinstance(X_test, list) or isinstance(X_test, pd.DataFrame) or isinstance(X_test, pd.Series) or isinstance(X_test, np.ndarray)):
            raise ValueError(
                "The X_test should be either a list or numpy array or dataframe.")

        if not (isinstance(y_test, list) or isinstance(y_test, np.ndarray)) or isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
            raise ValueError(
                "The target data should be either a list or numpy array or dataframe.")

        if len(X_test) != len(y_test):
            raise ValueError(
                "The X_test and y_test should have same number of data points.")

        X_test = np.array(X_test)
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        if len(self.X_train.shape) > 1:
            if X_test.shape[1] != self.X_train.shape[1]:
                raise ValueError(
                    "The number of features in train and test set must be same.")

        X_test = self.scaler_features.transform(X_test)
        y_test = y_test.reshape(-1, 1)

        test_dataset = []
        for i in range(X_test.shape[0]):
            test_dataset.append(Tensor(X_test[i]))
        test_dl = DataLoader(test_dataset, self.batch_size)

        predictions = []
        for batch_idx, data in enumerate(test_dl):
            data = data.to(self.device)
            pred = self.scaler_target.inverse_transform(
                self.model(data).cpu().detach().numpy())
            predictions.extend(pred)
        predictions = np.array(predictions)

        rmse = np.sqrt(np.square(predictions - y_test).mean())

        return rmse
