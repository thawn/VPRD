import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as pl
import numpy as np


class EnergyPredictionBaseModel(pl.LightningModule):
    """
    Base model for energy prediction.

    This class provides the basic structure and functionality for training, validation, testing,
    and prediction steps in an energy prediction model.

    Attributes
    ----------
    mean_energy : torch.Tensor
        The mean energy of the training data.
    penalty_weight : float
        The weight of the penalty term in the loss function.
    model : torch.nn.Module
        The model for energy prediction.
    lr : float
        The learning rate for the optimizer.
    learning_rate_scheduler_patience : int
        The patience for the learning rate scheduler.
    learning_rate_scheduler_factor : float
        The factor for the learning rate scheduler.
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.learning_rate_scheduler_patience, factor=self.learning_rate_scheduler_factor),
            'name': 'lr_scheduler',
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.loss_fn(output, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.loss_fn(output, target)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self.model(data)
        loss = self.loss_fn(output, target)
        self.log('test_loss', loss)

    def predict_step(self, batch):
        data, _ = batch
        return self.model(data)

    def get_num_parameters(self):
        """
        Returns the number of parameters in the model.

        Returns
        -------
        num_params : int
            The number of parameters in the model.

        """
        return self.model.get_num_parameters()

    def initialize_loss_function(self, training_data_loader: DataLoader, penalty_weight: float = 0.5):
        """
        Sets the loss function to use for training.
        We need the training data to compute the mean energy of the training data.
        The loss function adds a penalty for predictions that are close to the mean energy.

        Parameters
        ----------
        training_data_loader : DataLoader
            The DataLoader containing the training data.
        penalty_weight : float, optional
            The weight of the penalty term in the loss function (default: 0.5).

        """
        np_energy = np.asarray([target for _, target in training_data_loader.dataset])
        self.mean_energy = torch.tensor(np.mean(np_energy, axis=0), dtype=torch.float32, device=self.device)
        self.penalty_weight = penalty_weight

    def loss_fn(self, output, target):
        """
        Computes the loss for the given output and target.

        Parameters
        ----------
        output : torch.Tensor
            The predicted output values.
        target : torch.Tensor
            The target values.

        Returns
        -------
        loss : torch.Tensor
            The computed loss value.

        """
        if self.penalty_weight == 0:
            return nn.MSELoss()(output, target)
        else:
            self.mean_energy = self.mean_energy.to(self.device)
            batch_mean_energy = self.mean_energy.repeat(output.shape[0], 1)
            return nn.MSELoss()(output, target) - self.penalty_weight * nn.MSELoss()(output, batch_mean_energy)


class EnergyPredictionMLP(nn.Module):
    """
    A linear network model for energy prediction.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor.
    output_shape : tuple
        The shape of the output tensor.
    hidden_layers : int, optional
        The number of hidden layers in the network. Defaults to 1.
    dropout : float, optional
        The dropout probability. Defaults to 0.5.

    Attributes
    ----------
    layers : torch.nn.Sequential
        A sequence of linear layers with ReLu activation and Dropout.
    """

    def __init__(self, input_shape, output_shape, hidden_layers=1, dropout=0.5):
        super(EnergyPredictionMLP, self).__init__()
        params = np.linspace(input_shape[0], output_shape[0], num=hidden_layers + 2).astype(int)
        self.layers = nn.Sequential()
        for i in range(hidden_layers):
            self.layers.add_module(
                f'linear_{i}', nn.Linear(params[i], params[i + 1]))
            self.layers.add_module(f'relu_{i}', nn.ReLU())
            if dropout > 0:
                # add a dropout layer
                self.layers.add_module(f'dropout_{i}', nn.Dropout(dropout))
        self.layers.add_module(f'output', nn.Linear(
            params[hidden_layers], output_shape[0]))

    def forward(self, x):
        """
        Forward pass of the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        return self.layers(x)

    def get_num_parameters(self):
        """
        Get the total number of trainable parameters in the network.

        Returns
        -------
        int
            The number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnergyPredictionMLPModel(EnergyPredictionBaseModel):
    """
    Linear regression model for energy prediction.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    output_shape : tuple
        The shape of the output data.
    hidden_layers : int, optional
        The number of hidden layers in the linear network. Defaults to 1.
    dropout : float, optional
        The dropout rate. Defaults to 0.5.
    lr : float, optional
        The learning rate for the optimizer. Defaults to 1e-3.
    learning_rate_scheduler_patience : int, optional
        The patience for the learning rate scheduler. Defaults to 50.
    learning_rate_scheduler_factor : float, optional
        The factor for the learning rate scheduler. Defaults to 0.5.

    Attributes
    ----------
    model : EnergyPredictionLinearNetwork
        The linear network model for energy prediction.
    """

    def __init__(self, input_shape, output_shape, hidden_layers=1, dropout=0.5, lr=1e-3,
                 learning_rate_scheduler_patience=50, learning_rate_scheduler_factor=0.5):
        super(EnergyPredictionMLPModel, self).__init__()
        self.model = EnergyPredictionMLP(input_shape, output_shape, hidden_layers, dropout)
        self.lr = lr
        self.learning_rate_scheduler_patience = learning_rate_scheduler_patience
        self.learning_rate_scheduler_factor = learning_rate_scheduler_factor
        self.save_hyperparameters()


class ConvLayers(nn.Module):
    """
    A class representing a stack of convolutional layers.

    Parameters
    ----------
    in_channels : int
        The number of channels in the input tensor.
    out_channels : int
        The number of channels in the output tensor.
    kernel_size : int
        The size of the convolutional kernel.
    layers : int
        The number of convolutional layers.


    Attributes:
    -----------
    conv_layers: torch.nn.Sequential
        The sequential container that holds the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, layers):
        super().__init__()
        self.conv_layers = nn.Sequential()
        pad = kernel_size // 2
        for i in range(layers):
            self.conv_layers.add_module(f'conv_{i}', nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad))
            self.conv_layers.add_module(f'batchnorm_{i}', nn.BatchNorm1d(out_channels))
            self.conv_layers.add_module(f'relu_{i}', nn.ReLU())
            in_channels = out_channels

    def forward(self, x):
        """
        Performs forward pass through the convolutional layers.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to be passed through the network layers.

        Returns
        -------
        torch.Tensor
            The output tensor after passing through the network layers.

        """
        return self.conv_layers(x)


class EnergyPredictionConv1DNetwork(nn.Module):
    """
    A convolutional neural network model for energy prediction.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor.
    output_shape : tuple
        The shape of the output tensor.
    kernel_size : int, optional
        The size of the convolutional kernel. Defaults to 3.

    Attributes
    ----------
    linear : nn.Sequential
        The linear layers of the network.
    layers : nn.Sequential
        The convolutional layers of the network.
    output_shape : int
        The shape of the output tensor.
    """

    def __init__(self, input_shape, output_shape, kernel_size=3):
        super(EnergyPredictionConv1DNetwork, self).__init__()
        layers = np.ceil(np.emath.logn(kernel_size, output_shape[0])).astype(int)
        channels = np.linspace(input_shape[0], 1, num=layers + 2).astype(int)
        channels = channels * 2
        self.linear = nn.Sequential()
        self.linear.add_module('linear', nn.Linear(input_shape[0], channels[0]))
        self.linear.add_module('relu', nn.ReLU())
        self.linear.add_module('dropout', nn.Dropout(0.5))
        self.layers = nn.Sequential()
        for i in range(layers):
            self.layers.add_module(f'conv_transpose_{i}', nn.ConvTranspose1d(
                channels[i], channels[i + 1], kernel_size, stride=kernel_size))
            self.layers.add_module(f'double_conv_{i}', ConvLayers(channels[i + 1], channels[i + 1], kernel_size, 2))
        self.layers.add_module('output', nn.Conv1d(channels[layers], 1, kernel_size, padding=kernel_size // 2))
        self.output_shape = output_shape[0]

    def forward(self, x):
        """
        Performs a forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x = self.linear(x)
        return torch.squeeze(self.layers(x[:, :, None])[:, :, :self.output_shape])

    def get_num_parameters(self):
        """
        Returns the total number of trainable parameters in the network.

        Returns
        -------
        int
            The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnergyPredictionConv1DModel(EnergyPredictionBaseModel):
    """
    A class representing a convolutional neural network model for energy prediction.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input data.
    output_shape : tuple
        The shape of the output data.
    kernel_size : int, optional
        The size of the convolutional kernel. Defaults to 3.
    lr : float, optional
        The learning rate for the optimizer. Defaults to 0.001.
    learning_rate_scheduler_patience : int, optional
        The patience for the learning rate scheduler. Defaults to 50.
    learning_rate_scheduler_factor : float, optional
        The factor for the learning rate scheduler. Defaults to 0.5.

    Attributes
    ----------
    model : EnergyPredictionConv1DNetwork
        The convolutional neural network model for energy prediction.
    """

    def __init__(self, input_shape, output_shape, kernel_size=3, lr=0.001,
                 learning_rate_scheduler_patience=20, learning_rate_scheduler_factor=0.1):
        super(EnergyPredictionConv1DModel, self).__init__()
        self.model = EnergyPredictionConv1DNetwork(
            input_shape, output_shape, kernel_size)
        self.lr = lr
        self.learning_rate_scheduler_patience = learning_rate_scheduler_patience
        self.learning_rate_scheduler_factor = learning_rate_scheduler_factor
        self.save_hyperparameters()
