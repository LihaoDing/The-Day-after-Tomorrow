import torch.nn as nn
import torch


__all__ = ['HybridNN']


class HybridNN(nn.Module):
    """
        Initializes the HybridNN model with CNN, LSTM, and FC layers.
    """
    def __init__(self):
        super(HybridNN, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        self.fc_size = self._get_conv_output((1, 366, 366))

        self.lstm = nn.LSTM(input_size=1, hidden_size=20, batch_first=True)

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_size + 20, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def _get_conv_output(self, shape):
        """
        Determines the size of the flattened CNN output for a given input
        shape.

        Args:
            shape (tuple): The shape of the input to the CNN layers, excluding
            batch size.

        Returns:
            int: The size of the flattened output tensor.

        Example:
            Assuming a single-channel 366x366 image input, the output size is
            calculated.
            >>> model = HybridNN()
            >>> model._get_conv_output((1, 366, 366))
            46656  # This is an example output and may vary based on the model
            architecture.
        """
        with torch.no_grad():
            batch_size = 1
            input = torch.autograd.Variable(torch.rand(batch_size, *shape))
            output_feat = self.cnn_layers(input)
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size

    def forward(self, x_img, x_relative_time):
        """
        Defines the forward pass of the HybridNN model.

        Args:
            x_img (torch.Tensor): The input tensor for image data with shape
            (batch_size, channels, height, width).
            x_relative_time (torch.Tensor): The input tensor for sequential
            data with shape (batch_size, sequence_length, features).

        Returns:
            torch.Tensor: The output tensor of the model.

        Example:
            This example provides a schematic idea and won't run as is due to
            the need for actual input tensors.
            >>> model = HybridNN()
            >>> x_img = torch.rand(4, 1, 366, 366)  # Example image input
            tensor
            >>> x_relative_time = torch.rand(4, 10, 1)  # Example sequential
            input tensor
            >>> output = model(x_img, x_relative_time)
            >>> output.shape
            torch.Size([4, 1])  # The expected shape of the output tensor for
            a batch of 4
        """
        img_features = self.cnn_layers(x_img)
        _, (h_n, _) = self.lstm(x_relative_time)
        time_features = h_n[-1]
        combined_features = torch.cat((img_features, time_features), dim=1)
        output = self.fc_layers(combined_features)
        return output
