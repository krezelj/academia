import torch.nn as nn
import torch

class CartPoleMLP(nn.Module):
    """
    CartPoleMLP class represents a neural network model for the CartPole reinforcement learning task.

    Attributes:
        network (nn.Sequential): A neural network composed of fully connected layers.

    Methods:
        __init__(self): Initializes the CartPoleMLP model by defining its architecture.
        forward(self, x) -> torch.Tensor: Defines the forward pass of the network.

    Usage:
        # Create an instance of the CartPoleMLP class
        cartpole_network = CartPoleMLP()
        
        # Forward pass: Get predicted action for a given state
        input_states = torch.Tensor([sample_states])  # Shape: (num_of_states, state_size)
        predicted_actions = cartpole_network(input_states)  # Shape: (num_of_states, num_of_actions)
        
    """
    def __init__(self):
        """
        Initializes the CartPoleMLP.

        Architecture:
            - Input Layer: Fully connected layer with 4 input features (representing the state space).
            - Hidden Layer 1: Fully connected layer with 120 units and ReLU activation.
            - Hidden Layer 2: Fully connected layer with 84 units and ReLU activation.
            - Output Layer: Fully connected layer with 2 units (representing the actions: left or right).
        """
        super(CartPoleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x) -> torch.Tensor:
        """
        Defines the forward pass of the CartPoleMLP network.

        Args:
            x (torch.Tensor): Input tensor representing the state space of the CartPole environment.
                Shape: (num_of_states, state_size)

        Returns:
            torch.Tensor: Output tensor representing the predicted action Q-values.
                Shape: (num_of_states, num_of_actions)
        """
        return self.network(x)