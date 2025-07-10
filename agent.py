import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Tuple, Optional

from utils import PrioritizedReplayBuffer, EpsilonGreedy


class DQN(nn.Module):
    """Deep Q-Network model for processing game frames."""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int):
        """
        Initializes the DQN model with convolutional and fully connected layers.

        Args:
            input_shape (Tuple[int, int, int]): The shape of the input state (C, H, W).
            n_actions (int): The number of possible actions.
        """
        super(DQN, self).__init__()

        # Convolutional layers for feature extraction from frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))  # Ensures consistent output size
        )

        # Calculate conv output size after adaptive pooling
        conv_out_size = 64 * 7 * 7

        # Fully connected layers for Q-value estimation
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the DQN."""
        conv_out = self.conv(x)
        flat = torch.flatten(conv_out, 1)
        return self.fc(flat)


class DQNAgent:
    """Complete DQN Agent with Double DQN and target network."""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int,
                 lr: float = 0.0001, gamma: float = 0.99, 
                 buffer_size: int = 100000, batch_size: int = 32,
                 target_update_freq: int = 1000, device: str = 'cuda'):
        """
        Initialize DQN Agent.
        
        Args:
            input_shape: Shape of input state (C, H, W)
            n_actions: Number of possible actions
            lr: Learning rate
            gamma: Discount factor
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        
        # Networks
        self.q_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network = DQN(input_shape, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.update_target_network()
        
        # Replay buffer and exploration
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.epsilon_greedy = EpsilonGreedy()
        
        # Loss function (using Huber loss for more stability)
        self.criterion = nn.SmoothL1Loss()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy or greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
        
        epsilon = self.epsilon_greedy.get_epsilon() if training else 0.0
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return q_values.argmax().item()

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        # For the first experience, we don't have a TD error yet, so add with max priority
        self.replay_buffer.add((state, action, reward, next_state, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch from replay buffer
        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute TD errors for priority updates
        td_errors = torch.abs(target_q_values - current_q_values.squeeze(1)).detach().cpu().numpy()

        # Update priorities in the replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Compute loss with importance sampling weights
        loss = (weights * self.criterion(current_q_values, target_q_values.unsqueeze(1))).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon_step': self.epsilon_greedy.step,
            'update_count': self.update_count
        }, filepath)

    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon_greedy.step = checkpoint['epsilon_step']
        self.update_count = checkpoint['update_count']

    def get_current_epsilon(self) -> float:
        """Get current epsilon value without incrementing."""
        return self.epsilon_greedy.peek()