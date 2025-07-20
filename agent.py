import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from typing import Tuple, Optional

from utils import PrioritizedReplayBuffer, EpsilonGreedy


class DQN(nn.Module):
    """Deep Q-Network model for processing game frames."""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int, numerical_input_size: int = 3, use_numerical: bool = True):
        """
        Initializes the DQN model with convolutional and fully connected layers.

        Args:
            input_shape: Shape of visual input (C, H, W)
            n_actions: Number of possible actions
            numerical_input_size: Size of numerical feature vector
            use_numerical: Whether to use numerical features
        """
        super(DQN, self).__init__()
        self.use_numerical = use_numerical
        
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
        
        if self.use_numerical:
            # Numerical feature processor
            self.numerical_fc = nn.Sequential(
                nn.Linear(numerical_input_size, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Combined feature processor
            combined_size = conv_out_size + 32
        else:
            combined_size = conv_out_size
        
        # Fully connected layers for Q-value estimation
        self.fc = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_actions)
        )

    def forward(self, visual_input: torch.Tensor, numerical_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Defines the forward pass of the DQN."""

        # Process visual features
        conv_out = self.conv(visual_input)
        visual_features = torch.flatten(conv_out, 1)
        
        if self.use_numerical and numerical_input is not None:
            # Process numerical features
            numerical_features = self.numerical_fc(numerical_input)
            
            # Combine features
            combined_features = torch.cat([visual_features, numerical_features], dim=1)
        else:
            combined_features = visual_features
        
        return self.fc(combined_features)


class DQNAgent:
    """Complete DQN Agent with Double DQN and target network."""
    
    def __init__(self, input_shape: Tuple[int, int, int], n_actions: int,
                 lr: float = 0.0001, gamma: float = 0.99, 
                 buffer_size: int = 100000, batch_size: int = 32,
                 target_update_freq: int = 1000, device: str = 'cuda',
                 start_eps: float = 1.0, end_eps: float = 0.01,
                 decay_steps: int = 10000, use_numerical: bool = True):
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
            start_eps: Initial epsilon value
            end_eps: Final epsilon value
            decay_steps: Number of steps to decay from start to end
            use_numerical: Whether to use numerical features
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps
        self.use_numerical = use_numerical

        # Networks
        self.q_network = DQN(input_shape, n_actions, use_numerical=use_numerical).to(self.device)
        self.target_network = DQN(input_shape, n_actions, use_numerical=use_numerical).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Initialize target network
        self.update_target_network()
        
        # Replay buffer and exploration
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.epsilon_greedy = EpsilonGreedy(self.start_eps, self.end_eps, self.decay_steps)
        
        # Loss function (using Huber loss for more stability)
        self.criterion = nn.SmoothL1Loss()

    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _extract_features(self, state):
        """Extract visual and numerical features from state."""
        if isinstance(state, dict):
            # Dictionary observation
            visual_data = state['frame']
            if self.use_numerical:
                numerical_data = np.array([
                    state['distance_to_obstacle'][0],
                    state['obstacle_y_position'][0],
                    state['obstacle_width'][0]
                ], dtype=np.float32)
            else:
                numerical_data = None
        else:
            # Simple array observation (backward compatibility)
            visual_data = state
            numerical_data = None
            
        return visual_data, numerical_data
      

    def select_action(self, state, training: bool = True) -> int:
        """
        Select action using epsilon-greedy or greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon-greedy)
            
        Returns:
            Selected action
        """
        visual_data, numerical_data = self._extract_features(state)
        
        visual_tensor = torch.FloatTensor(visual_data).unsqueeze(0).to(self.device)
        numerical_tensor = None
        if numerical_data is not None:
            numerical_tensor = torch.FloatTensor(numerical_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(visual_tensor, numerical_tensor).squeeze(0)
        
        epsilon = self.epsilon_greedy.get_epsilon() if training else 0.0
        
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return q_values.argmax().item()


    def store_experience(self, state, action: int, reward: float, next_state, done: bool):
        """Store experience with feature extraction."""
        # Extract features for both states
        visual_state, numerical_state = self._extract_features(state)
        visual_next_state, numerical_next_state = self._extract_features(next_state)
        
        # Store as tuple (visual, numerical) for each state
        state_tuple = (visual_state, numerical_state)
        next_state_tuple = (visual_next_state, numerical_next_state)
        
        self.replay_buffer.add((state_tuple, action, reward, next_state_tuple, done))

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        state_tuples, actions, rewards, next_state_tuples, dones = zip(*experiences)

        # Separate visual and numerical data
        visual_states = []
        numerical_states = []
        visual_next_states = []
        numerical_next_states = []

        for (vis_state, num_state), (vis_next_state, num_next_state) in zip(state_tuples, next_state_tuples):
            visual_states.append(vis_state)
            numerical_states.append(num_state if num_state is not None else np.zeros(3))
            visual_next_states.append(vis_next_state)
            numerical_next_states.append(num_next_state if num_next_state is not None else np.zeros(3))

        # Convert to tensors
        visual_states = torch.FloatTensor(np.array(visual_states)).to(self.device)
        numerical_states = torch.FloatTensor(np.array(numerical_states)).to(self.device) if self.use_numerical else None
        visual_next_states = torch.FloatTensor(np.array(visual_next_states)).to(self.device)
        numerical_next_states = torch.FloatTensor(np.array(numerical_next_states)).to(self.device) if self.use_numerical else None
        
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q values
        current_q_values = self.q_network(visual_states, numerical_states).gather(1, actions.unsqueeze(1))

        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(visual_next_states, numerical_next_states).argmax(1)
            next_q_values = self.target_network(visual_next_states, numerical_next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute TD errors
        td_errors = torch.abs(target_q_values - current_q_values.squeeze(1)).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)

        # Compute loss
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