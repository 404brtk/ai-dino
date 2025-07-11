import numpy as np
import math
import cv2
from collections import deque
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self):
        # Environment parameters
        self.headless = False
        self.frame_stack = 4
        self.frame_skip = 2
        self.processed_size = (84, 84)
        self.max_episode_steps = 10000
        self.reward_scale = 1.0
        
        # Agent parameters
        self.lr = 0.0001
        self.gamma = 0.99
        self.buffer_size = 100000
        self.batch_size = 32
        self.target_update_freq = 1000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.start_eps = 1.0
        self.end_eps = 0.01
        self.decay_steps = 15000
        
        # Training parameters
        self.total_episodes = 10000
        self.warmup_episodes = 40
        self.train_freq = 4
        self.eval_freq = 50
        self.save_freq = 100
        self.log_freq = 1
        
        # Paths
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        self.results_dir = 'results'



class TrainingMetrics:
    """Class to track and manage training metrics."""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_scores: List[int] = []
        self.episode_lengths: List[int] = []
        self.losses: List[float] = []
        self.epsilon_values: List[float] = []
        
        # Running averages
        self.avg_reward_window = 100
        self.avg_score_window = 100
        
    def add_episode(self, reward: float, score: int, length: int, epsilon: float):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        
    def add_loss(self, loss: float):
        """Add training loss."""
        if loss is not None:
            self.losses.append(loss)
    
    def get_recent_avg_reward(self) -> float:
        """Get average reward over recent episodes."""
        if len(self.episode_rewards) == 0:
            return 0.0
        recent_rewards = self.episode_rewards[-self.avg_reward_window:]
        return np.mean(recent_rewards)
    
    def get_recent_avg_score(self) -> float:
        """Get average score over recent episodes."""
        if len(self.episode_scores) == 0:
            return 0.0
        recent_scores = self.episode_scores[-self.avg_score_window:]
        return np.mean(recent_scores)
    
    def get_best_score(self) -> int:
        """Get the best score achieved."""
        return max(self.episode_scores) if self.episode_scores else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for saving."""
        return {
            # Core training data
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon_values': self.epsilon_values,

            # Summary statistics
            'best_score': self.get_best_score(),
            'final_avg_reward': self.get_recent_avg_reward(),
            'final_avg_score': self.get_recent_avg_score()
        }

class PrioritizedReplayBuffer:
    """Advanced experience buffer with prioritized sampling."""
    
    def __init__(self, 
                 capacity: int = 100000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Initialize prioritized experience buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.position = 0

    def add(self, experience: Tuple, priority: float = None):
        """Add experience with priority."""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small epsilon to avoid zero priority

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return len(self.buffer)


class EpsilonGreedy:
    """Epsilon-greedy exploration strategy with decay."""
    
    def __init__(self, start_eps: float = 1.0, end_eps: float = 0.01, 
                 decay_steps: int = 10000):
        """
        Initialize epsilon-greedy strategy.
        
        Args:
            start_eps: Initial epsilon value
            end_eps: Final epsilon value
            decay_steps: Number of steps to decay from start to end
        """
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = decay_steps
        self.step = 0

    def peek(self) -> float:
        """Get current epsilon value without incrementing step."""
        if self.step >= self.decay_steps:
            return self.end_eps
        
        return self.end_eps + (self.start_eps - self.end_eps) * \
                  math.exp(-1. * self.step / self.decay_steps)

    def get_epsilon(self) -> float:
        """Get current epsilon value and increment step."""
        epsilon = self.peek()
        self.step += 1
        return epsilon


class FrameProcessor:
    """Frame processing utilities for game observations."""
    
    def __init__(self, 
                target_size: Tuple[int, int] = (84, 84),
                stack_frames: int = 4):
        """
        Initialize frame processor.
        
        Args:
            target_size: (width, height) to resize frames to
            stack_frames: Number of consecutive frames to stack
        """
        self.target_size = target_size
        self.stack_frames = stack_frames
        self.frame_buffer = deque(maxlen=stack_frames)
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame as numpy array (H, W, C)
            
        Returns:
            Processed frame as numpy array (C, H, W)
        """
        # Convert to grayscale if the image is in color (3 dimensions)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Resize
        frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Add to frame buffer
        self.frame_buffer.append(frame)
        
        # For initial frames, pad with zeros
        while len(self.frame_buffer) < self.stack_frames:
            self.frame_buffer.appendleft(np.zeros_like(frame))
            
        # Convert deque to list for numpy stacking
        return np.stack(list(self.frame_buffer), axis=0)
    
    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()


class TensorBoardLogger:
    """A simple logger for TensorBoard visualization."""


    def __init__(self, log_dir: str):
        """
        Initializes the TensorBoard logger.

        Args:
            log_dir (str): The directory to save TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir)


    def log_scalar(self, tag: str, value: float, step: int):
        """Logs a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)


    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()