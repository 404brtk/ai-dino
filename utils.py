import numpy as np
import math
from collections import deque
from typing import List, Tuple

class PrioritizedReplayBuffer:
    """Advanced experience buffer with prioritized sampling and analysis."""
    
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
        if self.step >= self.decay_steps:
            return self.end_eps
        
        epsilon = self.end_eps + (self.start_eps - self.end_eps) * \
                  math.exp(-1. * self.step / self.decay_steps)
        self.step += 1
        return epsilon
