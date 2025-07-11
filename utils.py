import numpy as np
import math
import cv2
from collections import deque
from typing import List, Tuple, Optional

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


class FrameProcessor:
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
        return np.stack(self.frame_buffer, axis=0)
    
    def reset(self):
        """Reset the frame buffer."""
        self.frame_buffer.clear()