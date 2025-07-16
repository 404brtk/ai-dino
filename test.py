import os
import time
import argparse
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional

from environment import DinoGameEnvironment
from agent import DQNAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("dino_test")

class DinoTester:
    """Class for testing a trained DQN agent on the Chrome Dino game."""
    
    def __init__(self, model_path: str, headless: bool = False, device: str = 'auto'):
        """
        Initialize the tester with a trained model.
        
        Args:
            model_path: Path to the trained model (.pth file)
            headless: Whether to run in headless mode
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.logger = logging.getLogger("dino_test")
        self.model_path = model_path
        
        # Validate model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.logger.info(f"Using device: {self.device}")
        
        # Environment settings
        self.frame_stack = 4
        self.processed_size = (84, 84)
        
        # Initialize environment
        self.logger.info("Initializing environment...")
        self.env = DinoGameEnvironment(
            headless=headless,
            frame_stack=self.frame_stack,
            processed_size=self.processed_size,
            max_episode_steps=100000  # Set a high value for testing
        )
        
        # Initialize and load agent
        self._init_agent()
        
    def _init_agent(self):
        """Initialize the agent and load the model."""
        self.logger.info(f"Loading model from: {self.model_path}")
        
        # Create agent
        input_shape = (self.frame_stack, *self.processed_size)
        self.agent = DQNAgent(
            input_shape=input_shape,
            n_actions=self.env.action_space.n,
            device=self.device
        )
        
        # Load model
        self.agent.load_model(self.model_path)
        self.logger.info("Model loaded successfully")
    
    def run_episode(self, render: bool = True, delay: Optional[float] = None) -> Dict[str, Any]:
        """
        Run a single test episode.
        
        Args:
            render: Whether to render the game (print progress)
            delay: Optional delay between steps in seconds to visualize gameplay
            
        Returns:
            Episode results dictionary
        """
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_score = 0
        done = False
        
        self.logger.info("Starting test episode...")
        
        start_time = time.time()
        
        while not done:
            # Select action (no training, so epsilon = 0)
            action = self.agent.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update stats
            episode_reward += reward
            episode_steps += 1
            episode_score = info.get('score', 0)
            
            # Optional rendering
            if render and episode_steps % 10 == 0:
                self.logger.info(f"Step: {episode_steps}, Score: {episode_score}, Action: {action}")
            
            # Optional delay for visualization
            if delay is not None:
                time.sleep(delay)
            
            # Update state
            state = next_state
        
        # Calculate duration and speed
        duration = time.time() - start_time
        steps_per_sec = episode_steps / duration if duration > 0 else 0
        
        results = {
            'reward': episode_reward,
            'score': episode_score,
            'steps': episode_steps,
            'duration': duration,
            'steps_per_sec': steps_per_sec
        }
        
        self.logger.info("Episode complete!")
        self.logger.info(f"Final score: {episode_score}")
        self.logger.info(f"Steps: {episode_steps}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        
        return results
    
    def test(self, num_episodes: int = 5, render: bool = True, delay: Optional[float] = None) -> Dict[str, Any]:
        """
        Run multiple test episodes and report statistics.
        
        Args:
            num_episodes: Number of episodes to run
            render: Whether to render the game
            delay: Optional delay between steps in seconds
            
        Returns:
            Dictionary with test statistics
        """
        self.logger.info(f"Running {num_episodes} test episodes...")
        
        scores = []
        rewards = []
        steps = []
        
        try:
            for episode in range(1, num_episodes + 1):
                self.logger.info(f"\n--- Episode {episode}/{num_episodes} ---")
                result = self.run_episode(render=render, delay=delay)
                
                scores.append(result['score'])
                rewards.append(result['reward'])
                steps.append(result['steps'])
                
                self.logger.info(f"Episode {episode} complete | Score: {result['score']}\n")
        
        except KeyboardInterrupt:
            self.logger.info("Testing interrupted by user")
        
        finally:
            # Calculate statistics
            stats = {
                'episodes_completed': len(scores),
                'avg_score': np.mean(scores) if scores else 0,
                'max_score': np.max(scores) if scores else 0,
                'min_score': np.min(scores) if scores else 0,
                'std_score': np.std(scores) if scores else 0,
                'avg_reward': np.mean(rewards) if rewards else 0,
                'avg_steps': np.mean(steps) if steps else 0
            }
            
            # Display results
            self.logger.info("\n--- Test Results ---")
            self.logger.info(f"Episodes completed: {stats['episodes_completed']}/{num_episodes}")
            self.logger.info(f"Average score: {stats['avg_score']:.2f}")
            self.logger.info(f"Max score: {stats['max_score']}")
            self.logger.info(f"Min score: {stats['min_score']}")
            self.logger.info(f"Score std dev: {stats['std_score']:.2f}")
            self.logger.info(f"Average reward: {stats['avg_reward']:.2f}")
            self.logger.info(f"Average steps: {stats['avg_steps']:.2f}")
            
            # Close environment
            self.env.close()
            
            return stats

def main():
    """Parse arguments and run test."""
    parser = argparse.ArgumentParser(description="Test a trained DQN agent on Chrome Dino game")
    
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to the model file (.pth)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of test episodes to run')
    parser.add_argument('--headless', action='store_true',
                       help='Run in headless mode (no browser UI)')
    parser.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                       default='auto', help='Device to use')
    parser.add_argument('--delay', type=float, default=None,
                       help='Delay between steps in seconds (to slow down visualization)')
    
    args = parser.parse_args()
    
    # Create tester and run tests
    tester = DinoTester(
        model_path=args.model,
        headless=args.headless,
        device=args.device
    )
    
    tester.test(
        num_episodes=args.episodes,
        delay=args.delay
    )

if __name__ == '__main__':
    main()
