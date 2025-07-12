import os
import glob
import time
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import torch

from environment import DinoGameEnvironment
from agent import DQNAgent
from utils import TrainingConfig, TrainingMetrics, TensorBoardLogger

# Configure logging to use console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Silence the environment logger to reduce console noise during training
logging.getLogger('environment').setLevel(logging.CRITICAL + 1)

class DinoTrainer:
    """Main training class for the Dino game DQN agent."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Initialize environment
        self.logger.info("Initializing environment...")
        self.env = DinoGameEnvironment(
            headless=config.headless,
            frame_stack=config.frame_stack,
            frame_skip=config.frame_skip,
            processed_size=config.processed_size,
            max_episode_steps=config.max_episode_steps,
            reward_scale=config.reward_scale
        )
        
        # Initialize agent
        self.logger.info(f"Initializing agent on device: {config.device}")
        input_shape = (config.frame_stack, *config.processed_size)
        self.agent = DQNAgent(
            input_shape=input_shape,
            n_actions=self.env.action_space.n,
            lr=config.lr,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            target_update_freq=config.target_update_freq,
            device=config.device,
            start_eps=config.start_eps,
            end_eps=config.end_eps,
            decay_steps=config.decay_steps
        )
        
        # Initialize logging and metrics
        self.tb_logger = TensorBoardLogger(config.log_dir)
        self.metrics = TrainingMetrics()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_avg_score = 0.0
        
        self.logger.info("Trainer initialized successfully!")
    
    def run_episode(self, training: bool = True) -> Dict[str, Any]:
        """Run a single episode."""
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_score = 0
        
        while True:
            # Select action
            action = self.agent.select_action(state, training=training)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            episode_score = info.get('score', 0)
            self.total_steps += 1
            
            # Store experience for training
            if training:
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Train agent
                if (self.total_steps % self.config.train_freq == 0 and 
                    self.episode >= self.config.warmup_episodes):
                    loss = self.agent.train_step()
                    self.metrics.add_loss(loss)
            
            state = next_state
            
            if done:
                break
        
        return {
            'reward': episode_reward,
            'score': episode_score,
            'steps': episode_steps,
            'epsilon': self.agent.get_current_epsilon()
        }
    
    def evaluate_agent(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        self.logger.info(f"Evaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_scores = []
        
        for _ in range(num_episodes):
            result = self.run_episode(training=False)
            eval_rewards.append(result['reward'])
            eval_scores.append(result['score'])
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_score': np.mean(eval_scores),
            'max_score': np.max(eval_scores),
            'std_reward': np.std(eval_rewards),
            'std_score': np.std(eval_scores)
        }
    

    def save_checkpoint(self):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always save regular checkpoint
        model_path = os.path.join(self.config.checkpoint_dir, f'model_ep{self.episode}.pth')
        self.agent.save_model(model_path)
        
        # Determine if this should be saved as "best" using our criteria
        should_save_as_best = self._evaluate_best_model_criteria()
        
        if should_save_as_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            self.agent.save_model(best_path)
            self.best_avg_score = self.metrics.get_recent_avg_score()
            self.logger.info(f"New best model saved! Avg score: {self.best_avg_score:.2f}")
        
        # Save training state
        state = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_avg_score': self.best_avg_score,
            'timestamp': timestamp,
            'config': self.config.__dict__,
            'metrics': self.metrics.to_dict()
        }
        
        state_path = os.path.join(self.config.checkpoint_dir, f'training_state_ep{self.episode}.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(keep_last=3)
        
        self.logger.info(f"Checkpoint saved: episode {self.episode}")
    
    def _evaluate_best_model_criteria(self) -> bool:
        """Multi-criteria evaluation for best model."""
        # Criterion 1: Minimum episodes threshold
        if self.episode < 50:
            return False
    
        # Criterion 2: Recent average improvement
        current_avg = self.metrics.get_recent_avg_score()
        avg_improved = current_avg > self.best_avg_score
    
        # Criterion 3: Best single score improvement
        current_best_single = self.metrics.get_best_score()
        single_score_improved = current_best_single > getattr(self, 'best_single_score', 0)
    
        # Criterion 4: Consistency check (low variance in recent episodes)
        recent_scores = self.metrics.episode_scores[-20:]
        if len(recent_scores) >= 20:
            variance = np.var(recent_scores)
            is_consistent = variance < np.var(self.metrics.episode_scores) * 0.8
        else:
            is_consistent = True
    
        # Save as best if average improved AND (single score improved OR performance is consistent)
        should_save = avg_improved and (single_score_improved or is_consistent)
    
        if should_save and single_score_improved:
            self.best_single_score = current_best_single
    
        return should_save

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files, keeping only the most recent ones."""
        try:
            # Find all model files (excluding best_model.pth)
            model_pattern = os.path.join(self.config.checkpoint_dir, 'model_ep*.pth')
            model_files = glob.glob(model_pattern)
            
            # Find all state files
            state_pattern = os.path.join(self.config.checkpoint_dir, 'training_state_ep*.json')
            state_files = glob.glob(state_pattern)
            
            # Sort by episode number and remove oldest files
            if len(model_files) > keep_last:
                # Extract episode numbers and sort
                model_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
                files_to_remove = model_files[:-keep_last]
                
                for old_file in files_to_remove:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old model checkpoint: {os.path.basename(old_file)}")
            
            if len(state_files) > keep_last:
                # Extract episode numbers and sort
                state_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
                files_to_remove = state_files[:-keep_last]
                
                for old_file in files_to_remove:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old state file: {os.path.basename(old_file)}")
                    
        except Exception as e:
            self.logger.warning(f"Error during checkpoint cleanup: {e}")
            # Don't raise exception - cleanup failure shouldn't stop training

    
    def log_progress(self, episode_result: Dict[str, Any]):
        """Log training progress."""
        avg_reward = self.metrics.get_recent_avg_reward()
        avg_score = self.metrics.get_recent_avg_score()
        best_score = self.metrics.get_best_score()
        
        # Console logging
        self.logger.info(
            f"Episode {self.episode:4d} | "
            f"Score: {episode_result['score']:4d} | "
            f"Reward: {episode_result['reward']:7.2f} | "
            f"Steps: {episode_result['steps']:4d} | "
            f"Epsilon: {episode_result['epsilon']:.3f} | "
            f"Avg Score: {avg_score:.1f} | "
            f"Best: {best_score}"
        )
        
        # TensorBoard logging
        if self.tb_logger.writer:
            self.tb_logger.log_scalar('Episode/Reward', episode_result['reward'], self.episode)
            self.tb_logger.log_scalar('Episode/Score', episode_result['score'], self.episode)
            self.tb_logger.log_scalar('Episode/Steps', episode_result['steps'], self.episode)
            self.tb_logger.log_scalar('Episode/Epsilon', episode_result['epsilon'], self.episode)
            self.tb_logger.log_scalar('Average/Reward', avg_reward, self.episode)
            self.tb_logger.log_scalar('Average/Score', avg_score, self.episode)
            
            if self.metrics.losses:
                recent_loss = np.mean(self.metrics.losses[-100:])
                self.tb_logger.log_scalar('Training/Loss', recent_loss, self.episode)
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.total_episodes} episodes")
        self.logger.info(f"Device: {self.config.device}")
        
        start_time = time.time()
        
        try:
            for episode in range(1, self.config.total_episodes + 1):
                self.episode = episode
                
                # Run training episode
                episode_result = self.run_episode(training=True)
                
                # Update metrics
                self.metrics.add_episode(
                    episode_result['reward'],
                    episode_result['score'],
                    episode_result['steps'],
                    episode_result['epsilon']
                )

                # Log progress
                if episode % self.config.log_freq == 0:
                    self.log_progress(episode_result)
                
                # Evaluate agent
                if episode % self.config.eval_freq == 0:
                    eval_results = self.evaluate_agent()
                    self.logger.info(
                        f"Evaluation - Avg Score: {eval_results['avg_score']:.1f} ± {eval_results['std_score']:.1f} | "
                        f"Max Score: {eval_results['max_score']} | "
                        f"Avg Reward: {eval_results['avg_reward']:.2f}"
                    )
                    
                    # Log evaluation results
                    if self.tb_logger.writer:
                        self.tb_logger.log_scalar('Evaluation/Avg_Score', eval_results['avg_score'], episode)
                        self.tb_logger.log_scalar('Evaluation/Max_Score', eval_results['max_score'], episode)
                        self.tb_logger.log_scalar('Evaluation/Avg_Reward', eval_results['avg_reward'], episode)
                
                # Save checkpoint
                should_save_best = self._evaluate_best_model_criteria()
                
                if episode % self.config.save_freq == 0 or should_save_best:
                    self.save_checkpoint()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
        finally:
            # Final save and cleanup
            self.save_checkpoint()
            
            # Save final metrics
            final_metrics_path = os.path.join(self.config.results_dir, 'final_metrics.json')
            with open(final_metrics_path, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            
            # Close resources
            self.tb_logger.close()
            self.env.close()
            
            # Training summary
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time/3600:.2f} hours")
            self.logger.info(f"Best average score: {self.best_avg_score:.2f}")
            self.logger.info(f"Best single score: {self.metrics.get_best_score()}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Chrome Dino game")
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=None, help='Replay buffer size')
    parser.add_argument('--device', type=str, default=None, choices=[None, 'auto', 'cuda', 'cpu'], 
                       help='Device to use for training')
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default=None, help='Log directory')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config with defaults
    config = TrainingConfig()
    
    # Update config ONLY if args were provided
    if args.episodes is not None:
        config.total_episodes = args.episodes
    
    # headless is special - store_true means it's False by default
    if args.headless:
        config.headless = True
    
    if args.lr is not None:
        config.lr = args.lr
    
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    
    # Device handling
    if args.device is not None:
        if args.device == 'auto':
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            config.device = args.device
    
    # Create trainer and start training
    trainer = DinoTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.logger.info(f"Resuming training from {args.resume}")
        trainer.agent.load_model(args.resume)
    
    trainer.train()

if __name__ == '__main__':
    main()
