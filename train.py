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
        self.env = None
        self._initialize_environment()
        
        # Initialize agent with proper configuration
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
            decay_steps=config.decay_steps,
            use_visual=getattr(config, 'use_visual', False),
            use_numerical=getattr(config, 'use_numerical', True)
        )
        
        # Initialize logging and metrics
        self.tb_logger = TensorBoardLogger(config.log_dir)
        self.metrics = TrainingMetrics()
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_avg_score = 0.0
        self.best_single_score = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        self.logger.info("Trainer initialized successfully!")
    
    def _initialize_environment(self, max_retries: int = 3):
        """Initialize environment with retry logic."""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Initializing environment (attempt {attempt + 1}/{max_retries})...")
                if self.env:
                    self.env.close()
                
                self.env = DinoGameEnvironment(
                    frame_stack=self.config.frame_stack,
                    processed_size=self.config.processed_size,
                    max_episode_steps=self.config.max_episode_steps,
                    reward_scale=self.config.reward_scale,
                    use_visual=getattr(self.config, 'use_visual', False),
                    use_numerical=getattr(self.config, 'use_numerical', True),
                    normalize_numerical=getattr(self.config, 'normalize_numerical', True)
                )
                self.logger.info("Environment initialized successfully!")
                return
                
            except Exception as e:
                self.logger.error(f"Failed to initialize environment (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize environment after {max_retries} attempts")
                time.sleep(2)  # Wait before retry
    
    def run_episode(self, training: bool = True) -> Dict[str, Any]:
        """Run a single episode."""
        max_episode_retries = 3
        
        for retry in range(max_episode_retries):
            try:
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
                            if loss is not None:
                                self.metrics.add_loss(loss)
                    
                    state = next_state
                    
                    if done:
                        break
                
                # Episode completed successfully
                self.consecutive_failures = 0
                return {
                    'reward': episode_reward,
                    'score': episode_score,
                    'steps': episode_steps,
                    'epsilon': self.agent.get_current_epsilon()
                }
                
            except Exception as e:
                self.logger.warning(f"Episode failed (retry {retry + 1}/{max_episode_retries}): {e}")
                self.consecutive_failures += 1
                
                if retry == max_episode_retries - 1:
                    # Last retry failed
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        self.logger.error("Too many consecutive failures. Reinitializing environment...")
                        self._initialize_environment()
                        self.consecutive_failures = 0
                    
                    # Return default episode result
                    return {
                        'reward': -10.0,
                        'score': 0,
                        'steps': 1,
                        'epsilon': self.agent.get_current_epsilon()
                    }
                
                # Wait before retry
                time.sleep(1)
    
    def evaluate_agent(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        self.logger.info(f"Evaluating agent over {num_episodes} episodes...")
        
        eval_rewards = []
        eval_scores = []
        successful_episodes = 0
        
        for eval_ep in range(num_episodes):
            try:
                result = self.run_episode(training=False)
                eval_rewards.append(result['reward'])
                eval_scores.append(result['score'])
                successful_episodes += 1
            except Exception as e:
                self.logger.warning(f"Evaluation episode {eval_ep + 1} failed: {e}")
                # Add default values for failed episode
                eval_rewards.append(0.0)
                eval_scores.append(0)
        
        if successful_episodes == 0:
            self.logger.warning("All evaluation episodes failed!")
            return {
                'avg_reward': 0.0,
                'avg_score': 0.0,
                'max_score': 0,
                'std_reward': 0.0,
                'std_score': 0.0
            }
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'avg_score': np.mean(eval_scores),
            'max_score': np.max(eval_scores),
            'std_reward': np.std(eval_rewards),
            'std_score': np.std(eval_scores),
            'success_rate': successful_episodes / num_episodes
        }
    
    def save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint with complete state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Always save regular checkpoint
        if is_final:
            model_path = os.path.join(self.config.checkpoint_dir, 'final_model.pth')
        else:
            model_path = os.path.join(self.config.checkpoint_dir, f'model_ep{self.episode}.pth')
        
        try:
            self.agent.save_model(model_path)
            
            # Determine if this should be saved as "best"
            should_save_as_best = self._evaluate_best_model_criteria()
            
            if should_save_as_best:
                best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                self.agent.save_model(best_path)
                self.best_avg_score = self.metrics.get_recent_avg_score()
                self.logger.info(f"New best model saved! Avg score: {self.best_avg_score:.2f}")
            
            # Save complete training state
            state = {
                'episode': self.episode,
                'total_steps': self.total_steps,
                'best_avg_score': self.best_avg_score,
                'best_single_score': getattr(self, 'best_single_score', 0),
                'timestamp': timestamp,
                'config': self.config.__dict__,
                'metrics': self.metrics.to_dict(),
                'agent_config': {
                    'use_visual': self.agent.use_visual,
                    'use_numerical': self.agent.use_numerical
                }
            }
            
            if is_final:
                state_path = os.path.join(self.config.checkpoint_dir, 'final_training_state.json')
            else:
                state_path = os.path.join(self.config.checkpoint_dir, f'training_state_ep{self.episode}.json')
            
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)  # default=str handles numpy types
            
            # Cleanup old checkpoints (but not for final save)
            if not is_final:
                self._cleanup_old_checkpoints(keep_last=3)
            
            self.logger.info(f"Checkpoint saved: episode {self.episode}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load checkpoint and restore training state."""
        try:
            # Load model
            self.agent.load_model(checkpoint_path)
            
           # Try to load corresponding training state
            if checkpoint_path.endswith('final_model.pth'):
                # For final model, look for final_training_state.json
                state_path = os.path.join(self.config.checkpoint_dir, 'final_training_state.json')
            elif checkpoint_path.endswith('best_model.pth'):
                # Look for the most recent training state
                pattern = os.path.join(self.config.checkpoint_dir, 'training_state_ep*.json')
                state_files = glob.glob(pattern)
                if state_files:
                    state_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
                    state_path = state_files[-1]
                else:
                    state_path = None
            else:
                # Regular checkpoint
                state_path = checkpoint_path.replace('model_', 'training_state_').replace('.pth', '.json')
            
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.episode = state.get('episode', 0)
                self.total_steps = state.get('total_steps', 0)
                self.best_avg_score = state.get('best_avg_score', 0.0)
                self.best_single_score = state.get('best_single_score', 0)
                
                # Restore metrics if available
                if 'metrics' in state:
                    metrics_data = state['metrics']
                    self.metrics.episode_rewards = metrics_data.get('episode_rewards', [])
                    self.metrics.episode_scores = metrics_data.get('episode_scores', [])
                    self.metrics.episode_lengths = metrics_data.get('episode_lengths', [])
                    self.metrics.losses = metrics_data.get('losses', [])
                    self.metrics.epsilon_values = metrics_data.get('epsilon_values', [])
                
                self.logger.info(f"Restored training state from episode {self.episode}")
            else:
                self.logger.warning(f"Training state file not found: {state_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
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
        single_score_improved = current_best_single > self.best_single_score
    
        # Criterion 4: Consistency check (low variance in recent episodes)
        recent_scores = self.metrics.episode_scores[-20:] if len(self.metrics.episode_scores) >= 20 else []
        if recent_scores:
            variance = np.var(recent_scores)
            all_scores_variance = np.var(self.metrics.episode_scores) if self.metrics.episode_scores else float('inf')
            is_consistent = variance < all_scores_variance * 0.8
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
            # Find all model files (excluding best_model.pth and final_model.pth)
            model_pattern = os.path.join(self.config.checkpoint_dir, 'model_ep*.pth')
            model_files = glob.glob(model_pattern)
            
            # Find all state files
            state_pattern = os.path.join(self.config.checkpoint_dir, 'training_state_ep*.json')
            state_files = glob.glob(state_pattern)
            
            # Sort by episode number and remove oldest files
            if len(model_files) > keep_last:
                model_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
                files_to_remove = model_files[:-keep_last]
                
                for old_file in files_to_remove:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old model checkpoint: {os.path.basename(old_file)}")
            
            if len(state_files) > keep_last:
                state_files.sort(key=lambda x: int(x.split('_ep')[1].split('.')[0]))
                files_to_remove = state_files[:-keep_last]
                
                for old_file in files_to_remove:
                    os.remove(old_file)
                    self.logger.debug(f"Removed old state file: {os.path.basename(old_file)}")
                    
        except Exception as e:
            self.logger.warning(f"Error during checkpoint cleanup: {e}")
    
    def log_progress(self, episode_result: Dict[str, Any]):
        """Log training progress with error handling."""
        try:
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
            if self.tb_logger and self.tb_logger.writer:
                self.tb_logger.log_scalar('Episode/Reward', episode_result['reward'], self.episode)
                self.tb_logger.log_scalar('Episode/Score', episode_result['score'], self.episode)
                self.tb_logger.log_scalar('Episode/Steps', episode_result['steps'], self.episode)
                self.tb_logger.log_scalar('Episode/Epsilon', episode_result['epsilon'], self.episode)
                self.tb_logger.log_scalar('Average/Reward', avg_reward, self.episode)
                self.tb_logger.log_scalar('Average/Score', avg_score, self.episode)
                
                if self.metrics.losses:
                    recent_loss = np.mean(self.metrics.losses[-100:])
                    self.tb_logger.log_scalar('Training/Loss', recent_loss, self.episode)
        except Exception as e:
            self.logger.warning(f"Error logging progress: {e}")
    
    def train(self):
        """Main training loop with comprehensive error handling."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.total_episodes} episodes")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Using visual input: {self.agent.use_visual}")
        self.logger.info(f"Using numerical input: {self.agent.use_numerical}")
        
        start_time = time.time()
        
        try:
            start_episode = self.episode + 1 if self.episode > 0 else 1
            
            for episode in range(start_episode, self.config.total_episodes + 1):
                self.episode = episode
                
                try:
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
                            f"Success Rate: {eval_results.get('success_rate', 1.0):.2f}"
                        )
                        
                        # Log evaluation results
                        if self.tb_logger and self.tb_logger.writer:
                            self.tb_logger.log_scalar('Evaluation/Avg_Score', eval_results['avg_score'], episode)
                            self.tb_logger.log_scalar('Evaluation/Max_Score', eval_results['max_score'], episode)
                            self.tb_logger.log_scalar('Evaluation/Success_Rate', eval_results.get('success_rate', 1.0), episode)
                    
                    # Save checkpoint
                    should_save_best = self._evaluate_best_model_criteria()
                    
                    if episode % self.config.save_freq == 0 or should_save_best:
                        self.save_checkpoint()
                
                except Exception as e:
                    self.logger.error(f"Error in episode {episode}: {e}")
                    # Continue with next episode
                    continue
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
        finally:
            # Final save and cleanup
            try:
                self.save_checkpoint(is_final=True)
                
                # Save final metrics
                final_metrics_path = os.path.join(self.config.results_dir, 'final_metrics.json')
                with open(final_metrics_path, 'w') as f:
                    json.dump(self.metrics.to_dict(), f, indent=2, default=str)
                
                # Training summary
                training_time = time.time() - start_time
                self.logger.info(f"Training completed in {training_time/3600:.2f} hours")
                self.logger.info(f"Best average score: {self.best_avg_score:.2f}")
                self.logger.info(f"Best single score: {self.metrics.get_best_score()}")
                
            except Exception as e:
                self.logger.error(f"Error during final cleanup: {e}")
            finally:
                # Close resources
                if self.tb_logger:
                    self.tb_logger.close()
                if self.env:
                    self.env.close()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Chrome Dino game")
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=None, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=None, help='Replay buffer size')
    parser.add_argument('--device', type=str, default=None, choices=[None, 'auto', 'cuda', 'cpu'], 
                       help='Device to use for training')
    
    # Input configuration
    parser.add_argument('--use-visual', action='store_true', help='Enable visual input processing')
    parser.add_argument('--use-numerical', action='store_true', default=None, help='Enable numerical input processing')
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default=None, help='Log directory')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config with defaults
    config = TrainingConfig()
    
    # Update config based on arguments
    if args.episodes is not None:
        config.total_episodes = args.episodes
    
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
    
    # Input configuration
    if args.use_visual:
        config.use_visual = True
    
    if args.use_numerical is not None:
        config.use_numerical = args.use_numerical
    elif not args.use_visual:
        # If visual is not enabled and numerical is not specified, default to True
        config.use_numerical = True
    
    # Device handling
    if args.device is not None:
        if args.device == 'auto':
            config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            config.device = args.device
    
    # Validate configuration
    if not getattr(config, 'use_visual', False) and not getattr(config, 'use_numerical', True):
        raise ValueError("At least one of --use-visual or --use-numerical must be enabled")
    
    # Create trainer
    trainer = DinoTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.logger.info(f"Resuming training from {args.resume}")
        if not trainer.load_checkpoint(args.resume):
            trainer.logger.error("Failed to load checkpoint. Starting fresh training.")
    
    trainer.train()

if __name__ == '__main__':
    main()
