# AI-Dino: Reinforcement Learning for Chrome's Dinosaur Game

A machine learning project that trains an AI agent to play the Chrome Dinosaur Game using reinforcement learning techniques and browser automation.

## Project Overview

AI-Dino uses a Deep Q-Network (DQN) approach to teach an AI agent how to play the offline dinosaur game that appears in Chrome when you're disconnected from the internet. The project leverages Playwright for browser automation and PyTorch for the neural network implementation.

## Features

- **Browser Automation**: Uses Playwright to control the Chrome browser and interact with the dinosaur game
- **Deep Reinforcement Learning**: Implements DQN with several modern improvements:
  - Double DQN for more stable learning
  - Prioritized Experience Replay for efficient training
  - Frame stacking for temporal awareness
- **Robust Error Handling**: Includes recovery mechanisms for browser crashes and game resets
- **Visualization Tools**: Training progress visualization using TensorBoard
- **Configurable Training**: Easily adjustable training parameters

## Requirements

The project requires Python 3.6+ and the following major packages:

- PyTorch: Neural network implementation
- Playwright: Browser automation
- OpenCV: Image processing
- Gym: Reinforcement learning environment structure
- TensorBoard: Training visualization
- NumPy: Numerical operations

For the complete list of requirements, see `requirements.txt`.

## Project Structure

- `environment.py`: Contains the `DinoGameEnvironment` class that handles browser interaction and implements the OpenAI Gym interface
- `agent.py`: Implements the `DQNAgent` and neural network architecture
- `utils.py`: Contains utility classes such as:
  - `FrameProcessor`: Handles image preprocessing
  - `PrioritizedReplayBuffer`: Manages experience replay with prioritization
  - `EpsilonGreedy`: Handles exploration/exploitation balance
  - `TrainingMetrics`: Tracks training progress
- `train.py`: Manages the training loop and logging
- `test.py`: Evaluates trained models without additional training

## How to Use

### Installation

1. Clone the repository `git clone https://github.com/404brtk/ai-dino.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Install Playwright browsers: `playwright install chromium`

### Training the Agent

To start training with default parameters:

```bash
python train.py
```

With custom parameters, for example:
```bash
python train.py --episodes 5000 --headless --device cuda
```

To resume training from a previous checkpoint:
```bash
python train.py --resume ./checkpoints/dqn_checkpoint_episode_1000.pth
```

Available options:
- `--episodes`: Number of training episodes
- `--headless`: Run in headless mode (no visible browser)
- `--lr`: Learning rate
- `--batch-size`: Batch size for training
- `--buffer-size`: Replay buffer size
- `--device`: Device to use for training ('auto', 'cuda', or 'cpu')
- `--checkpoint-dir`: Directory to save checkpoints
- `--log-dir`: Directory for TensorBoard logs
- `--results-dir`: Directory for results
- `--resume`: Path to checkpoint to resume training from

### Testing Trained Models

To evaluate a trained model without further training:

```bash
python test.py --model ./checkpoints/best_model.pth
```

With custom parameters:
```bash
python test.py --model ./checkpoints/best_model.pth --episodes 3 --delay 0.05
```

Available options:
- `--model`: Path to the trained model file (.pth) [required]
- `--episodes`: Number of test episodes to run (default: 5)
- `--headless`: Run in headless mode (no browser UI)
- `--device`: Device to use ('auto', 'cuda', or 'cpu')
- `--delay`: Optional delay between steps in seconds (to slow down visualization)

### Testing the Environment

To verify that the environment works correctly:

```bash
python environment.py --visualize
```

This will show the processed frames that the agent sees.

## How It Works

1. **Game Environment**: The `DinoGameEnvironment` class creates a Playwright browser session that loads the Chrome dinosaur game.

2. **Observation Processing**: Game screenshots are captured, converted to grayscale, resized, and normalized to create the agent's observations.

3. **DQN Agent**: The agent receives these processed frames and uses its neural network to decide which action to take (do nothing, long jump, short jump, or duck).

4. **Training Loop**: The `DinoTrainer` class manages the training process, collecting experiences from the environment and training the agent.

5. **Reward System**: The agent receives rewards based on survival time and game score, with penalties for crashing.

## Technical Details

### Neural Network Architecture

The DQN uses a convolutional neural network with:
- Three convolutional layers for feature extraction
- Adaptive average pooling for consistent output size
- Three fully connected layers with dropout for Q-value estimation

### Reinforcement Learning Implementation

- **Double DQN**: Uses separate target and policy networks to reduce overestimation bias
- **Prioritized Experience Replay**: Prioritizes important experiences for more efficient learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation with a decaying epsilon value

### Browser Automation

- Asynchronous browser control with Playwright
- Thread-safe communication between the environment and agent
- Robust error handling and session recovery