# AI-Dino: Reinforcement Learning for Chrome's Dinosaur Game

A machine learning project that trains an AI agent to play the Chrome Dinosaur Game using reinforcement learning with a hybrid input system (visual and numerical) and browser automation.

## Project Overview

AI-Dino uses a Deep Q-Network (DQN) to master the Chrome dinosaur game. It uniquely combines **visual data** from game frames with **numerical data** extracted directly from the game's state (e.g., speed, distance to obstacles).

The project is built with PyTorch, Playwright for browser automation, and follows the OpenAI Gym (Gymnasium) interface for a standardized environment.

## Key Features

- **Hybrid Input System**: The agent can be trained using visual features, numerical features, or both, allowing for flexible experimentation.
- **Deep Q-Network (DQN)**: Implemented a Deep Q-Network to learn the optimal policy for the game.
  - Double DQN to mitigate Q-value overestimation.
  - Prioritized Experience Replay for efficient learning from important transitions.
- **Robust Browser Automation**: Built on Playwright, it features a threaded, asynchronous game loop with robust error handling and session recovery.
- **Comprehensive Logging & Visualization**: Detailed console output, TensorBoard integration for real-time monitoring, and automatic saving of training metrics.
- **Advanced Training Management**: Sophisticated checkpointing system that saves the best models based on multiple criteria and allows for seamless resumption of training.

## Project Structure

- `train.py`: The main entry point for training the agent. Contains the `DinoTrainer` class.
- `test.py`: The entry point for evaluating a trained agent. Contains the `DinoTester` class.
- `agent.py`: Implements the `DQNAgent` and the `DQN` neural network architecture.
- `environment.py`: The `DinoGameEnvironment` class, which handles all browser interaction and conforms to the OpenAI Gym interface.
- `utils.py`: Contains utility classes:
  - `TrainingConfig`: Centralized configuration for all training parameters.
  - `TrainingMetrics`: Tracks and saves detailed training progress.
  - `PrioritizedReplayBuffer`: Implements the prioritized experience replay mechanism.
  - `EpsilonGreedy`: Manages the exploration-exploitation strategy.
  - `FrameProcessor`: Handles preprocessing of visual game frames.
  - `TensorBoardLogger`: A simple wrapper for TensorBoard logging.
- `requirements.txt`: A list of all Python dependencies.

## How to Use

### Installation

1.  **Clone the repository**:
    ```bash
    (HTTPS) git clone https://github.com/404brtk/ai-dino.git
    (SSH) git clone git@github.com:404brtk/ai-dino.git
    cd ai-dino
    ```
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install Playwright browsers**:
    ```bash
    playwright install chromium
    ```

### Training the Agent

The training script is highly configurable. By default, it runs using only numerical inputs.

**Default Training (Numerical Inputs Only):**
```bash
python train.py
```

**Training with Visual Inputs:**
To enable visual processing, use the `--use-visual` flag. You can also disable numerical inputs if desired.
```bash
# Use both visual and numerical inputs
python train.py --episodes 5000 --use-visual

# Use only visual inputs
python train.py --episodes 5000 --use-visual --use-numerical false
```

**Resuming Training:**
Resume from the last saved checkpoint. You can also reset the exploration (epsilon) schedule to encourage new learning.
```bash
# Resume from a specific checkpoint
python train.py --resume ./checkpoints/model_ep1000.pth

# Resume and reset epsilon for renewed exploration
python train.py --resume ./checkpoints/model_ep1000.pth --reset-epsilon
```

**Key Training Arguments:**

- `--episodes`: Number of training episodes.
- `--device`: Device to use (`auto`, `cuda`, `cpu`).
- `--lr`: Learning rate (e.g., `0.0001`).
- `--batch-size`: Batch size for training.
- `--buffer-size`: Replay buffer capacity.
- `--use-visual`: Enable visual feature processing.
- `--use-numerical`: Enable/disable numerical feature processing (enabled by default).
- `--resume`: Path to a checkpoint file to resume training.
- `--reset-epsilon`: Reset the exploration rate when resuming.

### Testing a Trained Model

The testing script automatically detects the model's configuration (visual/numerical) from the checkpoint file.

**Run a Test:**
```bash
python test.py --model ./checkpoints/model_ep1000.pth
```

**Key Testing Arguments:**

- `--model`: (Required) Path to the trained model file (`.pth`).
- `--episodes`: Number of test episodes to run.
- `--delay`: Optional delay in seconds between actions.
- `--no-render`: Run silently without printing step-by-step progress.

### Monitoring with TensorBoard

Visualize training metrics in real-time:
```bash
# In a separate terminal
tensorboard --logdir=logs
```
Navigate to `http://localhost:6006` in your browser.

## Technical Details

### Hybrid Neural Network Architecture

The DQN model dynamically adjusts its architecture based on the input configuration:
- **Visual Path**: A series of convolutional layers processes the stacked game frames to extract spatial features.
- **Numerical Path**: A multi-layer perceptron (MLP) processes the vector of numerical game states.
- **Combined Path**: The outputs from the visual and/or numerical paths are concatenated and passed through a final set of fully connected layers to produce the Q-values for each action.

This flexible design allows the agent to leverage different sources of information effectively.