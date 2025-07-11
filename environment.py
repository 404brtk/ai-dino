import gym
from gym import spaces
import numpy as np
import asyncio
import threading
import io
import logging
import argparse
import time
from queue import Queue, Empty
from typing import Tuple, Optional, Dict, Any
from playwright.async_api import async_playwright, Playwright, Page, Browser, Error as PlaywrightError
from PIL import Image
import matplotlib.pyplot as plt

from utils import FrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DinoGameEnvironment(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 headless: bool = False,
                 frame_stack: int = 4,
                 frame_skip: int = 2,
                 processed_size: Tuple[int, int] = (84, 84),
                 max_episode_steps: int = 10000,
                 reward_scale: float = 1.0):
        super().__init__()

        # Core parameters
        self.headless = headless
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.processed_size = processed_size
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale

        self.logger = logging.getLogger(__name__)

        # Initialize the frame processor utility
        self.frame_processor = FrameProcessor(target_size=self.processed_size, stack_frames=self.frame_stack)

        # Gym-specific definitions
        self.action_space = spaces.Discrete(4)  # 0: Do Nothing, 1: Long Jump, 2: Short Jump, 3: Duck

        processed_height, processed_width = self.processed_size

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.frame_stack, processed_height, processed_width),
            dtype=np.float32
        )

        # State and performance tracking
        self.current_step = 0
        self.last_score = 0
        self.episode_reward = 0.0

        # Playwright and browser state
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Threading and async management
        self.action_queue = Queue()
        self.state_queue = Queue()
        self.stop_event = threading.Event()
        self.game_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        
        # Start the game thread and wait for it to be ready
        self.game_thread.start()
        self._wait_for_ready()


    async def _init_browser(self):
        """Initializes Playwright, launches the browser, and navigates to the game."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

        await self.context.set_offline(True)
        try:
            await self.page.goto("https://www.google.com", timeout=5000)
        except PlaywrightError:
            # This is expected as we are offline. It loads the dino game.
            self.logger.info("Successfully loaded dino game page.")

        await self.page.set_viewport_size({"width": 800, "height": 600})

        # NOTE: The code below works correctly, but causes the window to flash.
        """
        # Get the bounding box of the game canvas to crop screenshots
        canvas_handle = await self.page.query_selector('.runner-canvas')
        if canvas_handle:
            box = await canvas_handle.bounding_box()
            if box:
                self.game_clip_box = {
                    'x': int(box['x']),
                    'y': int(box['y']),
                    'width': int(box['width']),
                    'height': int(box['height'])
                }
                self.logger.info(f"Game canvas found at: {self.game_clip_box}")
            if not self.game_clip_box:
                self.logger.warning("Could not find game canvas, screenshots will not be cropped.")
        """

    def _run_async_loop(self):
        """Sets up and runs the asyncio event loop in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_game_loop())
        except Exception as e:
            self.logger.error(f"Async loop crashed: {e}", exc_info=True)
            self.state_queue.put(e)
        finally:
            loop.close()

    async def _async_game_loop(self):
        """The main async loop that handles all browser interactions."""
        try:
            if self.page is None:
                await self._init_browser()
                self.state_queue.put("ready")

            while not self.stop_event.is_set():
                try:
                    command = self.action_queue.get_nowait()
                    if command == 'reset':
                        state = await self._async_reset()
                        self.state_queue.put(state)
                    elif isinstance(command, int):
                        result = await self._async_step(command)
                        self.state_queue.put(result)
                    elif command == 'close':
                        break
                except Empty:
                    # Use a small sleep to prevent CPU spinning
                    await asyncio.sleep(0.001)
                except PlaywrightError as pe:
                    self.logger.error(f"Playwright error: {pe}")
                    # Try to recover the browser session
                    await self._recover_browser_session()
                    self.state_queue.put(Exception(f"Browser error occurred: {pe}"))
                except Exception as e:
                    self.logger.error(f"Error in async game loop: {e}")
                    self.state_queue.put(e)
        except Exception as e:
            self.logger.error(f"Fatal error in async game loop: {e}")
            self.state_queue.put(e)
        finally:
            await self._cleanup_async()

    async def _capture_and_process_frame(self) -> np.ndarray:
        """Captures a raw frame and processes it using the FrameProcessor.
        
        Returns:
            np.ndarray: Processed frame stack with shape (frame_stack, height, width)
                    or zeros with same shape if an error occurs
        """
        try:
            # 1. Get canvas bounding box
            canvas_handle = await self.page.query_selector('.runner-canvas')
            box = await canvas_handle.bounding_box() if canvas_handle else None

            # 2. Take screenshot
            screenshot_bytes = await self.page.screenshot(type='png')
            img = Image.open(io.BytesIO(screenshot_bytes))
            raw_frame = np.array(img)

            # 3. Crop the frame using the bounding box
            if box:
                x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
                raw_frame = raw_frame[y:y+h, x:x+w, :] # Crop using numpy slicing
            else:
                self.logger.warning("Game canvas not found, using uncropped frame.")

            # 4. Process the frame (handles grayscale, resize, normalize, stacking)
            return self.frame_processor.process_frame(raw_frame)
            
        except Exception as e:
            self.logger.error(f"Error in _capture_and_process_frame: {str(e)}", 
                            exc_info=self.logger.level <= logging.DEBUG)
            # Return a blank frame stack with correct shape
            return np.zeros((self.frame_stack, *self.processed_size), dtype=np.float32)


    async def _is_game_over(self) -> bool:
        """Checks if the 'Game Over' screen is visible."""
        return await self.page.evaluate('() => window.Runner.instance_.crashed')
    
    async def _get_current_score(self) -> int:
        """Retrieves the current score."""
        digits = await self.page.evaluate('() => window.Runner.instance_.distanceMeter.digits')
        if not digits:
            return 0
        return int(''.join(digits))
    
    async def _get_current_speed(self) -> float:
        """Retrieves the current speed."""
        return await self.page.evaluate('() => window.Runner.instance_.currentSpeed')

    async def _get_game_state(self) -> Dict[str, Any]:
        """Gets the current game state by calling helper methods."""
        try:
            # Use existing helper methods to fetch game state components
            crashed, score, speed = await asyncio.gather(
                self._is_game_over(),
                self._get_current_score(),
                self._get_current_speed()
            )
            return {'crashed': crashed, 'score': score, 'speed': speed}
        except PlaywrightError as e:
            self.logger.error(f"Error getting game state: {e}")
            # Return a default state indicating a crash
            return {'crashed': True, 'score': 0, 'speed': 0}
    
    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculates the reward based on the final game state of a step."""
        if game_state.get('crashed', False):
            return -10.0  # Large penalty for crashing

        # Base survival reward
        reward = 0.1 * self.reward_scale
        
        # Score-based reward - progressive difficulty handling
        current_score = game_state.get('score', self.last_score)
        score_diff = current_score - self.last_score

        if score_diff > 0:
            # Higher reward for higher speeds (difficulty)
            speed = game_state.get('speed', 0)
            # Scale reward by speed, capping the factor to avoid extreme rewards
            speed_factor = min(2.0, speed / 10.0)
            reward += score_diff * 0.1 * speed_factor * self.reward_scale
        
        self.last_score = current_score
        
        return reward

    async def _async_reset(self) -> np.ndarray:
        """Resets the game state asynchronously, with robust recovery logic."""
        try:
            if self.page is None or self.page.is_closed():
                self.logger.warning("Page not found during reset, attempting to recover.")
                await self._recover_browser_session()

            # Robust game restart logic
            max_restart_attempts = 5
            for attempt in range(max_restart_attempts):
                try:
                    # Check if game is crashed and needs restart
                    if await self._is_game_over():
                        self.logger.info(f"Game crashed, attempting restart (attempt {attempt + 1}/{max_restart_attempts})")
                        
                        # Press space to restart
                        await self.page.keyboard.press('Space')
                        await asyncio.sleep(0.3)  # Longer wait for game to restart
                        
                        # Verify the game has actually restarted
                        is_still_crashed = await self._is_game_over()
                        if not is_still_crashed:
                            self.logger.info("Game successfully restarted")
                            break
                        else:
                            self.logger.warning(f"Game still crashed after restart attempt {attempt + 1}")
                            if attempt < max_restart_attempts - 1:
                                await asyncio.sleep(0.5)  # Wait before next attempt
                    else:
                        self.logger.info("Game is running")
                        break
                        
                except PlaywrightError as e:
                    self.logger.error(f"Playwright error during restart attempt {attempt + 1}: {e}")
                    if attempt == max_restart_attempts - 1:
                        # Last attempt failed, try browser recovery
                        recovered = await self._recover_browser_session()
                        if recovered:
                            await self.page.keyboard.press('Space')
                            await asyncio.sleep(0.3)
                        else:
                            self.logger.error("All restart attempts failed. Returning zero state.")
                            return np.zeros(self.observation_space.shape, dtype=np.float32)
                    else:
                        await asyncio.sleep(0.5)  # Wait before next attempt

            # Wait for game to be fully ready
            await asyncio.sleep(0.2)
            
            # Verify final state
            final_crashed = await self._is_game_over()
            if final_crashed:
                self.logger.error("Game is still crashed after all restart attempts")
                return np.zeros(self.observation_space.shape, dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Unexpected error during reset: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Reset internal state
        self.current_step = 0
        self.last_score = 0
        self.episode_reward = 0.0

        # Reset the frame processor's internal buffer
        self.frame_processor.reset()

        # Get the initial stacked observation
        return await self._capture_and_process_frame()


    async def _async_step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one game step efficiently and correctly."""
        try:
            # 1. Perform action
            if action == 1:  # Long Jump
                await self.page.keyboard.press('ArrowUp', delay=200)
            elif action == 2:  # Short Jump
                await self.page.keyboard.press('ArrowUp', delay=50)
            elif action == 3:  # Duck
                await self.page.keyboard.down('ArrowDown')

            # 2. Wait for action to take effect (frame skip duration)
            await asyncio.sleep(0.01 * self.frame_skip)

            # 3. Release duck key if it was held down
            if action == 3:
                await self.page.keyboard.up('ArrowDown')

            # 4. Get the definitive state of the game after the action
            game_state = await self._get_game_state()
            terminated = game_state.get('crashed', False)

            # 5. Calculate reward based on the final state
            reward = self._calculate_reward(game_state)

            # 6. Capture the visual observation
            observation = await self._capture_and_process_frame()
            if observation is None:
                observation = np.zeros(self.observation_space.shape, dtype=np.float32)
                self.logger.warning("Failed to capture frame, returning blank observation.")

            # 7. Update internal state and prepare return values
            self.episode_reward += reward
            self.current_step += 1
            truncated = self.current_step >= self.max_episode_steps

            info = {'score': game_state.get('score', 0), 'speed': game_state.get('speed', 0)}

            if terminated:
                self.logger.info(f"Episode terminated at step {self.current_step}. Score: {info['score']}")

            return observation, reward, terminated, truncated, info

        except PlaywrightError as e:
            self.logger.error(f"A Playwright error occurred during step: {e}")
            return (np.zeros(self.observation_space.shape, dtype=np.float32),
                    -10.0, True, False, {'score': self.last_score, 'speed': 0, 'error': str(e)})


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # Ensure the game thread is alive
        if not self.game_thread.is_alive():
            self.logger.warning("Game thread not alive during reset, restarting...")
            self._restart_game_thread()
            
        # Send reset command
        self.action_queue.put('reset')
        
        try:
            state = self.state_queue.get(timeout=10)
            if isinstance(state, Exception):
                self.logger.error(f"Error during reset: {state}")
                raise state
            return state, {}
        except Empty:
            self.logger.error("Reset timeout: no response from game thread")
            raise TimeoutError("The game thread did not respond to the reset command.")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        self.action_queue.put(action)
        try:
            result = self.state_queue.get(timeout=10)
            if isinstance(result, Exception):
                raise result
            return result
        except Empty:
            raise TimeoutError("The game thread did not respond to the step command.")

    def close(self):
        """Cleans up all resources, ensuring the browser is closed."""
        if self.game_thread.is_alive():
            self.action_queue.put('close')
            self.stop_event.set()
            self.game_thread.join(timeout=10)
        self.logger.info("Environment resources cleaned up.")

    async def _cleanup_async(self):
        """Asynchronously and safely cleans up Playwright resources."""
        try:
            if self.page: 
                await self.page.close()
                self.page = None
            if self.browser: 
                await self.browser.close()
                self.browser = None
            if self.playwright: 
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            self.logger.error(f"Error during async cleanup: {e}")
            # Don't re-raise, as this is a cleanup method

    def render(self, mode='human'):
        """Renders the environment. Not implemented for this headless setup."""
        pass
        
    async def _recover_browser_session(self):
        """Attempts to recover the browser session after a failure."""
        self.logger.info("Attempting to recover browser session...")
        try:
            await self._cleanup_async()
            await asyncio.sleep(1)  # Give time for resources to be freed
            await self._init_browser()
            self.logger.info("Browser session recovery successful")
            return True
        except Exception as e:
            self.logger.error(f"Failed to recover browser session: {e}")
            return False
            
    def _wait_for_ready(self, timeout: int = 10):
        """Waits for the game thread to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = self.state_queue.get_nowait()
                if result == "ready":
                    return
                elif isinstance(result, Exception):
                    raise result
            except Empty:
                time.sleep(0.1)
        raise TimeoutError("Game thread did not become ready in time")
    
    def _restart_game_thread(self):
        """Restarts the game thread if it has died."""
        self.logger.info("Restarting game thread...")
        # Clean up any existing resources
        self.stop_event.set()
        if self.game_thread.is_alive():
            self.game_thread.join(timeout=5)
            
        # Reset state
        self.stop_event.clear()
        self.action_queue = Queue()
        self.state_queue = Queue()
        
        # Start new thread
        self.game_thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.game_thread.start()
        self._wait_for_ready() 
        self.logger.info("Game thread successfully restarted")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the DinoGameEnvironment.")
    parser.add_argument('--visualize', action='store_true', help='Enable visualization of processed frames.')
    args = parser.parse_args()

    print("Testing DinoGameEnvironment...")
    env = None
    
    # --- Visualization setup ---
    fig, axes = None, None
    if args.visualize:
        plt.ion()  # Turn on interactive mode
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        fig.suptitle("Processed Game Frames (Agent's View)")
    # -------------------------

    try:
        env = DinoGameEnvironment(headless=False)
        obs, info = env.reset()
        print(f"Initial state shape: {obs.shape}, dtype: {obs.dtype}")
        print(f"Action space: {env.action_space}")

        if args.visualize and obs is not None:
            for i in range(4):
                axes[i].imshow(obs[i], cmap='gray')
                axes[i].set_title(f"Frame {i+1}")
                axes[i].axis('off')
            plt.tight_layout()
            plt.pause(0.5)

        total_reward = 0
        for step_num in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            print(f"Step {step_num+1:03d}: Action={action}, Reward={reward:.3f}, Score={info['score']}, Done={done}")

            if args.visualize and obs is not None:
                for i in range(4):
                    axes[i].imshow(obs[i], cmap='gray')
                plt.pause(0.01) # Update plot

            if done:
                print(f"Episode finished after {step_num+1} steps! Resetting...")
                obs, info = env.reset()
                total_reward = 0
                if args.visualize and obs is not None:
                    for i in range(4):
                        axes[i].imshow(obs[i], cmap='gray')
                    plt.pause(0.5)

    except (Exception, TimeoutError) as e:
        print(f"\nAn error occurred during testing: {e}")
    finally:
        if env:
            env.close()
        if args.visualize:
            plt.ioff() # Turn off interactive mode
            plt.show() # Keep the last frame visible
        print("\nTest finished and environment closed.")
