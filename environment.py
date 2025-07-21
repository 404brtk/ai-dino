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
import cv2
import matplotlib.pyplot as plt

from utils import FrameProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DinoGameEnvironment(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    GAME_CONSTANTS = {
        'distance_to_obstacle': {'min': 0, 'max': 600, 'default': 600},
        'obstacle_y_position': {'min': 80, 'max': 150, 'default': 140},
        'obstacle_width': {'min': 15, 'max': 80, 'default': 15},
        'current_speed': {'min': 6, 'max': 150, 'default': 6},
        'obstacle_height': {'min': 10, 'max': 60, 'default': 10}
    }

    def __init__(self,
                 frame_stack: int = 4,
                 processed_size: Tuple[int, int] = (84, 84),
                 max_episode_steps: int = 10000,
                 reward_scale: float = 1.0,
                 normalize_numerical: bool = True,
                 use_visual: bool = False, 
                 use_numerical: bool = True):
        super().__init__()

        # Core parameters
        self.headless = False
        self.frame_stack = frame_stack
        self.processed_size = processed_size
        self.max_episode_steps = max_episode_steps
        self.reward_scale = reward_scale
        self.normalize_numerical = normalize_numerical
        self.use_visual = use_visual
        self.use_numerical = use_numerical

        # Validate configuration
        if not self.use_visual and not self.use_numerical:
            raise ValueError("At least one of use_visual or use_numerical must be True")

        self.logger = logging.getLogger(__name__)

        # Initialize the frame processor utility ONLY if using visual
        if self.use_visual:
            self.frame_processor = FrameProcessor(target_size=self.processed_size, stack_frames=self.frame_stack)
        else:
            self.frame_processor = None

        # Gym-specific definitions
        self.action_space = spaces.Discrete(3)  # 0: Do Nothing, 1: Jump, 2: Duck

        processed_height, processed_width = self.processed_size

        # Build observation space based on what's enabled
        observation_spaces = {}
        
        if self.use_visual:
            observation_spaces['frame'] = gym.spaces.Box(
                low=0, high=1, shape=(self.frame_stack, processed_height, processed_width), dtype=np.float32
            )
        
        if self.use_numerical:
            if self.normalize_numerical:
                # Normalized to [0, 1] range
                observation_spaces.update({
                    'distance_to_obstacle': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    'obstacle_y_position': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    'obstacle_width': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    'current_speed': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                    'obstacle_height': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                })
            else:
                # Raw game values
                observation_spaces.update({
                    'distance_to_obstacle': gym.spaces.Box(
                        low=self.GAME_CONSTANTS['distance_to_obstacle']['min'], 
                        high=self.GAME_CONSTANTS['distance_to_obstacle']['max'], 
                        shape=(1,), dtype=np.float32
                    ),
                    'obstacle_y_position': gym.spaces.Box(
                        low=self.GAME_CONSTANTS['obstacle_y_position']['min'], 
                        high=self.GAME_CONSTANTS['obstacle_y_position']['max'], 
                        shape=(1,), dtype=np.float32
                    ),
                    'obstacle_width': gym.spaces.Box(
                        low=self.GAME_CONSTANTS['obstacle_width']['min'], 
                        high=self.GAME_CONSTANTS['obstacle_width']['max'], 
                        shape=(1,), dtype=np.float32
                    ),
                    'current_speed': gym.spaces.Box(
                        low=self.GAME_CONSTANTS['current_speed']['min'], 
                        high=self.GAME_CONSTANTS['current_speed']['max'], 
                        shape=(1,), dtype=np.float32
                    ),
                    'obstacle_height': gym.spaces.Box(
                        low=self.GAME_CONSTANTS['obstacle_height']['min'], 
                        high=self.GAME_CONSTANTS['obstacle_height']['max'], 
                        shape=(1,), dtype=np.float32
                    )
                })

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # State and performance tracking
        self.current_step = 0
        self.last_score = 0
        self.episode_reward = 0.0
        self.prev_obstacle_cleared = False
        self.last_obstacle_len = 0
        self.last_first_obstacle_x = -1

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

    def _normalize_numerical_feature(self, value: float, feature_name: str) -> float:
        """Normalize numerical features to [0, 1] range."""
        if not self.normalize_numerical:
            return value
            
        constants = self.GAME_CONSTANTS[feature_name]
        min_val = constants['min']
        max_val = constants['max']
        
        # Clamp value to valid range
        value = np.clip(value, min_val, max_val)
        
        # Normalize to [0, 1]
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.0
            
        return float(normalized)
    
    def _get_default_observation(self) -> Dict[str, np.ndarray]:
        """Get default observation when game state is unavailable."""
        observation = {}
        
        if self.use_visual:
            observation['frame'] = np.zeros((self.frame_stack, *self.processed_size), dtype=np.float32)
        
        if self.use_numerical:
            if self.normalize_numerical:
                # For normalized features, use reasonable defaults
                distance_default = self._normalize_numerical_feature(
                    self.GAME_CONSTANTS['distance_to_obstacle']['default'], 'distance_to_obstacle'
                )
                y_default = self._normalize_numerical_feature(
                    self.GAME_CONSTANTS['obstacle_y_position']['default'], 'obstacle_y_position'
                )
                width_default = self._normalize_numerical_feature(
                    self.GAME_CONSTANTS['obstacle_width']['default'], 'obstacle_width'
                )
                speed_default = self._normalize_numerical_feature(
                    self.GAME_CONSTANTS['current_speed']['default'], 'current_speed'
                )
                height_default = self._normalize_numerical_feature(
                    self.GAME_CONSTANTS['obstacle_height']['default'], 'obstacle_height'
                )
            else:
                # Raw values
                distance_default = float(self.GAME_CONSTANTS['distance_to_obstacle']['default'])
                y_default = float(self.GAME_CONSTANTS['obstacle_y_position']['default'])
                width_default = float(self.GAME_CONSTANTS['obstacle_width']['default'])
                speed_default = float(self.GAME_CONSTANTS['current_speed']['default'])
                height_default = float(self.GAME_CONSTANTS['obstacle_height']['default'])            
            observation.update({
                'distance_to_obstacle': np.array([distance_default], dtype=np.float32),
                'obstacle_y_position': np.array([y_default], dtype=np.float32),
                'obstacle_width': np.array([width_default], dtype=np.float32),
                'current_speed': np.array([speed_default], dtype=np.float32),
                'obstacle_height': np.array([height_default], dtype=np.float32)
            })
        
        return observation

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

    async def _capture_and_process_frame(self) -> Dict[str, np.ndarray]:
        """Captures and processes frame - optimized based on what's needed."""
        observation = {}
        
        try:
            # Get numerical features (always needed if use_numerical=True)
            if self.use_numerical:
                distance = await self._get_distance_to_obstacle()
                y_position = await self._get_obstacle_y_position()
                width = await self._get_obstacle_width()
                speed = await self._get_current_speed()
                height = await self._get_obstacle_height()

                observation.update({
                    'distance_to_obstacle': np.array([
                        self._normalize_numerical_feature(distance, 'distance_to_obstacle')
                    ], dtype=np.float32),
                    'obstacle_y_position': np.array([
                        self._normalize_numerical_feature(y_position, 'obstacle_y_position')
                    ], dtype=np.float32),
                    'obstacle_width': np.array([
                        self._normalize_numerical_feature(width, 'obstacle_width')
                    ], dtype=np.float32),
                    'current_speed': np.array([
                        self._normalize_numerical_feature(speed, 'current_speed')
                    ], dtype=np.float32),
                    'obstacle_height': np.array([
                        self._normalize_numerical_feature(height, 'obstacle_height')
                    ], dtype=np.float32)
                })

            # ONLY process visual if needed
            if self.use_visual:
                # 1. Get canvas bounding box
                canvas_handle = await self.page.query_selector('.runner-canvas')
                box = await canvas_handle.bounding_box() if canvas_handle else None

                # 2. Take screenshot
                screenshot_bytes = await self.page.screenshot(type='png')

                # 3. Decode PNG bytes → ndarray using OpenCV, then convert BGR → RGB
                img_arr = np.frombuffer(screenshot_bytes, dtype=np.uint8)
                raw_frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if raw_frame is None:
                    raise ValueError("Failed to decode screenshot bytes with OpenCV.")
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

                # 4. Crop the frame using the bounding box
                if box:
                    x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
                    raw_frame = raw_frame[y:y+h, x:x+w, :]
                else:
                    self.logger.warning("Game canvas not found, using uncropped frame.")

                # 5. Process the frame
                observation['frame'] = self.frame_processor.process_frame(raw_frame)

            return observation
            
        except Exception as e:
            self.logger.error(f"Error in _capture_and_process_frame: {str(e)}", 
                            exc_info=self.logger.level <= logging.DEBUG)
            # Return default observation
            return self._get_default_observation()

    # [Keep all the existing game state methods unchanged]
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
        """Retrieves the current speed, including any obstacle speed offset."""
        try:
            return await self.page.evaluate('''
                () => {
                    const runner = window.Runner.instance_;
                    let speedOffset = 0;

                    try {
                        const obstacles = runner.horizon.obstacles;
                        if (obstacles.length > 0 && 'speedOffset' in obstacles[0]) {
                            speedOffset = obstacles[0].speedOffset;
                        }
                    } catch (e) {
                        speedOffset = 0;
                    }

                    return runner.currentSpeed + speedOffset;
                }
            ''')
        except Exception as e:
            self.logger.error(f"Error getting current speed: {e}")
            return float(self.GAME_CONSTANTS['current_speed']['default'])
    
    async def _is_first_obstacle_cleared(self) -> bool:
        """Returns True if the closest obstacle's right edge is behind the T-Rex (i.e., just cleared)."""
        return await self.page.evaluate("""
        () => {
            const runner = window.Runner.instance_;
            const obs = runner.horizon.obstacles;
            if (!obs.length) return false;
            const first = obs[0];
            const tRexX = runner.tRex.xPos;
            return (first.xPos + first.width) < tRexX;
        }
        """)
        
    async def _get_distance_to_obstacle(self) -> float:
        """Returns the distance to the next obstacle."""
        try:
            result = await self.page.evaluate("""
            () => {
                const runner = window.Runner.instance_;
                const obstacles = runner.horizon.obstacles;
                
                if (!obstacles || obstacles.length === 0) {
                    return 600;
                }
                
                const firstObstacle = obstacles[0];
                const dinoX = runner.tRex.xPos;
                const distance = Math.max(0, firstObstacle.xPos - dinoX);
                
                return Math.min(distance, 600);
            }
        """)
            return float(result)
        except Exception as e:
            self.logger.error(f"Error getting obstacle distance: {e}")
            return float(self.GAME_CONSTANTS['distance_to_obstacle']['default'])
    
    async def _get_obstacle_y_position(self) -> float:
        """Returns the Y position of the next obstacle."""
        try:
            result = await self.page.evaluate("""
            () => {
                const runner = window.Runner.instance_;
                const obstacles = runner.horizon.obstacles;
                
                if (!obstacles || obstacles.length === 0) {
                    return 140; // Default ground level
                }
                
                const firstObstacle = obstacles[0];
                const pixels = 140; // Default height
                const height = firstObstacle.typeConfig?.height || 0;
                const yPos = firstObstacle.yPos || 0;
                
                return Math.max(0, pixels - (height + yPos));
            }
        """)
            return float(result)
        except Exception as e:
            self.logger.error(f"Error getting obstacle Y position: {e}")
            return float(self.GAME_CONSTANTS['obstacle_y_position']['default'])
    
    async def _get_obstacle_width(self) -> float:
        """Returns the width of the next obstacle."""
        try:
            result = await self.page.evaluate("""
            () => {
                const runner = window.Runner.instance_;
                const obstacles = runner.horizon.obstacles;
                
                if (!obstacles || obstacles.length === 0) {
                    return 0;
                }
                
                return obstacles[0].width || 0;
            }
        """)
            return float(result)
        except Exception as e:
            self.logger.error(f"Error getting obstacle width: {e}")
            return float(self.GAME_CONSTANTS['obstacle_width']['default'])

    async def _get_obstacle_len(self) -> int:
        """Returns the number of obstacles in the horizon."""
        return await self.page.evaluate('() => window.Runner.instance_.horizon.obstacles.length')

    async def _get_first_obstacle_x(self) -> float:
        """Returns the x-position of the first obstacle."""
        return await self.page.evaluate('() => window.Runner.instance_.horizon.obstacles.length ? \
        window.Runner.instance_.horizon.obstacles[0].xPos : -1')

    async def _get_obstacle_height(self) -> float:
        """Returns the height of the next obstacle."""
        try:
            result = await self.page.evaluate("""
            () => {
                const runner = window.Runner.instance_;
                const obstacles = runner.horizon.obstacles;
                
                if (!obstacles || obstacles.length === 0) {
                    return 0;
                }
                
                return obstacles[0].typeConfig.height || 0;
            }
        """)
            return float(result)
        except Exception as e:
            self.logger.error(f"Error getting obstacle height: {e}")
            return float(self.GAME_CONSTANTS['obstacle_height']['default'])

    async def _get_game_state(self) -> Dict[str, Any]:
        """Gets the current game state by calling helper methods."""
        try:
            # Use existing helper methods to fetch game state components
            crashed, score, speed, is_obstacle_cleared, obstacle_len, first_obstacle_x, obstacle_distance, obstacle_y, obstacle_width, obstacle_height = await asyncio.gather(
                self._is_game_over(),
                self._get_current_score(),
                self._get_current_speed(),
                self._is_first_obstacle_cleared(),
                self._get_obstacle_len(),
                self._get_first_obstacle_x(),
                self._get_distance_to_obstacle(),
                self._get_obstacle_y_position(),
                self._get_obstacle_width(),
                self._get_obstacle_height()
            )
            return {
                'crashed': crashed, 
                'score': score, 
                'speed': speed, 
                'is_obstacle_cleared': is_obstacle_cleared, 
                'obstacle_len': obstacle_len, 
                'first_obstacle_x': first_obstacle_x,
                'obstacle_distance': obstacle_distance,
                'obstacle_y': obstacle_y,
                'obstacle_width': obstacle_width,
                'obstacle_height': obstacle_height
            }
        except PlaywrightError as e:
            self.logger.error(f"Error getting game state: {e}")
            # Return a default state indicating a crash
            return {
                'crashed': True, 
                'score': 0, 
                'speed': self.GAME_CONSTANTS['current_speed']['default'], 
                'is_obstacle_cleared': False, 
                'obstacle_len': 0, 
                'first_obstacle_x': -1,
                'obstacle_distance': self.GAME_CONSTANTS['distance_to_obstacle']['default'],
                'obstacle_y': self.GAME_CONSTANTS['obstacle_y_position']['default'],
                'obstacle_width': self.GAME_CONSTANTS['obstacle_width']['default'],
                'obstacle_height': self.GAME_CONSTANTS['obstacle_height']['default']
            }
    
    def _calculate_reward(self, game_state: Dict[str, Any]) -> float:
        """Calculates the reward based on the final game state of a step."""
        if game_state.get('crashed', False):
            return -10.0  # Large penalty for crashing

        # Base survival reward
        reward = 0.1 * self.reward_scale

        # Score-based reward
        current_score = game_state.get('score', self.last_score)
        score_diff = current_score - self.last_score
        speed = game_state.get('speed', 0)
        is_obstacle_cleared = game_state.get('is_obstacle_cleared', self.prev_obstacle_cleared)
        obstacle_len = game_state.get('obstacle_len', self.last_obstacle_len)
        first_obstacle_x = game_state.get('first_obstacle_x', self.last_first_obstacle_x)

        # Reward for clearing an obstacle
        obstacle_cleared_event = False       
        # 1. An obstacle's right edge passed the T-Rex (rising edge detection)
        if is_obstacle_cleared and not self.prev_obstacle_cleared:
            obstacle_cleared_event = True
        # 2. An obstacle scrolled off screen (count decreased)
        elif obstacle_len < self.last_obstacle_len:
            obstacle_cleared_event = True
        # 3. An obstacle was replaced by a new one
        if (obstacle_len > 0 and self.last_obstacle_len == obstacle_len and
              first_obstacle_x > self.last_first_obstacle_x and self.last_first_obstacle_x != -1):
            obstacle_cleared_event = True
        
        if obstacle_cleared_event:
            #self.logger.info("***************** Obstacle cleared! *****************")
            reward += 2.0 * self.reward_scale  # Bonus for clearing an obstacle

        if score_diff > 0:
            # Higher reward for higher speeds (difficulty)
            speed_factor = min(1.0, speed / 10.0)
            reward += score_diff * 0.1 * speed_factor * self.reward_scale

        self.last_score = current_score
        self.prev_obstacle_cleared = is_obstacle_cleared
        self.last_obstacle_len = obstacle_len
        if first_obstacle_x != -1:
            self.last_first_obstacle_x = first_obstacle_x

        return reward

    async def _async_reset(self) -> Dict[str, np.ndarray]:
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
                            self.logger.error("All restart attempts failed. Returning default observation.")
                            return self._get_default_observation()
                    else:
                        await asyncio.sleep(0.5)  # Wait before next attempt

            # Wait for game to be fully ready
            await asyncio.sleep(0.2)
            
            # Verify final state
            final_crashed = await self._is_game_over()
            if final_crashed:
                self.logger.error("Game is still crashed after all restart attempts")
                return self._get_default_observation()

        except Exception as e:
            self.logger.error(f"Unexpected error during reset: {e}")
            return self._get_default_observation()

        # Reset the frame processor's internal buffer (only if using visual)
        if self.use_visual and self.frame_processor:
            self.frame_processor.reset()
        
        # Get the initial observation
        return await self._capture_and_process_frame()

    async def _async_step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Executes game action in the browser and returns results."""
        try:
            # 1. Perform action in browser
            if action == 1:  # Long Jump
                await self.page.keyboard.press('ArrowUp', delay=100)
            elif action == 2:  # Duck
                await self.page.keyboard.press('ArrowDown', delay=400)
            # action == 0: Do Nothing
            
            # 2. Get the definitive state of the game after the action
            game_state = await self._get_game_state()
            terminated = game_state.get('crashed', False)

            # 3. Calculate reward based on the final state
            reward = self._calculate_reward(game_state)

            # 4. Capture the observation (optimized based on what's needed)
            observation = await self._capture_and_process_frame()

            # 5. Update internal state and prepare return values
            self.episode_reward += reward
            self.current_step += 1
            truncated = self.current_step >= self.max_episode_steps

            info = {'score': game_state.get('score', 0), 'speed': game_state.get('speed', 0)}

            if terminated:
                self.logger.info(f"Episode terminated at step {self.current_step}. Score: {info['score']}")

            return observation, reward, terminated, truncated, info

        except PlaywrightError as e:
            self.logger.error(f"A Playwright error occurred during step: {e}")
            default_obs = self._get_default_observation()
            return (default_obs, -10.0, True, False, {'score': self.last_score, 'speed': 0, 'error': str(e)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset internal state
        self.current_step = 0
        self.last_score = 0
        self.episode_reward = 0.0
        self.prev_obstacle_cleared = False
        self.last_obstacle_len = 0
        self.last_first_obstacle_x = -1
        
        # Ensure the game thread is alive
        if not self.game_thread.is_alive():
            self.logger.warning("Game thread not alive during reset, restarting...")
            self._restart_game_thread()
            
        # Send reset command to game thread
        self.action_queue.put('reset')
        try:
            result = self.state_queue.get(timeout=10)
            if isinstance(result, Exception):
                raise result
            return result, {}
        except Empty:
            self.logger.error("Reset timeout: no response from game thread")
            
            # Check if the game thread is still alive
            if not self.game_thread.is_alive():
                self.logger.warning("Game thread has died during reset. Attempting to restart it...")
                self._restart_game_thread()
                
                # Clear the action queue and try again
                while not self.action_queue.empty():
                    try:
                        self.action_queue.get_nowait()
                    except Empty:
                        break
                        
                # Try reset one more time after restart
                self.action_queue.put('reset')
                try:
                    result = self.state_queue.get(timeout=10)
                    if isinstance(result, Exception):
                        raise result
                    self.logger.info("Successfully recovered after game thread restart during reset")
                    return result, {}
                except Empty:
                    self.logger.error("Still timeout after game thread restart during reset")
            
            # If we get here, recovery failed or wasn't attempted
            raise TimeoutError("The game thread did not respond to the reset command.")

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Executes one step in the environment."""
        # Send action to the game thread
        self.action_queue.put(action)
        try:
            result = self.state_queue.get(timeout=10)
            if isinstance(result, Exception):
                raise result
            return result
        except Empty:
            # First check if the game thread is still alive
            if not self.game_thread.is_alive():
                self.logger.warning("Game thread has died. Attempting to restart it...")
                self._restart_game_thread()
                # Clear the action queue and try again
                while not self.action_queue.empty():
                    try:
                        self.action_queue.get_nowait()
                    except Empty:
                        break
                # Try step one more time after restart
                self.action_queue.put(action)
                try:
                    result = self.state_queue.get(timeout=10)
                    if isinstance(result, Exception):
                        raise result
                    self.logger.info("Successfully recovered after game thread restart")
                    return result
                except Empty:
                    self.logger.error("Still timeout after game thread restart")
            
            # If we get here, recovery failed or wasn't attempted
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
        """Renders the environment."""
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
    parser.add_argument('--no-normalize', action='store_true', help='Disable numerical feature normalization.')
    parser.add_argument('--use-visual', action='store_true', help='Enable visual processing.')
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
        env = DinoGameEnvironment(
            normalize_numerical=not args.no_normalize,
            use_visual=args.visualize,  
            use_numerical=True
        )
        obs, info = env.reset()
        
        # Handle dictionary observation
        if isinstance(obs, dict):
            print(f"Observation keys: {list(obs.keys())}")
            if 'frame' in obs:
                print(f"Frame shape: {obs['frame'].shape}")
                frame_data = obs['frame']
            else:
                frame_data = None
                
            if 'distance_to_obstacle' in obs:
                print(f"Distance: {obs['distance_to_obstacle'][0]:.3f}")
                print(f"Y position: {obs['obstacle_y_position'][0]:.3f}")
                print(f"Width: {obs['obstacle_width'][0]:.3f}")
                print(f"Current speed: {obs['current_speed'][0]:.3f}")
                print(f"Obstacle height: {obs['obstacle_height'][0]:.3f}")
        else:
            print(f"Observation shape: {obs.shape}")
            frame_data = obs
            
        print(f"Action space: {env.action_space}")
        print(f"Visual processing: {env.use_visual}")
        print(f"Numerical processing: {env.use_numerical}")
        print(f"Normalization enabled: {env.normalize_numerical}")

        if args.visualize and frame_data is not None:
            for i in range(4):
                axes[i].imshow(frame_data[i], cmap='gray')
                axes[i].set_title(f"Frame {i+1}")
                axes[i].axis('off')
            plt.tight_layout()
            plt.pause(0.5)

        total_reward = 0
        for step_num in range(250):
            action = env.action_space.sample()
            #action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Extract obstacle info for logging
            if isinstance(obs, dict):
                log_str = f"Step {step_num+1:03d}: Action={action}, Reward={reward:.3f}, Score={info['score']}"
                
                if 'distance_to_obstacle' in obs:
                    distance = obs['distance_to_obstacle'][0]
                    y_pos = obs['obstacle_y_position'][0]
                    width = obs['obstacle_width'][0]
                    speed = obs['current_speed'][0]
                    height = obs['obstacle_height'][0]
                    log_str += f", Distance={distance:.3f}, Y={y_pos:.3f}, Width={width:.3f}, Speed={speed:.3f}, Height={height:.3f}"
                
                log_str += f", Done={done}"
                print(log_str)
                
                frame_data = obs.get('frame', None)
            else:
                print(f"Step {step_num+1:03d}: Action={action}, Reward={reward:.3f}, "
                      f"Score={info['score']}, Done={done}")
                frame_data = obs

            if args.visualize and frame_data is not None:
                for i in range(4):
                    axes[i].imshow(frame_data[i], cmap='gray')
                plt.pause(0.01)

            if done:
                print(f"Episode finished after {step_num+1} steps! Resetting...")
                obs, info = env.reset()
                total_reward = 0
                if args.visualize and isinstance(obs, dict) and 'frame' in obs:
                    frame_data = obs['frame']
                    for i in range(4):
                        axes[i].imshow(frame_data[i], cmap='gray')
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
