"""MetaWorld MT10/MT50 evaluation wrapper for VL-JEPA.

Evaluates VL-JEPA policies on MetaWorld multi-task benchmarks using the
gymnasium API.

Reference:
    Yu et al., "Meta-World: A Benchmark and Evaluation for Multi-Task and
    Meta Reinforcement Learning", CoRL 2019.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# MetaWorld benchmark variants
BENCHMARKS = {
    "MT10": "Meta-World/MT10",
    "MT50": "Meta-World/MT50",
    "ML1": "Meta-World/ML1",
    "ML10": "Meta-World/ML10",
    "ML45": "Meta-World/ML45",
}

# Task descriptions for MT10 (the 10 most common tasks)
MT10_TASK_DESCRIPTIONS = {
    "reach": "reach the target position with the gripper",
    "push": "push the puck to the target position",
    "pick-place": "pick up the puck and place it at the target",
    "door-open": "open the door by pulling the handle",
    "drawer-open": "open the drawer by pulling the handle",
    "drawer-close": "close the drawer by pushing it shut",
    "button-press": "press the button with the gripper",
    "peg-insert-side": "insert the peg into the hole from the side",
    "window-open": "open the window by sliding it",
    "window-close": "close the window by sliding it shut",
}


@dataclass
class MetaWorldEvalConfig:
    """Configuration for MetaWorld evaluation."""

    benchmark: str = "MT10"
    num_episodes: int = 20
    max_episode_steps: int = 300
    resolution: int = 256
    seed: int = 42
    device: str = "cuda"
    render_mode: Optional[str] = None  # "human", "rgb_array", or None
    camera_name: Optional[str] = None


class MetaWorldEvaluator:
    """Evaluate VL-JEPA policies on MetaWorld benchmarks.

    Uses the gymnasium API: ``gym.make("Meta-World/MT10", ...)``

    Args:
        config: Evaluation configuration.
        policy: VL-JEPA policy with an `act(obs, instruction) -> action` method.
        encoder: V-JEPA 2 encoder.
    """

    def __init__(
        self,
        config: MetaWorldEvalConfig,
        policy: Any,
        encoder: Any,
    ) -> None:
        self.config = config
        self.policy = policy
        self.encoder = encoder
        self.device = torch.device(config.device)

        self._env = None
        self._task_names: list[str] = []

    def _init_env(self) -> None:
        """Initialize the MetaWorld environment."""
        try:
            import metaworld
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "metaworld and gymnasium are required. "
                "Install via: pip install metaworld gymnasium"
            )

        benchmark_id = BENCHMARKS.get(self.config.benchmark, self.config.benchmark)

        # Create the benchmark
        if self.config.benchmark == "MT10":
            ml1 = metaworld.MT10(seed=self.config.seed)
            self._task_names = list(ml1.train_classes.keys())
        elif self.config.benchmark == "MT50":
            ml50 = metaworld.MT50(seed=self.config.seed)
            self._task_names = list(ml50.train_classes.keys())
        else:
            raise ValueError(f"Unsupported benchmark: {self.config.benchmark}")

        logger.info(
            "Initialized MetaWorld evaluator: %s (%d tasks, %d episodes each)",
            self.config.benchmark, len(self._task_names), self.config.num_episodes,
        )

    def evaluate(self) -> dict[str, Any]:
        """Run full evaluation across all tasks.

        Returns:
            results: Dictionary with per-task and aggregate success rates.
        """
        if self._env is None:
            self._init_env()

        import metaworld

        if self.config.benchmark == "MT10":
            benchmark = metaworld.MT10(seed=self.config.seed)
        elif self.config.benchmark == "MT50":
            benchmark = metaworld.MT50(seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported benchmark: {self.config.benchmark}")

        per_task_results: dict[str, float] = {}

        for task_name in self._task_names:
            success_rate = self._evaluate_task(benchmark, task_name)
            per_task_results[task_name] = success_rate
            logger.info(
                "Task %s: %.1f%% success",
                task_name, success_rate * 100,
            )

        aggregate = np.mean(list(per_task_results.values())) if per_task_results else 0.0

        results = {
            "benchmark": self.config.benchmark,
            "per_task_success": per_task_results,
            "aggregate_success": float(aggregate),
            "num_tasks": len(self._task_names),
            "num_episodes": self.config.num_episodes,
        }

        logger.info(
            "=== MetaWorld %s: %.1f%% aggregate success ===",
            self.config.benchmark, aggregate * 100,
        )

        return results

    def _evaluate_task(self, benchmark: Any, task_name: str) -> float:
        """Evaluate a single task.

        Args:
            benchmark: MetaWorld benchmark instance.
            task_name: Name of the task.

        Returns:
            success_rate: Fraction of successful episodes.
        """
        import gymnasium as gym

        # Get the task environment
        env_cls = benchmark.train_classes[task_name]
        tasks = [t for t in benchmark.train_tasks if t.env_name == task_name]

        if not tasks:
            logger.warning("No training tasks found for %s", task_name)
            return 0.0

        successes = 0
        total = self.config.num_episodes

        for ep in range(total):
            task = tasks[ep % len(tasks)]

            try:
                env = env_cls.render_mode(self.config.render_mode)
                env.set_task(task)
                obs, info = env.reset(seed=self.config.seed + ep)

                success = self._run_episode(env, obs, task_name)
                if success:
                    successes += 1
            except Exception as e:
                logger.warning("Episode %d for %s failed: %s", ep, task_name, str(e))
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        return successes / max(total, 1)

    def _run_episode(
        self,
        env: Any,
        initial_obs: np.ndarray,
        task_name: str,
    ) -> bool:
        """Run a single episode.

        Args:
            env: MetaWorld environment.
            initial_obs: Initial observation array.
            task_name: Name of the current task.

        Returns:
            success: Whether the task was completed.
        """
        obs = initial_obs

        # Get task description
        instruction = MT10_TASK_DESCRIPTIONS.get(
            task_name, f"complete the {task_name} task"
        )

        for step in range(self.config.max_episode_steps):
            # Render image from environment
            image = self._render_image(env)  # [H, W, 3]

            # Convert to tensor: [1, 3, 1, H, W]
            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(2)
                .float()
                .to(self.device) / 255.0
            )

            # Extract proprioception (end-effector pose + gripper)
            proprio = self._extract_proprio(obs)
            proprio_tensor = (
                torch.from_numpy(proprio)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            # Policy action
            with torch.no_grad():
                action_T = self.policy.act(
                    images=image_tensor,
                    instruction=instruction,
                    proprioception=proprio_tensor,
                    encoder=self.encoder,
                )

            # Convert SE(3) to MetaWorld action format
            env_action = self._se3_to_env_action(action_T)

            # Step
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            # Check success
            if isinstance(info, dict) and info.get("success", False):
                return True
            if done:
                break

        return False

    def _render_image(self, env: Any) -> np.ndarray:
        """Render an image from the MetaWorld environment.

        Args:
            env: MetaWorld environment.

        Returns:
            image: [H, W, 3] uint8 RGB array.
        """
        try:
            img = env.render()
            if img is None:
                # Some environments need explicit render call
                img = env.unwrapped.sim.render(
                    self.config.resolution, self.config.resolution,
                    camera_name=self.config.camera_name or "corner",
                )
        except Exception:
            # Fallback: render from simulator
            img = env.unwrapped.sim.render(
                self.config.resolution, self.config.resolution,
                camera_name=self.config.camera_name or "corner",
            )

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        # Ensure correct orientation (some renders are flipped)
        if img.shape[0] != self.config.resolution:
            img = np.flip(img, axis=0).copy()

        return img

    def _extract_proprio(self, obs: np.ndarray) -> np.ndarray:
        """Extract proprioceptive state from MetaWorld observation.

        MetaWorld observations are typically 39D or 60D depending on the env.
        We extract end-effector position (first 3) and gripper (last 2).

        Args:
            obs: Observation array.

        Returns:
            proprio: [7] proprioceptive state.
        """
        if len(obs) >= 4:
            # First 3 = end-effector pos, last 2 = gripper fingers
            eef_pos = obs[:3]
            gripper = obs[-2:] if len(obs) >= 5 else np.zeros(2)
            # Pad to 7D: [eef_x, eef_y, eef_z, gripper_left, gripper_right, 0, 0]
            proprio = np.zeros(7, dtype=np.float32)
            proprio[:3] = eef_pos
            proprio[3:3 + len(gripper)] = gripper
            return proprio
        return np.zeros(7, dtype=np.float32)

    def _se3_to_env_action(self, T: Tensor) -> np.ndarray:
        """Convert SE(3) action to MetaWorld action format.

        MetaWorld expects 4D actions: [dx, dy, dz, gripper]

        Args:
            T: [1, 4, 4] SE(3) action matrix.

        Returns:
            action: [4] numpy array.
        """
        from ..flow.se3_utils import se3_logmap

        T_np = T.detach().cpu()
        xi = se3_logmap(T_np)  # [1, 6]
        translation = xi[0, 3:].numpy()  # [3] linear velocity

        # Gripper: positive = close, negative = open
        gripper = np.array([-1.0], dtype=np.float32)  # Default: open

        return np.concatenate([translation, gripper])


def run_metaworld_benchmark(
    policy: Any,
    encoder: Any,
    benchmark: str = "MT10",
    num_episodes: int = 20,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run evaluation on a MetaWorld benchmark.

    Args:
        policy: VL-JEPA policy.
        encoder: V-JEPA 2 encoder.
        benchmark: Benchmark name ("MT10" or "MT50").
        num_episodes: Episodes per task.
        device: Device.

    Returns:
        results: Evaluation results dictionary.
    """
    config = MetaWorldEvalConfig(
        benchmark=benchmark,
        num_episodes=num_episodes,
        device=device,
    )
    evaluator = MetaWorldEvaluator(config, policy, encoder)
    return evaluator.evaluate()
