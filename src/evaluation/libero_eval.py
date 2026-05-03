"""LIBERO benchmark evaluation wrapper for VL-JEPA.

Evaluates VL-JEPA policies on the LIBERO benchmark suite, which includes
four task families:
    - Spatial: Tasks requiring spatial reasoning (e.g., "pick up the left mug")
    - Object: Tasks requiring object recognition (e.g., "pick up the red bowl")
    - Goal: Tasks requiring goal understanding (e.g., "put the plate in the sink")
    - Long: Long-horizon tasks with multiple subgoals

Reference:
    Liu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning",
    NeurIPS 2023.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# LIBERO task suite names
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
SUITE_TASK_COUNTS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_long": 10,
}


@dataclass
class LIBEROEvalConfig:
    """Configuration for LIBERO evaluation."""

    suite_name: str = "libero_spatial"
    num_episodes: int = 20
    max_episode_steps: int = 300
    resolution: int = 256
    seed: int = 42
    device: str = "cuda"
    save_videos: bool = False
    video_dir: str = "eval_videos"


class LIBEROEvaluator:
    """Evaluate VL-JEPA policies on LIBERO benchmark suites.

    Loads task suites, runs closed-loop rollouts with the VL-JEPA policy,
    and computes per-task and aggregate success rates.

    Args:
        config: Evaluation configuration.
        policy: VL-JEPA policy module with an `act(obs, instruction) -> action` method.
        encoder: V-JEPA 2 encoder for visual feature extraction.
    """

    def __init__(
        self,
        config: LIBEROEvalConfig,
        policy: Any,
        encoder: Any,
    ) -> None:
        self.config = config
        self.policy = policy
        self.encoder = encoder
        self.device = torch.device(config.device)

        self._env = None
        self._task_descriptions: list[str] = []

    def _init_env(self) -> None:
        """Initialize the LIBERO environment."""
        try:
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
        except ImportError:
            raise ImportError(
                "LIBERO is required for evaluation. "
                "Install via: pip install libero"
            )

        benchmark = get_benchmark(self.config.suite_name)()
        self._num_tasks = benchmark.n_tasks
        self._task_descriptions = [
            benchmark.get_task(i).language for i in range(self._num_tasks)
        ]

        logger.info(
            "Initialized LIBERO evaluator: %s (%d tasks, %d episodes each)",
            self.config.suite_name, self._num_tasks, self.config.num_episodes,
        )

    def evaluate(self) -> dict[str, Any]:
        """Run full evaluation across all tasks in the suite.

        Returns:
            results: Dictionary with:
                - "suite_name": name of the evaluated suite
                - "per_task_success": list of per-task success rates
                - "aggregate_success": mean success rate across tasks
                - "task_descriptions": list of task language descriptions
                - "num_tasks": number of tasks
                - "num_episodes": episodes per task
        """
        if self._env is None:
            self._init_env()

        per_task_success: list[float] = []

        for task_idx in range(self._num_tasks):
            success_rate = self._evaluate_task(task_idx)
            per_task_success.append(success_rate)
            logger.info(
                "Task %d/%d (%s): %.1f%% success",
                task_idx + 1, self._num_tasks,
                self._task_descriptions[task_idx][:60],
                success_rate * 100,
            )

        aggregate = np.mean(per_task_success) if per_task_success else 0.0

        results = {
            "suite_name": self.config.suite_name,
            "per_task_success": per_task_success,
            "aggregate_success": float(aggregate),
            "task_descriptions": self._task_descriptions,
            "num_tasks": self._num_tasks,
            "num_episodes": self.config.num_episodes,
        }

        logger.info(
            "=== LIBERO %s: %.1f%% aggregate success ===",
            self.config.suite_name, aggregate * 100,
        )

        return results

    def _evaluate_task(self, task_idx: int) -> float:
        """Evaluate a single task.

        Args:
            task_idx: Index of the task in the suite.

        Returns:
            success_rate: Fraction of successful episodes [0, 1].
        """
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        benchmark = get_benchmark(self.config.suite_name)()
        task = benchmark.get_task(task_idx)
        task_description = task.language
        task_bddl_file = task.bddl_file

        successes = 0
        total = self.config.num_episodes

        for ep in range(total):
            # Create environment for this task
            env_args = {
                "bddl_file_name": task_bddl_file,
                "camera_heights": self.config.resolution,
                "camera_widths": self.config.resolution,
            }
            env = OffScreenRenderEnv(**env_args)

            try:
                obs = env.reset()
                success = self._run_episode(env, obs, task_description)
                if success:
                    successes += 1
            except Exception as e:
                logger.warning("Episode %d failed: %s", ep, str(e))
            finally:
                env.close()

        return successes / max(total, 1)

    def _run_episode(
        self,
        env: Any,
        initial_obs: dict,
        instruction: str,
    ) -> bool:
        """Run a single episode with the VL-JEPA policy.

        Args:
            env: LIBERO environment.
            initial_obs: Initial observation dict.
            instruction: Natural language task instruction.

        Returns:
            success: Whether the task was completed successfully.
        """
        obs = initial_obs
        frames: list[np.ndarray] = []

        for step in range(self.config.max_episode_steps):
            # Extract image observation
            image = self._extract_image(obs)  # [H, W, 3]
            frames.append(image)

            # Convert to tensor: [1, 3, 1, H, W]
            image_tensor = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(2)
                .float()
                .to(self.device) / 255.0
            )

            # Get proprioceptive state
            proprio = self._extract_proprio(obs)
            proprio_tensor = (
                torch.from_numpy(proprio)
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            # Policy action
            with torch.no_grad():
                action = self.policy.act(
                    images=image_tensor,
                    instruction=instruction,
                    proprioception=proprio_tensor,
                    encoder=self.encoder,
                )

            # action: [1, 4, 4] SE(3) → convert to env action format
            env_action = self._se3_to_env_action(action)

            # Step environment
            obs, reward, done, info = env.step(env_action)

            if done:
                break

        # Check success (LIBERO uses "task_success" in info or done flag)
        success = bool(info.get("task_success", False)) if isinstance(info, dict) else bool(done)
        return success

    def _extract_image(self, obs: dict) -> np.ndarray:
        """Extract RGB image from LIBERO observation.

        Args:
            obs: Observation dictionary from LIBERO env.

        Returns:
            image: [H, W, 3] uint8 RGB array.
        """
        # LIBERO observations typically have 'agentview_image' or 'robot0_eye_in_hand_image'
        if "agentview_image" in obs:
            img = obs["agentview_image"]
        elif "robot0_eye_in_hand_image" in obs:
            img = obs["robot0_eye_in_hand_image"]
        else:
            # Fallback: find first image key
            for key, val in obs.items():
                if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[-1] == 3:
                    img = val
                    break
            else:
                raise ValueError(f"No image found in observation keys: {list(obs.keys())}")

        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    def _extract_proprio(self, obs: dict) -> np.ndarray:
        """Extract proprioceptive state from LIBERO observation.

        Args:
            obs: Observation dictionary.

        Returns:
            proprio: [D] proprioceptive state (joint pos + gripper).
        """
        parts = []
        # Joint positions
        for key in ["robot0_joint_pos", "joint_positions"]:
            if key in obs:
                parts.append(obs[key])
        # Gripper state
        for key in ["robot0_gripper_qpos", "gripper_position"]:
            if key in obs:
                parts.append(obs[key])

        if parts:
            return np.concatenate(parts)
        # Fallback: 7-DOF (6 joints + 1 gripper)
        return np.zeros(7, dtype=np.float32)

    def _se3_to_env_action(self, T: Tensor) -> np.ndarray:
        """Convert SE(3) action matrix to LIBERO action format.

        LIBERO expects 7D actions: [dx, dy, dz, ax, ay, az, gripper]

        Args:
            T: [1, 4, 4] SE(3) action matrix.

        Returns:
            action: [7] numpy array.
        """
        from ..flow.se3_utils import se3_logmap

        T_np = T.detach().cpu()
        xi = se3_logmap(T_np)  # [1, 6]
        action_6d = xi[0].numpy()  # [6]

        # Add gripper action (default: open = 1.0)
        gripper = np.array([1.0], dtype=np.float32)
        return np.concatenate([action_6d, gripper])


def run_libero_benchmark(
    policy: Any,
    encoder: Any,
    suites: Optional[list[str]] = None,
    num_episodes: int = 20,
    device: str = "cuda",
) -> dict[str, dict]:
    """Run evaluation across multiple LIBERO suites.

    Args:
        policy: VL-JEPA policy.
        encoder: V-JEPA 2 encoder.
        suites: List of suite names (default: all four).
        num_episodes: Episodes per task.
        device: Device.

    Returns:
        all_results: Dictionary mapping suite name to results.
    """
    if suites is None:
        suites = SUITES

    all_results: dict[str, dict] = {}

    for suite in suites:
        config = LIBEROEvalConfig(
            suite_name=suite,
            num_episodes=num_episodes,
            device=device,
        )
        evaluator = LIBEROEvaluator(config, policy, encoder)
        results = evaluator.evaluate()
        all_results[suite] = results

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("LIBERO Benchmark Summary")
    logger.info("=" * 60)
    for suite, results in all_results.items():
        logger.info(
            "  %-20s: %.1f%%", suite, results["aggregate_success"] * 100
        )
    avg = np.mean([r["aggregate_success"] for r in all_results.values()])
    logger.info("  %-20s: %.1f%%", "Overall Average", avg * 100)
    logger.info("=" * 60)

    return all_results
