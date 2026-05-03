"""LIBERO benchmark evaluation with conformal safety.

Evaluates any VLA policy wrapped with SafePolicyWrapper on the LIBERO
benchmark suite, tracking conformal coverage, radius evolution, and
safety interventions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from ..conformal.safe_policy import SafePolicyWrapper
from ..policies.base import BasePolicy

logger = logging.getLogger(__name__)

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]


@dataclass
class LIBEROEvalConfig:
    suite_name: str = "libero_spatial"
    num_episodes: int = 20
    max_episode_steps: int = 300
    resolution: int = 256
    seed: int = 42
    device: str = "cuda"
    conformal_alpha: float = 0.1
    max_radius: float = 2.0


class LIBEROEvaluator:
    """Evaluate VLA policies on LIBERO with conformal safety tracking.

    Args:
        config: Evaluation configuration.
        safe_policy: Wrapped policy with conformal safety.
    """

    def __init__(
        self,
        config: LIBEROEvalConfig,
        safe_policy: SafePolicyWrapper,
    ) -> None:
        self.config = config
        self.safe_policy = safe_policy

    def evaluate(self) -> dict[str, Any]:
        """Run full evaluation.

        Returns:
            results: Per-task and aggregate metrics including conformal stats.
        """
        from libero.libero.benchmark import get_benchmark

        benchmark = get_benchmark(self.config.suite_name)()
        num_tasks = benchmark.n_tasks

        per_task_results = []
        for task_idx in range(num_tasks):
            task = benchmark.get_task(task_idx)
            task_result = self._evaluate_task(benchmark, task, task_idx)
            per_task_results.append(task_result)
            logger.info(
                "Task %d/%d (%s): success=%.1f%%, coverage=%.1f%%, radius=%.3f, interventions=%d",
                task_idx + 1, num_tasks, task.language[:50],
                task_result["success_rate"] * 100,
                task_result["coverage_rate"] * 100,
                task_result["mean_radius"],
                task_result["interventions"],
            )

        # Aggregate
        stats = self.safe_policy.stats
        return {
            "suite": self.config.suite_name,
            "per_task": per_task_results,
            "aggregate_success": np.mean([r["success_rate"] for r in per_task_results]),
            "aggregate_coverage": stats["coverage_rate"],
            "aggregate_fallback_rate": stats["fallback_rate"],
            "conformal_stats": stats,
        }

    def _evaluate_task(self, benchmark: Any, task: Any, task_idx: int) -> dict:
        from libero.libero.envs import OffScreenRenderEnv

        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": self.config.resolution,
            "camera_widths": self.config.resolution,
        }

        successes = 0
        interventions = 0
        radii = []

        for ep in range(self.config.num_episodes):
            env = OffScreenRenderEnv(**env_args)
            try:
                obs = env.reset()
                result = self._run_episode(env, obs, task.language)
                if result["success"]:
                    successes += 1
                interventions += result["interventions"]
                radii.extend(result["radii"])
            except Exception as e:
                logger.warning("Episode %d failed: %s", ep, str(e))
            finally:
                env.close()

        n = self.config.num_episodes
        return {
            "task": task.language,
            "success_rate": successes / n,
            "coverage_rate": self.safe_policy.stats["coverage_rate"],
            "mean_radius": np.mean(radii) if radii else 0.0,
            "interventions": interventions,
        }

    def _run_episode(self, env: Any, initial_obs: dict, instruction: str) -> dict:
        obs = initial_obs
        successes = 0
        interventions = 0
        radii = []

        for step in range(self.config.max_episode_steps):
            image = self._extract_image(obs)
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1)
                .unsqueeze(0).unsqueeze(2).float() / 255.0
            )

            proprio = self._extract_proprio(obs)
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float()

            observation = {
                "image": image_tensor,
                "proprioception": proprio_tensor,
            }

            T_pred, info = self.safe_policy.act(observation, instruction)
            radii.append(info["radius"])
            if info["fallback"]:
                interventions += 1

            env_action = self._se3_to_env_action(T_pred)
            obs, reward, done, env_info = env.step(env_action)

            if isinstance(env_info, dict) and env_info.get("task_success", False):
                return {"success": True, "interventions": interventions, "radii": radii}
            if done:
                break

        return {"success": False, "interventions": interventions, "radii": radii}

    def _extract_image(self, obs: dict) -> np.ndarray:
        for key in ["agentview_image", "robot0_eye_in_hand_image"]:
            if key in obs:
                img = obs[key]
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                return img
        raise ValueError(f"No image in obs: {list(obs.keys())}")

    def _extract_proprio(self, obs: dict) -> np.ndarray:
        parts = []
        for key in ["robot0_joint_pos", "joint_positions", "robot0_gripper_qpos"]:
            if key in obs:
                parts.append(obs[key])
        return np.concatenate(parts) if parts else np.zeros(7, dtype=np.float32)

    def _se3_to_env_action(self, T: Tensor) -> np.ndarray:
        from ..flow.se3_utils import se3_logmap
        xi = se3_logmap(T.detach().cpu())
        action_6d = xi[0].numpy()
        return np.concatenate([action_6d, [1.0]])


def run_libero_benchmark(
    safe_policy: SafePolicyWrapper,
    suites: Optional[list[str]] = None,
    num_episodes: int = 20,
    device: str = "cuda",
) -> dict[str, dict]:
    """Run LIBERO benchmark across multiple suites."""
    if suites is None:
        suites = SUITES

    all_results = {}
    for suite in suites:
        config = LIBEROEvalConfig(suite_name=suite, num_episodes=num_episodes, device=device)
        evaluator = LIBEROEvaluator(config, safe_policy)
        all_results[suite] = evaluator.evaluate()

    logger.info("\n" + "=" * 60)
    logger.info("LIBERO Benchmark Summary")
    logger.info("=" * 60)
    for suite, results in all_results.items():
        logger.info(
            "  %-20s: success=%.1f%%, coverage=%.1f%%, interventions=%.1f%%",
            suite,
            results["aggregate_success"] * 100,
            results["aggregate_coverage"] * 100,
            results["aggregate_fallback_rate"] * 100,
        )
    logger.info("=" * 60)

    return all_results
