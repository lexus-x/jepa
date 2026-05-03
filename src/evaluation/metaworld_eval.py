"""MetaWorld benchmark evaluation with conformal safety.

Evaluates VLA policies wrapped with SafePolicyWrapper on MetaWorld
MT10/MT50 benchmarks, tracking conformal safety metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from ..conformal.safe_policy import SafePolicyWrapper

logger = logging.getLogger(__name__)

BENCHMARKS = {"MT10": "Meta-World/MT10", "MT50": "Meta-World/MT50"}

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
    benchmark: str = "MT10"
    num_episodes: int = 20
    max_episode_steps: int = 300
    resolution: int = 256
    seed: int = 42
    device: str = "cuda"


class MetaWorldEvaluator:
    """Evaluate VLA policies on MetaWorld with conformal safety.

    Args:
        config: Evaluation configuration.
        safe_policy: Wrapped policy with conformal safety.
    """

    def __init__(
        self,
        config: MetaWorldEvalConfig,
        safe_policy: SafePolicyWrapper,
    ) -> None:
        self.config = config
        self.safe_policy = safe_policy

    def evaluate(self) -> dict[str, Any]:
        """Run full evaluation."""
        import metaworld

        if self.config.benchmark == "MT10":
            benchmark = metaworld.MT10(seed=self.config.seed)
        elif self.config.benchmark == "MT50":
            benchmark = metaworld.MT50(seed=self.config.seed)
        else:
            raise ValueError(f"Unsupported: {self.config.benchmark}")

        task_names = list(benchmark.train_classes.keys())
        per_task = {}

        for task_name in task_names:
            success_rate = self._evaluate_task(benchmark, task_name)
            per_task[task_name] = success_rate
            logger.info("Task %s: %.1f%%", task_name, success_rate * 100)

        stats = self.safe_policy.stats
        aggregate = np.mean(list(per_task.values()))

        return {
            "benchmark": self.config.benchmark,
            "per_task": per_task,
            "aggregate_success": float(aggregate),
            "conformal_stats": stats,
        }

    def _evaluate_task(self, benchmark: Any, task_name: str) -> float:
        import gymnasium as gym

        env_cls = benchmark.train_classes[task_name]
        tasks = [t for t in benchmark.train_tasks if t.env_name == task_name]
        if not tasks:
            return 0.0

        successes = 0
        for ep in range(self.config.num_episodes):
            task = tasks[ep % len(tasks)]
            try:
                env = env_cls()
                env.set_task(task)
                obs, _ = env.reset(seed=self.config.seed + ep)
                success = self._run_episode(env, obs, task_name)
                if success:
                    successes += 1
            except Exception as e:
                logger.warning("Episode %d for %s: %s", ep, task_name, str(e))
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        return successes / max(self.config.num_episodes, 1)

    def _run_episode(self, env: Any, initial_obs: np.ndarray, task_name: str) -> bool:
        obs = initial_obs
        instruction = MT10_TASK_DESCRIPTIONS.get(task_name, f"complete {task_name}")

        for _ in range(self.config.max_episode_steps):
            image = self._render_image(env)
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1)
                .unsqueeze(0).unsqueeze(2).float() / 255.0
            )

            proprio = self._extract_proprio(obs)
            proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float()

            observation = {"image": image_tensor, "proprioception": proprio_tensor}
            T_pred, info = self.safe_policy.act(observation, instruction)

            env_action = self._se3_to_env_action(T_pred)
            obs, reward, terminated, truncated, env_info = env.step(env_action)

            if isinstance(env_info, dict) and env_info.get("success", False):
                return True
            if terminated or truncated:
                break

        return False

    def _render_image(self, env: Any) -> np.ndarray:
        try:
            img = env.render()
        except Exception:
            img = env.unwrapped.sim.render(
                self.config.resolution, self.config.resolution,
                camera_name="corner",
            )
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    def _extract_proprio(self, obs: np.ndarray) -> np.ndarray:
        proprio = np.zeros(7, dtype=np.float32)
        if len(obs) >= 3:
            proprio[:3] = obs[:3]
        if len(obs) >= 5:
            proprio[3:5] = obs[-2:]
        return proprio

    def _se3_to_env_action(self, T: Tensor) -> np.ndarray:
        from ..flow.se3_utils import se3_logmap
        xi = se3_logmap(T.detach().cpu())
        translation = xi[0, 3:].numpy()
        return np.concatenate([translation, [-1.0]])
