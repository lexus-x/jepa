"""Main evaluation entry point.

Usage:
    python -m src.run_eval --policy=openvla --benchmark=libero --suite=libero_spatial
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from src.policies import OpenVLAPolicy, PiZeroPolicy, JEPAPolicy, RandomPolicy
from src.conformal import SafePolicyWrapper, OnlineConformalCalibrator
from src.evaluation.libero_eval import LIBEROEvaluator, LIBEROEvalConfig
from src.evaluation.metaworld_eval import MetaWorldEvaluator, MetaWorldEvalConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_policy(name: str, device: torch.device):
    """Build a policy by name."""
    policies = {
        "openvla": lambda: OpenVLAPolicy(device=device),
        "pi0": lambda: PiZeroPolicy(device=device),
        "jepa": lambda: JEPAPolicy(device=device),
        "random": lambda: RandomPolicy(),
    }
    if name not in policies:
        raise ValueError(f"Unknown policy '{name}'. Choose from: {list(policies.keys())}")
    return policies[name]()


def main():
    parser = argparse.ArgumentParser(description="SE(3) Conformal Safety Evaluation")
    parser.add_argument("--policy", default="random", help="Policy name")
    parser.add_argument("--benchmark", default="libero", choices=["libero", "metaworld"])
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--mt", default="MT10")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max_radius", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Build policy + conformal wrapper
    policy = build_policy(args.policy, device)
    calibrator = OnlineConformalCalibrator(alpha=args.alpha)
    safe_policy = SafePolicyWrapper(
        policy=policy,
        calibrator=calibrator,
        max_radius=args.max_radius,
    )

    logger.info("Policy: %s", policy.name())
    logger.info("Conformal: α=%.2f, max_radius=%.2f", args.alpha, args.max_radius)

    # Run evaluation
    if args.benchmark == "libero":
        config = LIBEROEvalConfig(
            suite_name=args.suite,
            num_episodes=args.num_episodes,
            device=str(device),
            conformal_alpha=args.alpha,
            max_radius=args.max_radius,
        )
        evaluator = LIBEROEvaluator(config, safe_policy)
        results = evaluator.evaluate()
    else:
        config = MetaWorldEvalConfig(
            benchmark=args.mt,
            num_episodes=args.num_episodes,
            device=str(device),
        )
        evaluator = MetaWorldEvaluator(config, safe_policy)
        results = evaluator.evaluate()

    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)
    for k, v in results.items():
        if isinstance(v, (int, float)):
            logger.info("  %s: %.4f", k, v)
        elif isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()):
            for k2, v2 in v.items():
                logger.info("  %s/%s: %.4f", k, k2, v2)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
