"""Tests de non-regression pour le simulateur minier."""

from __future__ import annotations

import math
import unittest

from memoire_master_rl_logistique.baselines.fifo_policy import FIFOPolicy
from memoire_master_rl_logistique.env.mine_env import MineEnv


class TestMineEnv(unittest.TestCase):
    """Validation minimale de l'environnement et des politiques de base."""

    def test_reset_returns_valid_observation_and_info(self) -> None:
        env = MineEnv(truck_count=3, shovel_count=2, dump_count=1)

        obs, info = env.reset(seed=123)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(info["current_time_min"], 0.0)
        self.assertEqual(len(info["shovel_available_at"]), 2)
        self.assertEqual(len(info["dump_available_at"]), 1)

    def test_dispatch_action_completes_a_cycle(self) -> None:
        env = MineEnv(
            truck_count=2,
            shovel_count=1,
            dump_count=1,
            breakdown_probability=0.0,
        )
        env.reset(seed=123)

        _obs, reward, terminated, truncated, info = env.step(0)

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertGreater(info["total_tonnage_t"], 0.0)
        self.assertEqual(info["total_cycles"], 1)
        self.assertTrue(math.isfinite(reward))

    def test_wait_action_only_waits(self) -> None:
        env = MineEnv(
            truck_count=2,
            shovel_count=1,
            dump_count=1,
            breakdown_probability=0.0,
            wait_action_minutes=3.0,
        )
        env.reset(seed=123)
        wait_action = env.shovel_count * env.dump_count

        _obs, reward, _terminated, _truncated, info = env.step(wait_action)

        self.assertEqual(info["total_tonnage_t"], 0.0)
        self.assertEqual(info["total_cycles"], 0)
        self.assertGreater(info["total_wait_min"], 0.0)
        self.assertLess(reward, 0.0)

    def test_fifo_policy_does_not_use_wait_action(self) -> None:
        env = MineEnv(truck_count=2, shovel_count=2, dump_count=2)
        obs, info = env.reset(seed=123)
        policy = FIFOPolicy(num_shovels=2, num_dumps=2)

        action = policy.predict(obs, info)

        self.assertGreaterEqual(action, 0)
        self.assertLess(action, env.shovel_count * env.dump_count)


if __name__ == "__main__":
    unittest.main()
