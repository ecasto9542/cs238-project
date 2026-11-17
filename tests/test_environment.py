import random
import unittest
from collections import Counter

from environment import (
    DatacenterEnv,
    WORKLOAD_LEVELS,
    WORKLOAD_TRANSITIONS,
    apply_workload_level,
    sample_next_workload_level,
    transition_vm_workload,
)
from simulator import Server, VM, generate_problem


class WorkloadTransitionTests(unittest.TestCase):
    def test_next_level_is_always_valid(self) -> None:
        rng = random.Random(1)

        for current_level in WORKLOAD_LEVELS:
            for _ in range(1_000):
                self.assertIn(
                    sample_next_workload_level(current_level, rng),
                    WORKLOAD_LEVELS,
                )

    def test_transition_frequencies_match_probabilities(self) -> None:
        samples_per_level = 50_000

        for seed, current_level in enumerate(WORKLOAD_LEVELS):
            with self.subTest(current_level=current_level):
                rng = random.Random(seed)
                counts = Counter(
                    sample_next_workload_level(current_level, rng)
                    for _ in range(samples_per_level)
                )

                for next_level, expected_probability in zip(
                    WORKLOAD_LEVELS,
                    WORKLOAD_TRANSITIONS[current_level],
                ):
                    observed_probability = counts[next_level] / samples_per_level
                    self.assertAlmostEqual(
                        observed_probability,
                        expected_probability,
                        delta=0.01,
                    )

    def test_seeded_transitions_are_reproducible(self) -> None:
        first_rng = random.Random(238)
        second_rng = random.Random(238)

        first_sequence = [
            sample_next_workload_level("medium", first_rng) for _ in range(100)
        ]
        second_sequence = [
            sample_next_workload_level("medium", second_rng) for _ in range(100)
        ]

        self.assertEqual(first_sequence, second_sequence)

    def test_demands_scale_from_fixed_baseline(self) -> None:
        vm = VM(vm_id=0, cpu=40.0, memory=20.0)

        expected_demands = {
            "low": (20.0, 10.0),
            "medium": (40.0, 20.0),
            "high": (60.0, 30.0),
        }
        for workload_level, expected in expected_demands.items():
            with self.subTest(workload_level=workload_level):
                apply_workload_level(vm, 40.0, 20.0, workload_level)
                self.assertEqual((vm.cpu, vm.memory), expected)

    def test_transition_updates_vm_to_sampled_level(self) -> None:
        expected_rng = random.Random(7)
        transition_rng = random.Random(7)
        expected_level = sample_next_workload_level("medium", expected_rng)
        vm = VM(vm_id=0, cpu=40.0, memory=20.0)

        next_level = transition_vm_workload(
            vm,
            current_level="medium",
            baseline_cpu=40.0,
            baseline_memory=20.0,
            rng=transition_rng,
        )

        self.assertEqual(next_level, expected_level)
        multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[next_level]
        self.assertEqual(vm.cpu, 40.0 * multiplier)
        self.assertEqual(vm.memory, 20.0 * multiplier)

    def test_unknown_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_next_workload_level("peak", random.Random(1))

        with self.assertRaises(ValueError):
            apply_workload_level(VM(0, 40.0, 20.0), 40.0, 20.0, "peak")


class DatacenterEnvTests(unittest.TestCase):
    def make_env(
        self,
        horizon: int = 100,
        seed: int = 42,
    ) -> DatacenterEnv:
        servers, vms = generate_problem(
            n_servers=3,
            n_vms=4,
            seed=42,
        )
        return DatacenterEnv(servers, vms, horizon=horizon, seed=seed)

    def test_reset_returns_round_robin_medium_state(self) -> None:
        env = self.make_env()

        state = env.reset()

        self.assertEqual(state[0], (0, 1, 2, 0))
        self.assertEqual(state[1], ("medium",) * 4)
        self.assertIsInstance(state, tuple)
        self.assertIsInstance(state[0], tuple)
        self.assertIsInstance(state[1], tuple)
        self.assertEqual(env.time, 0)

    def test_step_none_preserves_placement(self) -> None:
        env = self.make_env()
        initial_state = env.reset()

        next_state, _, done, info = env.step(None)

        self.assertEqual(next_state[0], initial_state[0])
        self.assertFalse(done)
        self.assertFalse(info["migrated"])
        self.assertEqual(info["migration_cost"], 0.0)

    def test_step_migrates_vm(self) -> None:
        env = self.make_env()
        env.reset()

        next_state, _, _, info = env.step((0, 2))

        self.assertEqual(next_state[0], (2, 1, 2, 0))
        self.assertTrue(info["migrated"])
        self.assertEqual(info["migration_cost"], 10.0)

    def test_invalid_actions_raise_value_error(self) -> None:
        cases = [
            (99, 1),
            (0, 99),
            (0, 0),
            (0,),
        ]

        for action in cases:
            with self.subTest(action=action):
                env = self.make_env()
                env.reset()
                with self.assertRaises(ValueError):
                    env.step(action)  # type: ignore[arg-type]

    def test_step_transitions_every_vm_to_valid_level(self) -> None:
        env = self.make_env()
        env.reset()

        next_state, _, _, _ = env.step(None)

        self.assertEqual(len(next_state[1]), 4)
        self.assertTrue(all(level in WORKLOAD_LEVELS for level in next_state[1]))
        for vm, workload_level in zip(env.vms, next_state[1]):
            baseline_cpu, baseline_memory = env.baseline_demands[vm.vm_id]
            multiplier = {
                "low": 0.5,
                "medium": 1.0,
                "high": 1.5,
            }[workload_level]
            self.assertEqual(vm.cpu, baseline_cpu * multiplier)
            self.assertEqual(vm.memory, baseline_memory * multiplier)

    def test_done_after_horizon_steps(self) -> None:
        env = self.make_env(horizon=3)
        env.reset()

        done_values = [env.step(None)[2] for _ in range(3)]

        self.assertEqual(done_values, [False, False, True])
        with self.assertRaises(RuntimeError):
            env.step(None)

    def test_same_seed_and_actions_reproduce_trajectory(self) -> None:
        first_env = self.make_env(seed=238)
        second_env = self.make_env(seed=238)
        actions = [None, (0, 2), None, (1, 0), None]

        first_trajectory = [first_env.reset()]
        second_trajectory = [second_env.reset()]
        for action in actions:
            first_trajectory.append(first_env.step(action))
            second_trajectory.append(second_env.step(action))

        self.assertEqual(first_trajectory, second_trajectory)

    def test_reset_restarts_seeded_trajectory(self) -> None:
        env = self.make_env(seed=238)
        first_trajectory = [env.reset(), env.step(None), env.step((0, 2))]
        second_trajectory = [env.reset(), env.step(None), env.step((0, 2))]

        self.assertEqual(first_trajectory, second_trajectory)

    def test_reward_contains_power_overload_and_migration_costs(self) -> None:
        servers = [Server(0), Server(1)]
        vms = [
            VM(vm_id=0, cpu=80.0, memory=30.0),
            VM(vm_id=1, cpu=40.0, memory=80.0),
        ]
        env = DatacenterEnv(
            servers,
            vms,
            seed=1,
            overload_penalty=1000.0,
            migration_penalty=10.0,
        )
        env.reset()

        _, reward, _, info = env.step((1, 0))

        self.assertEqual(info["power"], 250.0)
        self.assertEqual(info["overload_cost"], 30_000.0)
        self.assertEqual(info["migration_cost"], 10.0)
        self.assertTrue(info["migrated"])
        self.assertEqual(reward, -30_260.0)

    def test_capacity_violating_migration_is_allowed(self) -> None:
        servers = [Server(0), Server(1)]
        vms = [VM(0, 80.0, 80.0), VM(1, 80.0, 80.0)]
        env = DatacenterEnv(servers, vms, seed=1)
        env.reset()

        state, _, _, info = env.step((1, 0))

        self.assertEqual(state[0], (0, 0))
        self.assertGreater(info["overload_cost"], 0.0)

    def test_environment_must_be_reset_before_use(self) -> None:
        env = self.make_env()

        with self.assertRaises(RuntimeError):
            env.get_state()
        with self.assertRaises(RuntimeError):
            env.step(None)

    def test_constructor_validates_required_structure(self) -> None:
        with self.assertRaises(ValueError):
            DatacenterEnv([], [VM(0, 10.0, 10.0)])
        with self.assertRaises(ValueError):
            DatacenterEnv([Server(0)], [VM(0, 10.0, 10.0)], horizon=0)
        with self.assertRaises(ValueError):
            DatacenterEnv([Server(0)], [VM(1, 10.0, 10.0)])


if __name__ == "__main__":
    unittest.main()
