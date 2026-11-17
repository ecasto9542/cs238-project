import unittest

from environment import DatacenterEnv
from policies import (
    greedy_migration_policy,
    immediate_placement_cost,
    no_migration_policy,
)
from simulator import Server, VM


class BaselinePolicyTests(unittest.TestCase):
    def test_no_migration_returns_none(self) -> None:
        env = DatacenterEnv([Server(0)], [VM(0, 20.0, 20.0)], seed=1)
        env.reset()

        self.assertIsNone(no_migration_policy(env))

    def test_immediate_cost_includes_all_components(self) -> None:
        servers = [Server(0), Server(1)]
        vms = [VM(0, 80.0, 30.0), VM(1, 40.0, 80.0)]
        env = DatacenterEnv(
            servers,
            vms,
            seed=1,
            overload_penalty=1000.0,
            migration_penalty=10.0,
        )
        env.reset()

        cost = immediate_placement_cost(
            env,
            {0: 0, 1: 0},
            migrated=True,
        )

        self.assertEqual(cost, 30_260.0)

    def test_greedy_policy_eliminates_immediate_overload(self) -> None:
        servers = [Server(0), Server(1)]
        vms = [
            VM(0, 60.0, 10.0),
            VM(1, 10.0, 10.0),
            VM(2, 60.0, 10.0),
        ]
        env = DatacenterEnv(servers, vms, seed=1)
        env.reset()

        action = greedy_migration_policy(env)

        self.assertEqual(action, (0, 1))

    def test_greedy_policy_includes_do_nothing_candidate(self) -> None:
        env = DatacenterEnv(
            [Server(0), Server(1)],
            [VM(0, 20.0, 20.0)],
            seed=1,
        )
        env.reset()

        self.assertIsNone(greedy_migration_policy(env))

    def test_greedy_policy_does_not_mutate_environment(self) -> None:
        servers = [Server(0), Server(1)]
        vms = [VM(0, 60.0, 10.0), VM(1, 60.0, 10.0)]
        env = DatacenterEnv(servers, vms, seed=1)
        initial_state = env.reset()
        initial_demands = [(vm.cpu, vm.memory) for vm in env.vms]
        initial_rng_state = env.rng.getstate()

        greedy_migration_policy(env)

        self.assertEqual(env.get_state(), initial_state)
        self.assertEqual(
            [(vm.cpu, vm.memory) for vm in env.vms],
            initial_demands,
        )
        self.assertEqual(env.rng.getstate(), initial_rng_state)
        self.assertEqual(env.time, 0)


if __name__ == "__main__":
    unittest.main()
