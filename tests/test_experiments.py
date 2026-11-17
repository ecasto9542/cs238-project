import unittest

from environment import DatacenterEnv
from experiments import (
    run_baseline_experiments,
    run_episode,
    summarize_results,
)
from policies import no_migration_policy
from simulator import generate_problem


class ExperimentTests(unittest.TestCase):
    def test_run_episode_collects_requested_metrics(self) -> None:
        servers, vms = generate_problem(3, 4, seed=1)
        env = DatacenterEnv(servers, vms, horizon=5, seed=1)

        result = run_episode(env, no_migration_policy)

        self.assertEqual(result["steps"], 5)
        self.assertEqual(len(result["cumulative_costs"]), 5)
        self.assertEqual(result["migrations"], 0)
        self.assertEqual(result["total_migration_cost"], 0.0)
        self.assertAlmostEqual(
            result["cumulative_costs"][-1],
            -result["cumulative_reward"],
        )
        self.assertAlmostEqual(
            result["overload_fraction"],
            result["overload_events"] / 5,
        )

    def test_paired_experiment_runs_both_policies_for_each_seed(self) -> None:
        rows = run_baseline_experiments(seeds=[3, 7], horizon=5)

        self.assertEqual(len(rows), 4)
        self.assertEqual(
            {(row["seed"], row["policy"]) for row in rows},
            {
                (3, "No migration"),
                (3, "Greedy migration"),
                (7, "No migration"),
                (7, "Greedy migration"),
            },
        )

    def test_summary_averages_metrics_and_cost_curves(self) -> None:
        rows = run_baseline_experiments(seeds=[3, 7], horizon=5)

        summary = summarize_results(rows)

        self.assertEqual(len(summary), 2)
        for policy_summary in summary:
            self.assertEqual(policy_summary["episodes"], 2)
            self.assertEqual(len(policy_summary["mean_cumulative_costs"]), 5)
            self.assertGreater(policy_summary["mean_power"], 0.0)
            self.assertGreaterEqual(
                policy_summary["overload_steps_percent"],
                0.0,
            )
            self.assertGreaterEqual(policy_summary["mean_overload_events"], 0.0)
            self.assertLessEqual(
                policy_summary["overload_steps_percent"],
                100.0,
            )


if __name__ == "__main__":
    unittest.main()
