"""Run and compare baseline policies over stochastic workload episodes."""

import os
import tempfile
from collections.abc import Callable, Iterable

from environment import DatacenterEnv
from policies import Action, greedy_migration_policy, no_migration_policy
from simulator import generate_problem


Policy = Callable[[DatacenterEnv], Action]

POLICIES: tuple[tuple[str, Policy], ...] = (
    ("No migration", no_migration_policy),
    ("Greedy migration", greedy_migration_policy),
)


def run_episode(env: DatacenterEnv, policy: Policy) -> dict:
    """Run one episode and return reward and component-cost metrics."""
    env.reset()
    cumulative_reward = 0.0
    total_power = 0.0
    overload_events = 0
    total_overload_penalty = 0.0
    migrations = 0
    total_migration_cost = 0.0
    cumulative_costs: list[float] = []
    cumulative_cost = 0.0

    done = False
    steps = 0
    while not done:
        action = policy(env)
        _, reward, done, info = env.step(action)
        step_cost = -reward

        cumulative_reward += reward
        total_power += float(info["power"])
        total_overload_penalty += float(info["overload_cost"])
        total_migration_cost += float(info["migration_cost"])
        overload_events += int(float(info["overload_cost"]) > 0.0)
        migrations += int(bool(info["migrated"]))
        cumulative_cost += step_cost
        cumulative_costs.append(cumulative_cost)
        steps += 1

    return {
        "steps": steps,
        "cumulative_reward": cumulative_reward,
        "mean_power": total_power / steps,
        "overload_events": overload_events,
        "overload_fraction": overload_events / steps,
        "total_overload_penalty": total_overload_penalty,
        "migrations": migrations,
        "total_migration_cost": total_migration_cost,
        "cumulative_costs": cumulative_costs,
    }


def run_baseline_experiments(
    seeds: Iterable[int] = range(20),
    n_servers: int = 3,
    n_vms: int = 4,
    horizon: int = 100,
    overload_penalty: float = 1000.0,
    migration_penalty: float = 10.0,
) -> list[dict]:
    """Run paired policy episodes with identical problems and random draws."""
    rows = []
    for seed in seeds:
        for policy_name, policy in POLICIES:
            servers, vms = generate_problem(
                n_servers=n_servers,
                n_vms=n_vms,
                seed=seed,
            )
            env = DatacenterEnv(
                servers,
                vms,
                horizon=horizon,
                seed=seed,
                overload_penalty=overload_penalty,
                migration_penalty=migration_penalty,
            )
            row = run_episode(env, policy)
            row["seed"] = seed
            row["policy"] = policy_name
            rows.append(row)
    return rows


def summarize_results(rows: list[dict]) -> list[dict]:
    """Average episode metrics and cumulative-cost curves by policy."""
    summary = []
    for policy_name, _ in POLICIES:
        policy_rows = [row for row in rows if row["policy"] == policy_name]
        if not policy_rows:
            continue

        episodes = len(policy_rows)
        horizon = int(policy_rows[0]["steps"])
        mean_cumulative_costs = [
            sum(float(row["cumulative_costs"][step]) for row in policy_rows)
            / episodes
            for step in range(horizon)
        ]
        summary.append({
            "policy": policy_name,
            "episodes": episodes,
            "mean_power": _mean(policy_rows, "mean_power"),
            "mean_overload_events": _mean(policy_rows, "overload_events"),
            "overload_steps_percent": (
                _mean(policy_rows, "overload_fraction") * 100.0
            ),
            "mean_migrations": _mean(policy_rows, "migrations"),
            "mean_cumulative_reward": _mean(
                policy_rows,
                "cumulative_reward",
            ),
            "mean_total_overload_penalty": _mean(
                policy_rows,
                "total_overload_penalty",
            ),
            "mean_total_migration_cost": _mean(
                policy_rows,
                "total_migration_cost",
            ),
            "mean_cumulative_costs": mean_cumulative_costs,
        })
    return summary


def _mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def print_results_table(summary: list[dict]) -> None:
    """Print the primary baseline-comparison metrics."""
    header = (
        f"{'Policy':<18} {'Mean power (W)':>15} "
        f"{'Overload steps':>16} {'Migrations':>12} "
        f"{'Cumulative reward':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['policy']:<18} "
            f"{row['mean_power']:>15.2f} "
            f"{row['overload_steps_percent']:>15.2f}% "
            f"{row['mean_migrations']:>12.2f} "
            f"{row['mean_cumulative_reward']:>18.2f}"
        )

    print("\nComponent totals per episode")
    for row in summary:
        print(
            f"{row['policy']}: overload penalty="
            f"{row['mean_total_overload_penalty']:.2f}, migration cost="
            f"{row['mean_total_migration_cost']:.2f}"
        )


def save_cumulative_cost_plot(
    summary: list[dict],
    path: str = "baseline_cumulative_cost.png",
) -> None:
    """Save mean cumulative operating cost over time for each policy."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        os.path.join(tempfile.gettempdir(), "cs238-matplotlib"),
    )
    os.environ.setdefault(
        "XDG_CACHE_HOME",
        os.path.join(tempfile.gettempdir(), "cs238-cache"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for row in summary:
        costs = row["mean_cumulative_costs"]
        ax.plot(
            range(1, len(costs) + 1),
            costs,
            label=row["policy"],
            linewidth=2,
        )

    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean cumulative operating cost")
    ax.set_title("Baseline Policy Cost Under Stochastic Workloads")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = run_baseline_experiments()
    summary = summarize_results(rows)
    print_results_table(summary)
    save_cumulative_cost_plot(summary)
    print("\nSaved cumulative-cost plot to baseline_cumulative_cost.png")


if __name__ == "__main__":
    main()
