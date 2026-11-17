"""Minimal example of the sequential datacenter environment."""

from environment import DatacenterEnv
from simulator import generate_problem


def main() -> None:
    servers, vms = generate_problem(n_servers=3, n_vms=4, seed=42)
    env = DatacenterEnv(servers, vms, horizon=100, seed=42)

    print(f"Initial state: {env.reset()}")
    next_state, reward, done, info = env.step(None)
    print(f"Next state: {next_state}")
    print(f"Reward: {reward:.2f}")
    print(f"Done: {done}")
    print(f"Costs: {info}")


if __name__ == "__main__":
    main()
