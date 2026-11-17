"""Sequential datacenter environment with stochastic VM workloads."""

import random

from simulator import PENALTY, Server, VM, _build_loads


WORKLOAD_LEVELS = ("low", "medium", "high")

WORKLOAD_MULTIPLIERS = {
    "low": 0.5,
    "medium": 1.0,
    "high": 1.5,
}

WORKLOAD_TRANSITIONS = {
    "low": (0.7, 0.3, 0.0),
    "medium": (0.2, 0.6, 0.2),
    "high": (0.0, 0.3, 0.7),
}


def sample_next_workload_level(
    current_level: str,
    rng: random.Random,
) -> str:
    """Sample one VM's next workload level from the transition matrix."""
    if current_level not in WORKLOAD_TRANSITIONS:
        raise ValueError(f"Unknown workload level: {current_level!r}")

    draw = rng.random()
    cumulative_probability = 0.0
    for next_level, probability in zip(
        WORKLOAD_LEVELS,
        WORKLOAD_TRANSITIONS[current_level],
    ):
        cumulative_probability += probability
        if draw < cumulative_probability:
            return next_level

    # Protect against a possible floating-point roundoff gap at 1.0.
    return WORKLOAD_LEVELS[-1]


def demands_for_workload_level(
    baseline_cpu: float,
    baseline_memory: float,
    workload_level: str,
) -> tuple[float, float]:
    """Scale a VM's baseline demands without changing its resource profile."""
    if workload_level not in WORKLOAD_MULTIPLIERS:
        raise ValueError(f"Unknown workload level: {workload_level!r}")

    multiplier = WORKLOAD_MULTIPLIERS[workload_level]
    return baseline_cpu * multiplier, baseline_memory * multiplier


def apply_workload_level(
    vm: VM,
    baseline_cpu: float,
    baseline_memory: float,
    workload_level: str,
) -> None:
    """Update a VM's demands from its fixed baseline and workload level."""
    vm.cpu, vm.memory = demands_for_workload_level(
        baseline_cpu,
        baseline_memory,
        workload_level,
    )


def transition_vm_workload(
    vm: VM,
    current_level: str,
    baseline_cpu: float,
    baseline_memory: float,
    rng: random.Random,
) -> str:
    """Sample and apply one workload transition, returning the new level."""
    next_level = sample_next_workload_level(current_level, rng)
    apply_workload_level(vm, baseline_cpu, baseline_memory, next_level)
    return next_level


class DatacenterEnv:
    """Finite-horizon VM placement environment.

    A state contains immutable placement and workload tuples ordered the same
    way as ``vms``. An action is ``None`` or ``(vm_id, target_server_id)``.
    Capacity-violating migrations are allowed and charged through the overload
    cost. Calling ``reset()`` restarts the random stream from the constructor's
    seed, so repeated episodes are reproducible when a seed is supplied.
    """

    def __init__(
        self,
        servers: list[Server],
        vms: list[VM],
        horizon: int = 100,
        seed: int | None = None,
        overload_penalty: float = PENALTY,
        migration_penalty: float = 10.0,
    ):
        if not servers:
            raise ValueError("At least one server is required")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        expected_vm_ids = list(range(len(vms)))
        actual_vm_ids = [vm.vm_id for vm in vms]
        if actual_vm_ids != expected_vm_ids:
            raise ValueError("VM IDs must match their indices in vms")

        server_ids = [server.server_id for server in servers]
        if len(server_ids) != len(set(server_ids)):
            raise ValueError("Server IDs must be unique")

        self.servers = servers
        self.vms = vms
        self.horizon = horizon
        self.seed = seed
        self.overload_penalty = overload_penalty
        self.migration_penalty = migration_penalty
        self.rng = random.Random(seed)
        self.baseline_demands = {
            vm.vm_id: (vm.cpu, vm.memory)
            for vm in vms
        }
        self.placement: dict[int, int] = {}
        self.workload_levels: dict[int, str] = {}
        self.time = 0
        self._server_ids = set(server_ids)

    def reset(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Reset to round-robin placement with every VM at medium workload."""
        self.rng = random.Random(self.seed)
        self.time = 0
        self.workload_levels = {
            vm.vm_id: "medium"
            for vm in self.vms
        }
        self.placement = {}

        for index, vm in enumerate(self.vms):
            server = self.servers[index % len(self.servers)]
            self.placement[vm.vm_id] = server.server_id
            baseline_cpu, baseline_memory = self.baseline_demands[vm.vm_id]
            apply_workload_level(
                vm,
                baseline_cpu,
                baseline_memory,
                "medium",
            )

        return self.get_state()

    def get_state(self) -> tuple[tuple[int, ...], tuple[str, ...]]:
        """Return immutable placement and workload tuples ordered by VM list."""
        if len(self.placement) != len(self.vms):
            raise RuntimeError("Environment must be reset before use")

        placements = tuple(
            self.placement[vm.vm_id]
            for vm in self.vms
        )
        workloads = tuple(
            self.workload_levels[vm.vm_id]
            for vm in self.vms
        )
        return placements, workloads

    def step(
        self,
        action: tuple[int, int] | None,
    ) -> tuple[
        tuple[tuple[int, ...], tuple[str, ...]],
        float,
        bool,
        dict[str, float | bool],
    ]:
        """Apply an action, charge current costs, then sample next workloads."""
        if len(self.placement) != len(self.vms):
            raise RuntimeError("Environment must be reset before use")
        if self.time >= self.horizon:
            raise RuntimeError("Episode has already terminated")

        migrated = self._apply_action(action)
        power, overload_cost = self._operating_costs()
        migration_cost = self.migration_penalty if migrated else 0.0
        reward = -(power + overload_cost + migration_cost)

        for vm in self.vms:
            vm_id = vm.vm_id
            baseline_cpu, baseline_memory = self.baseline_demands[vm_id]
            self.workload_levels[vm_id] = transition_vm_workload(
                vm,
                self.workload_levels[vm_id],
                baseline_cpu,
                baseline_memory,
                self.rng,
            )

        self.time += 1
        done = self.time >= self.horizon
        info: dict[str, float | bool] = {
            "power": power,
            "overload_cost": overload_cost,
            "migration_cost": migration_cost,
            "migrated": migrated,
        }
        return self.get_state(), reward, done, info

    def _apply_action(self, action: tuple[int, int] | None) -> bool:
        if action is None:
            return False
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("action must be None or a (vm_id, server_id) tuple")

        vm_id, server_id = action
        if vm_id not in self.placement:
            raise ValueError(f"Unknown VM ID: {vm_id!r}")
        if server_id not in self._server_ids:
            raise ValueError(f"Unknown server ID: {server_id!r}")
        if self.placement[vm_id] == server_id:
            raise ValueError("Migration target must differ from the current server")

        self.placement[vm_id] = server_id
        return True

    def _operating_costs(self) -> tuple[float, float]:
        cpu_load, mem_load = _build_loads(
            self.placement,
            self.servers,
            self.vms,
        )
        power = 0.0
        overload = 0.0
        for server in self.servers:
            server_id = server.server_id
            cpu = cpu_load[server_id]
            memory = mem_load[server_id]
            if cpu > 0.0 or memory > 0.0:
                power += server.power_at(cpu)
            overload += max(0.0, cpu - server.CPU_CAP)
            overload += max(0.0, memory - server.MEM_CAP)

        return power, self.overload_penalty * overload
