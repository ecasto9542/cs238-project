"""Baseline policies for sequential VM placement."""

from environment import DatacenterEnv
from simulator import _build_loads


Action = tuple[int, int] | None


def no_migration_policy(env: DatacenterEnv) -> None:
    """Keep the current VM placement."""
    return None


def immediate_placement_cost(
    env: DatacenterEnv,
    placement: dict[int, int],
    migrated: bool,
) -> float:
    """Evaluate immediate cost without mutating or advancing the environment."""
    cpu_load, mem_load = _build_loads(placement, env.servers, env.vms)
    power = 0.0
    overload = 0.0

    for server in env.servers:
        server_id = server.server_id
        cpu = cpu_load[server_id]
        memory = mem_load[server_id]
        if cpu > 0.0 or memory > 0.0:
            power += server.power_at(cpu)
        overload += max(0.0, cpu - server.CPU_CAP)
        overload += max(0.0, memory - server.MEM_CAP)

    overload_cost = env.overload_penalty * overload
    migration_cost = env.migration_penalty if migrated else 0.0
    return power + overload_cost + migration_cost


def greedy_migration_policy(env: DatacenterEnv) -> Action:
    """Choose the single migration with the lowest immediate operating cost."""
    env.get_state()
    best_action: Action = None
    best_cost = immediate_placement_cost(env, env.placement, migrated=False)

    for vm in env.vms:
        current_server_id = env.placement[vm.vm_id]
        for server in env.servers:
            if server.server_id == current_server_id:
                continue

            candidate_placement = env.placement.copy()
            candidate_placement[vm.vm_id] = server.server_id
            candidate_cost = immediate_placement_cost(
                env,
                candidate_placement,
                migrated=True,
            )
            if candidate_cost < best_cost:
                best_action = (vm.vm_id, server.server_id)
                best_cost = candidate_cost

    return best_action
