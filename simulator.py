import random


class VM:
    def __init__(self, vm_id: int, cpu: float, memory: float):
        self.vm_id = vm_id
        self.cpu = cpu
        self.memory = memory


class Server:
    CPU_CAP = 100.0
    MEM_CAP = 100.0
    IDLE_POWER = 100.0
    MAX_POWER = 250.0

    def __init__(self, server_id: int):
        self.server_id = server_id

    def power_at(self, cpu_used: float) -> float:
        if cpu_used == 0.0:
            return 0.0
        util = min(cpu_used / self.CPU_CAP, 1.0)
        return self.IDLE_POWER + (self.MAX_POWER - self.IDLE_POWER) * util


PENALTY = 1000.0


def generate_problem(
    n_servers: int = 10,
    n_vms: int = 20,
    seed: int | None = None,
    workload_type: str = "mixed",
) -> tuple[list[Server], list[VM]]:
    """Generate a VM placement problem instance.

    workload_type controls the VM demand distribution:
      "mixed"       - alternating CPU-heavy and memory-heavy VMs (default)
      "balanced"    - CPU and memory both uniform in [20, 40]
      "cpu_heavy"   - CPU uniform in [40, 65], memory uniform in [5, 25]
      "memory_heavy"- CPU uniform in [5, 25], memory uniform in [40, 65]
    """
    if workload_type not in ("mixed", "balanced", "cpu_heavy", "memory_heavy"):
        raise ValueError(f"Unknown workload_type: {workload_type!r}")

    rng = random.Random(seed)
    servers = [Server(i) for i in range(n_servers)]
    vms = []
    for i in range(n_vms):
        if workload_type == "mixed":
            if i % 2 == 0:
                cpu = rng.uniform(40, 65)
                mem = rng.uniform(5, 25)
            else:
                cpu = rng.uniform(5, 25)
                mem = rng.uniform(40, 65)
        elif workload_type == "balanced":
            cpu = rng.uniform(20, 40)
            mem = rng.uniform(20, 40)
        elif workload_type == "cpu_heavy":
            cpu = rng.uniform(40, 65)
            mem = rng.uniform(5, 25)
        else:  # memory_heavy
            cpu = rng.uniform(5, 25)
            mem = rng.uniform(40, 65)
        vms.append(VM(i, cpu, mem))
    return servers, vms


def _build_loads(
    placement: dict[int, int],
    servers: list[Server],
    vms: list[VM],
) -> tuple[dict[int, float], dict[int, float]]:
    cpu_load: dict[int, float] = {s.server_id: 0.0 for s in servers}
    mem_load: dict[int, float] = {s.server_id: 0.0 for s in servers}
    for vm_id, srv_id in placement.items():
        cpu_load[srv_id] += vms[vm_id].cpu
        mem_load[srv_id] += vms[vm_id].memory
    return cpu_load, mem_load


def compute_objective(
    placement: dict[int, int],
    servers: list[Server],
    vms: list[VM],
) -> float:
    cpu_load, mem_load = _build_loads(placement, servers, vms)
    total = 0.0
    for s in servers:
        c = cpu_load[s.server_id]
        m = mem_load[s.server_id]
        if c > 0 or m > 0:
            total += s.power_at(c)
        total += PENALTY * max(0.0, c - s.CPU_CAP)
        total += PENALTY * max(0.0, m - s.MEM_CAP)
    return total


def is_feasible(
    placement: dict[int, int],
    servers: list[Server],
    vms: list[VM],
) -> bool:
    cpu_load, mem_load = _build_loads(placement, servers, vms)
    return all(
        cpu_load[s.server_id] <= s.CPU_CAP and mem_load[s.server_id] <= s.MEM_CAP
        for s in servers
    )


def compute_stats(
    placement: dict[int, int],
    servers: list[Server],
    vms: list[VM],
) -> dict:
    cpu_load, mem_load = _build_loads(placement, servers, vms)
    active = [
        s for s in servers
        if cpu_load[s.server_id] > 0 or mem_load[s.server_id] > 0
    ]
    total_power = sum(s.power_at(cpu_load[s.server_id]) for s in active)
    avg_cpu_util = (
        sum(cpu_load[s.server_id] / s.CPU_CAP for s in active) / len(active)
        if active else 0.0
    )
    return {
        "active_servers": len(active),
        "total_power": total_power,
        "avg_cpu_util": avg_cpu_util,
        "feasible": is_feasible(placement, servers, vms),
    }
