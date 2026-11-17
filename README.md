# Dynamic Data Center Resource Allocation Under Uncertain Workloads

This project will study sequential virtual machine (VM) placement in a data
center under uncertain workloads. The eventual model will treat placement and
migration decisions as a Markov decision process (MDP).

## Structure

- `simulator.py`: Existing VM/server models, workload generation, capacity
  checks, power calculations, objective calculation, and summary statistics.
- `environment.py`: Sequential datacenter environment, stochastic workload
  transitions, migration actions, and operating costs.
- `policies.py`: No-migration and immediate-cost greedy migration baselines.
- `experiments.py`: Episode runner, paired policy comparisons, metrics, and the
  cumulative-cost plot.
- `main.py`: Small smoke-test example using the sequential environment.
- `requirements.txt`: Current Python dependencies.

## Current status

The static simulator utilities reused from `cs361-project` and a finite-horizon
sequential environment are implemented. Environment states contain immutable
VM placement and workload tuples. At each step, the environment applies an
optional migration, charges power, overload, and migration costs, then samples
the next low/medium/high workload for every VM.

The overload and migration weights are configurable and default to `1000.0`
per unit of capacity exceedance and `10.0` per migration. Calling `reset()`
restarts the random stream from the environment's original seed. No-migration
and greedy single-migration baselines are implemented, along with a paired
multi-seed experiment. Learning and planning algorithms remain planned.
Because an environment updates its VM objects in place, separate environment
instances should be constructed with separate VM instances.

Run the current example with:

```bash
python3 main.py
```

Run the unit tests with:

```bash
python3 -m unittest discover -s tests
```

Run the 20-seed baseline comparison and generate its plot with:

```bash
python3 experiments.py
```
