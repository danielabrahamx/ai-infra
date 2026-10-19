# 10 - GPU queue autoscaler

Roadmap stage 10 of 12 (AI Infrastructure Engineering). A controller that scales GPU replicas from request queue depth.

## Why
GPUs are costly. CPU load is a poor signal for LLM serving. Queue depth reacts earlier and supports scale-to-zero.

## How it works
- `policy.py` - pure function: `desired = ceil(pending / target_per_replica)`, clamped to min/max. Separate scale-up and scale-down cooldowns. When the queue is empty, it drains to one warm replica, then goes to `min_replicas` (0) after an idle window. In-flight work keeps one replica alive.
- `queue.py` - `MemoryQueue` and `FileQueue` (default), `RedisQueue` (optional).
- `providers.py` - `Provider` interface. `FakeProvider` (tests), `KubernetesDeploymentProvider` (stub), `AwsSpotProvider` (unimplemented).
- `controller.py` - reconcile loop with an injectable clock.
- `demo.py` - simulated load.
- `k8s/keda-scaledobject.yaml` - KEDA equivalent using the Redis list trigger.

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m autoscaler.demo
```

```python
from autoscaler.controller import Controller
from autoscaler.policy import Policy
from autoscaler.providers import KubernetesDeploymentProvider
from autoscaler.queue import RedisQueue
Controller(RedisQueue(), KubernetesDeploymentProvider("gpu-inference"), Policy()).run(5)
```

## Status
**Verified (CPU, simulated)**
- Unit tests: `python -m pytest -q` -> 13 passed.
- `python -m autoscaler.demo` runs a SIMULATED scenario: synthetic arrivals, fake clock, fake provider, simple per-replica service rate and startup delay. Demo result: peak 6 replicas, scale calls [2, 6, 5, 1, 0], back to 0 after the burst. Output shows the logic only. It is not a measurement of any real system.

**NOT tested**
- `RedisQueue` against a live Redis.
- `KubernetesDeploymentProvider` against a cluster (stub).
- `AwsSpotProvider`: unimplemented. Raises `NotImplementedError`.
- `k8s/keda-scaledobject.yaml`: never applied to a cluster.
- No real GPU cold-start times are modelled.

## License
MIT, Daniel Abraham.
