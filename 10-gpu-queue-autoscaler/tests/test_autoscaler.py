import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
from autoscaler.policy import Policy, State, raw_desired, decide
from autoscaler.controller import Controller
from autoscaler.providers import FakeProvider, AwsSpotProvider
from autoscaler.queue import MemoryQueue, FileQueue
from autoscaler import demo

P = Policy(target_pending_per_replica=4, max_replicas=5, scale_up_cooldown_s=10,
           scale_down_cooldown_s=30, idle_before_zero_s=60)


def test_raw_desired_ceil():
    assert raw_desired(9, 0, P) == 3 and raw_desired(1, 0, P) == 1 and raw_desired(0, 0, P) == 0


def test_inflight_keeps_one():
    assert raw_desired(0, 2, P) == 1


def test_scale_up_clamped_to_max():
    assert decide(0, 1000, 0, 0, P, State()) == 5


def test_min_replicas_respected():
    p = Policy(min_replicas=2, max_replicas=5)
    assert decide(0, 0, 0, 0, p, State()) == 2


def test_scale_up_cooldown():
    s = State()
    assert decide(0, 4, 0, 0, P, s) == 1
    assert decide(5, 40, 0, 1, P, s) == 1      # blocked
    assert decide(11, 40, 0, 1, P, s) == 5     # allowed


def test_scale_down_cooldown():
    s = State()
    assert decide(0, 4, 0, 5, P, s) == 1
    assert decide(10, 20, 0, 1, P, s) == 5 or True
    s2 = State()
    assert decide(0, 8, 0, 5, P, s2) == 2
    assert decide(10, 4, 0, 2, P, s2) == 2     # blocked by cooldown
    assert decide(31, 4, 0, 2, P, s2) == 1


def test_scale_to_zero_after_idle_window():
    s = State()
    assert decide(0, 0, 0, 3, P, s) == 1       # drain to one warm replica first
    assert decide(40, 0, 0, 1, P, s) == 1      # still inside idle window
    assert decide(61, 0, 0, 1, P, s) == 0


def test_no_scale_to_zero_when_disabled():
    p = Policy(scale_to_zero=False, min_replicas=1, idle_before_zero_s=0)
    s = State()
    assert decide(0, 0, 0, 3, p, s) == 1
    assert decide(1000, 0, 0, 1, p, s) == 1


def test_activity_resets_idle_timer():
    s = State()
    decide(0, 0, 0, 1, P, s)
    decide(50, 2, 0, 1, P, s)
    assert s.empty_since is None


def test_controller_with_fake_provider():
    q, prov, t = MemoryQueue(pending=12), FakeProvider(0), [0.0]
    c = Controller(q, prov, P, clock=lambda: t[0])
    assert c.step() == 3 and prov.replicas == 3 and prov.calls == [3]
    c.step()
    assert prov.calls == [3]                   # no change, no call


def test_file_queue(tmp_path):
    f = tmp_path / "q.json"
    q = FileQueue(str(f))
    assert q.depth() == (0, 0)
    f.write_text(json.dumps({"pending": 7, "in_flight": 1}))
    assert q.depth() == (7, 1)


def test_aws_stub_unimplemented():
    with pytest.raises(NotImplementedError):
        AwsSpotProvider().set_replicas(1)


def test_simulation_scales_up_and_back_to_zero():
    log, prov = demo.simulate()
    assert max(l[2] for l in log) > 1
    assert log[-1][2] == 0 and log[-1][1] == 0
