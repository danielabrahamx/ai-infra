"""Controller loop: read queue, decide, tell provider."""
import time
from .policy import Policy, State, decide


class Controller:
    def __init__(self, queue, provider, policy: Policy, clock=time.monotonic):
        self.queue, self.provider, self.policy, self.clock = queue, provider, policy, clock
        self.state = State()

    def step(self) -> int:
        """One reconcile tick. Returns the replica count after the tick."""
        pending, in_flight = self.queue.depth()
        cur = self.provider.current_replicas()
        want = decide(self.clock(), pending, in_flight, cur, self.policy, self.state)
        if want != cur:
            self.provider.set_replicas(want)
        return want

    def run(self, interval_s: float = 5.0, sleep=time.sleep):
        while True:
            self.step()
            sleep(interval_s)
