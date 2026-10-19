"""Prometheus metrics. One private registry so tests can build many apps."""
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

MS_BUCKETS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class Metrics:
    def __init__(self):
        self.registry = r = CollectorRegistry()
        self.requests = Counter("gateway_requests_total", "Requests", ["route", "status"], registry=r)
        self.fallbacks = Counter("gateway_fallbacks_total", "Fallbacks", ["from_route"], registry=r)
        self.ttft = Histogram("gateway_ttft_ms", "Time to first token (ms)", ["route"], buckets=MS_BUCKETS, registry=r)
        self.itl = Histogram("gateway_itl_ms", "Inter-token latency (ms)", ["route"], buckets=MS_BUCKETS, registry=r)
        self.total = Histogram("gateway_request_ms", "Total request time (ms)", ["route"],
                               buckets=MS_BUCKETS + (30000,), registry=r)
        self.score = Histogram("gateway_complexity_score", "Complexity score", buckets=[i / 10 for i in range(11)], registry=r)
        self.slo_viol = Counter("gateway_slo_violations_total", "SLO violations", ["route", "slo"], registry=r)

    def render(self) -> bytes:
        return generate_latest(self.registry)
