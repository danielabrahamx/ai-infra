"""Optional OpenTelemetry. If the package is missing, spans are no-ops."""
from contextlib import contextmanager

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer("llm-gateway")
    AVAILABLE = True
except ImportError:  # optional dependency
    _tracer = None
    AVAILABLE = False


class _Noop:
    def set_attribute(self, *_a, **_k):
        pass


@contextmanager
def span(name: str, **attrs):
    """Yield a span-like object with set_attribute(); no-op without OTel."""
    if _tracer is None:
        yield _Noop()
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s
