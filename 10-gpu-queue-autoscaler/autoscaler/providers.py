"""Provider interface: how replicas are read and changed."""
from abc import ABC, abstractmethod


class Provider(ABC):
    @abstractmethod
    def current_replicas(self) -> int: ...

    @abstractmethod
    def set_replicas(self, n: int) -> None: ...


class FakeProvider(Provider):
    """In-memory provider for tests and the demo. Records every call."""

    def __init__(self, replicas: int = 0):
        self.replicas = replicas
        self.calls: list[int] = []

    def current_replicas(self) -> int:
        return self.replicas

    def set_replicas(self, n: int) -> None:
        self.calls.append(n)
        self.replicas = n


class KubernetesDeploymentProvider(Provider):
    """Scales a Deployment through the Kubernetes API.

    STUB: written from the client docs, UNTESTED against a cluster.
    Needs `pip install kubernetes` and RBAC for deployments/scale.
    """

    def __init__(self, name: str, namespace: str = "default"):
        from kubernetes import client, config  # optional dependency
        try:
            config.load_incluster_config()
        except Exception:
            config.load_kube_config()
        self.api, self.name, self.ns = client.AppsV1Api(), name, namespace

    def current_replicas(self) -> int:
        return self.api.read_namespaced_deployment_scale(self.name, self.ns).spec.replicas or 0

    def set_replicas(self, n: int) -> None:
        self.api.patch_namespaced_deployment_scale(self.name, self.ns, {"spec": {"replicas": n}})


class AwsSpotProvider(Provider):
    """UNIMPLEMENTED and UNTESTED. Placeholder for an EC2 Spot fleet scaler.

    Ideas for a real version: EC2 Fleet / Auto Scaling group with a spot
    allocation strategy, handle 2-minute interruption notices, drain first.
    """

    def __init__(self, *a, **k):
        pass

    def current_replicas(self) -> int:
        raise NotImplementedError("AwsSpotProvider is not implemented")

    def set_replicas(self, n: int) -> None:
        raise NotImplementedError("AwsSpotProvider is not implemented")
