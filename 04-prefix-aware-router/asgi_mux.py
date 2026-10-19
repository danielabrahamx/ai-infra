"""httpx transport that dispatches by hostname to in-process ASGI apps.
Lets router + replicas run in one process (tests, demo). No sockets."""
import httpx

class MuxTransport(httpx.AsyncBaseTransport):
    def __init__(self, apps: dict):          # {"replica0": app, ...}
        self.t = {h: httpx.ASGITransport(app=a) for h, a in apps.items()}

    async def handle_async_request(self, request):
        return await self.t[request.url.host].handle_async_request(request)
