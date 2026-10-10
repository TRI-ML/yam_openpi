import asyncio
import contextlib
import http
import logging
import signal
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        name: str | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._name = name or "policy"
        self._stop: asyncio.Event | None = None
        ws_logger = logging.getLogger("websockets.server")
        ws_logger.setLevel(logging.INFO)
        # yam_eval's VibeManager health-checks the port by opening a bare TCP connection and closing it; websockets
        # logs every such probe as an "opening handshake failed" ERROR with an EOFError traceback. Drop those.
        ws_logger.addFilter(_QuietProbeFilter())

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        # Graceful shutdown: yam_eval restarts serve_policy on every policy/variant switch with SIGTERM; exit 0 instead
        # of dying mid-request with a traceback.
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError):  # non-main thread / no signal support
                loop.add_signal_handler(sig, self._request_stop, sig.name)
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ):
            logger.info(
                "READY: %s serving on %s:%s (action_horizon=%s, action_dim=%s); health check GET /healthz",
                self._name,
                self._host,
                self._port,
                self._metadata.get("action_horizon", "?"),
                self._metadata.get("action_dim", "?"),
            )
            await self._stop.wait()
        logger.info("%s server on port %s stopped cleanly", self._name, self._port)

    def _request_stop(self, signame: str) -> None:
        logger.info("received %s: shutting down the policy server", signame)
        if self._stop is not None:
            self._stop.set()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                # The openpi-client WebsocketClientPolicy wraps the payload as
                # {"method": "infer", "obs": {...}} (+ optional "noise"); older
                # servers expected the bare obs dict. Unwrap for compatibility so
                # both the eval client and deployment/openpi_bridge.py work.
                # RTC (real-time chunking) rides along as sibling keys: "rtc_shift" (actions executed
                # since the last query, for prefix time-alignment) + "rtc_reset" (start-of-episode).
                infer_kwargs = {}
                if isinstance(obs, dict) and obs.get("method") == "infer" and "obs" in obs:
                    _noise = obs.get("noise")
                    if obs.get("rtc_shift") is not None:
                        infer_kwargs["rtc_shift"] = int(obs["rtc_shift"])
                    if obs.get("rtc_reset"):
                        infer_kwargs["rtc_reset"] = True
                    obs = obs["obs"]
                    if _noise is not None:
                        obs["noise"] = _noise

                infer_time = time.monotonic()
                action = self._policy.infer(obs, **infer_kwargs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


class _QuietProbeFilter(logging.Filter):
    """Drop websockets' handshake-failure records caused by bare TCP liveness probes (connect + close, no HTTP)."""

    _NOISE = (
        "connection closed while reading HTTP request line",
        "stream ends after 0 bytes",
        "opening handshake failed",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        exc = record.exc_info[1] if record.exc_info and len(record.exc_info) > 1 else None
        noisy = (
            isinstance(exc, EOFError) or any(n in text for n in self._NOISE) or any(n in str(exc) for n in self._NOISE)
        )
        return not noisy


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
