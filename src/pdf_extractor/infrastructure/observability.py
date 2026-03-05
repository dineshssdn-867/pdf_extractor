"""OpenTelemetry tracing setup via Arize Phoenix."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

logger = logging.getLogger(__name__)

_tracer: trace.Tracer | None = None


def setup_tracing(
    service_name: str = "pdf-extractor",
    endpoint: str = "http://localhost:4317",
    enabled: bool = True,
) -> trace.Tracer:
    """Configure OTel tracing and return a named tracer.

    Uses Arize Phoenix OTLP gRPC exporter when enabled.
    Falls back to a no-op tracer when disabled or the exporter is unavailable.
    """
    global _tracer

    if _tracer is not None:
        return _tracer

    if not enabled:
        _tracer = trace.get_tracer(service_name)
        return _tracer

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("OTel tracing enabled → %s", endpoint)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to configure OTel exporter (%s); using no-op tracer.", exc)

    _tracer = trace.get_tracer(service_name)
    return _tracer


def get_tracer() -> trace.Tracer:
    """Return the current tracer (no-op if setup_tracing() was never called)."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("pdf-extractor")
    return _tracer
