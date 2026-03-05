"""Unit tests for observability / tracing setup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from opentelemetry import trace

import pdf_extractor.infrastructure.observability as obs


@pytest.fixture(autouse=True)
def reset_tracer():
    """Reset the module-level tracer singleton before each test."""
    original = obs._tracer
    obs._tracer = None
    yield
    obs._tracer = original


class TestSetupTracing:
    def test_disabled_returns_noop_tracer(self) -> None:
        tracer = obs.setup_tracing(enabled=False)
        assert tracer is not None

    def test_returns_cached_tracer_on_second_call(self) -> None:
        t1 = obs.setup_tracing(enabled=False)
        t2 = obs.setup_tracing(enabled=False)
        assert t1 is t2

    def test_enabled_with_otlp_exporter(self) -> None:
        _otlp_path = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        with patch(_otlp_path) as mock_exp_cls, patch(
            "pdf_extractor.infrastructure.observability.TracerProvider"
        ) as mock_provider_cls, patch(
            "pdf_extractor.infrastructure.observability.trace.set_tracer_provider"
        ):
            mock_provider = MagicMock()
            mock_provider_cls.return_value = mock_provider
            mock_exp_cls.return_value = MagicMock()

            tracer = obs.setup_tracing(
                service_name="test-svc",
                endpoint="http://localhost:4317",
                enabled=True,
            )
            assert tracer is not None
            mock_provider.add_span_processor.assert_called_once()

    def test_exporter_failure_falls_back_gracefully(self) -> None:
        _otlp_path = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
        with patch(_otlp_path, side_effect=Exception("connection refused")):
            # Should not raise; falls back to no-op
            tracer = obs.setup_tracing(enabled=True, endpoint="http://bad:9999")
            assert tracer is not None


class TestGetTracer:
    def test_get_tracer_returns_tracer(self) -> None:
        t = obs.get_tracer()
        assert t is not None

    def test_get_tracer_initialises_if_none(self) -> None:
        obs._tracer = None
        t = obs.get_tracer()
        assert t is not None
        assert obs._tracer is not None
