"""Retry counters and failure tracking utilities."""

from dataclasses import dataclass

from data_pipeline import pipeline_config as config


@dataclass
class MetricRetryState:
    """Track retries for a specific CBSA within a metric fetch."""

    attempts: int = 0

    def register_attempt(self) -> None:
        self.attempts += 1

    @property
    def exhausted(self) -> bool:
        return self.attempts >= config.FRED_RETRY_LIMIT


@dataclass
class ConsecutiveFailureState:
    """Track consecutive CBSA-level failures to trigger abort."""

    consecutive_failures: int = 0

    def register_success(self) -> None:
        self.consecutive_failures = 0

    def register_failure(self) -> None:
        self.consecutive_failures += 1

    @property
    def reached_threshold(self) -> bool:
        return self.consecutive_failures >= config.CONSECUTIVE_CBSA_FAILURE_LIMIT
