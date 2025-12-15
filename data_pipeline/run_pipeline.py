"""Master orchestrator for the CBSA data pipeline."""
import argparse
import importlib

from data_pipeline.utils import logging_utils

STAGES = {
    "fetch": "data_pipeline.fetch_latest_fred_data",
    "process": "data_pipeline.process_latest_fred_data",
    "aggregates": "data_pipeline.update_aggregates",
}


def invoke(stage, **kwargs):
    """Invoke a pipeline stage dynamically.

    Args:
        stage (str): The stage name to invoke.
        **kwargs: Arguments to pass to the stage's main function.

    Returns:
        int: The exit code from the stage.

    Raises:
        AttributeError: If the stage module is missing a main() entry point.
    """
    module_name = STAGES[stage]
    module = importlib.import_module(module_name)

    if hasattr(module, "main"):
        return module.main(**kwargs)  # type: ignore[misc]
    raise AttributeError(f"Stage {stage} missing main() entry point")


def run_pipeline(selected_stages, metrics=None):
    """Run a sequence of pipeline stages.

    Args:
        selected_stages (list[str]): List of stages to run.
        metrics (list[str] | None): Optional list of metrics to filter by.

    Returns:
        int: The exit code (0 for success, non-zero for failure).
    """
    logger = logging_utils.get_logger("run_pipeline")
    for stage in selected_stages:
        logger.info("Running stage: %s", stage)
        kwargs = {}
        if stage == "aggregates":
            kwargs = {"rebuild": False, "metrics": metrics}
        if stage == "process" and metrics:
            kwargs = {"selected_metrics": metrics}
        result = invoke(stage, **kwargs)
        if result != 0:
            logger.error("Stage %s failed with exit code %s", stage, result)
            return result
    logger.info("pipeline run was successful")
    return 0


def main(stages=None, metrics=None):
    """Main entry point for the pipeline orchestrator.

    Args:
        stages (list[str] | None): Optional list of stages to run.
        metrics (list[str] | None): Optional list of metrics to process.

    Returns:
        int: The exit code.

    Raises:
        ValueError: If an unknown stage is requested.
    """
    if stages is None:
        stages = list(STAGES.keys())
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
    return run_pipeline(stages, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CBSA data pipeline")
    parser.add_argument(
        "--stages",
        nargs="*",
        default=None,
        help="Optional subset of stages to run (fetch, process, aggregates)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Optional subset of metrics to process/update",
    )
    args = parser.parse_args()
    raise SystemExit(main(args.stages, args.metrics))

