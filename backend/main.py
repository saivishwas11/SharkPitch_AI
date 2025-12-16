"""
Backend entrypoint utilities for running the analysis pipeline programmatically.

This module exposes:
- validate_input_file(file_path): basic file validation
- run_pipeline(input_file): runs the LangGraph-based pipeline and returns the result state
"""

import logging
from pathlib import Path
from typing import Dict, Any

from backend.graph.graph_multiagent import build_graph

logger = logging.getLogger(__name__)


def validate_input_file(file_path: str) -> bool:
    """Validate that the input file exists and is a supported format."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False

    # Check for supported file extensions
    supported_extensions = [".mp4", ".avi", ".mov", ".wav", ".mp3"]
    if path.suffix.lower() not in supported_extensions:
        logger.error(
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: {', '.join(supported_extensions)}"
        )
        return False

    return True


def run_pipeline(input_file: str) -> Dict[str, Any]:
    """Run the entire pipeline with the given input file and return the result state."""
    try:
        # Build the graph
        app = build_graph()
        logger.info("Graph compiled successfully")

        # Prepare initial state
        initial_state = {
            "input_path": str(Path(input_file).resolve()),
        }

        # Run the graph
        logger.info(f"Starting pipeline with input: {input_file}")

        if hasattr(app, "invoke"):
            logger.debug("Using invoke() method to run the graph")
            result = app.invoke(initial_state)
        elif hasattr(app, "__call__"):
            logger.debug("Using __call__ method to run the graph")
            result = app(initial_state)
        elif hasattr(app, "run"):
            logger.debug("Using run() method to run the graph")
            result = app.run(initial_state)
        else:
            raise RuntimeError(
                "Graph execution failed. The compiled graph doesn't support any known execution methods. "
                f"Available methods: {[m for m in dir(app) if not m.startswith('_')]}"
            )

        logger.info("Pipeline completed successfully")
        return result

    except Exception as e:
        logger.exception(f"Pipeline failed: {str(e)}")
        raise


