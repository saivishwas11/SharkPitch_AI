import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any

from graph.graph_multiagent import build_graph

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('athena_task.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_input_file(file_path: str) -> bool:
    """Validate that the input file exists and is a supported format."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Input file not found: {file_path}")
        return False
    
    # Check for supported file extensions
    supported_extensions = ['.mp4', '.avi', '.mov', '.wav', '.mp3']
    if path.suffix.lower() not in supported_extensions:
        logger.error(f"Unsupported file format: {path.suffix}. Supported formats: {', '.join(supported_extensions)}")
        return False
    
    return True

def run_pipeline(input_file: str) -> Dict[str, Any]:
    """Run the entire pipeline with the given input file."""
    try:
        # Build the graph
        app = build_graph()
        logger.info("Graph compiled successfully")

        # Prepare initial state
        initial_state = {
            "input_path": str(Path(input_file).resolve())
        }

        # Run the graph
        logger.info(f"Starting pipeline with input: {input_file}")
        
        # The compiled graph should be invoked using the invoke() method
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

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_media_file>")
        print("Supported formats: .mp4, .avi, .mov, .wav, .mp3")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Validate input file
    if not validate_input_file(input_file):
        sys.exit(1)
    
    try:
        # Run the pipeline
        result = run_pipeline(input_file)
        
        # Print a summary of the results
        print("\n=== Pipeline Execution Summary ===")
        print(f"Input file: {input_file}")
        
        # Print transcript if available
        if 'transcript' in result:
            print("\nTranscript:")
            print(result['transcript'][:500] + (result['transcript'][500:] and '...'))
        
        # Print analysis results if available
        if 'content_analysis' in result:
            print("\nContent Analysis:")
            print(json.dumps(result['content_analysis'], indent=2, ensure_ascii=False))
        
        # Print shark panel results if available
        if 'shark_panel' in result:
            print("\nShark Panel Feedback:")
            print(json.dumps(result['shark_panel'], indent=2, ensure_ascii=False))
        
        print("\nPipeline execution completed. Check athena_task.log for detailed logs.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"Error: {str(e)}")
        print("Check athena_task.log for more details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
