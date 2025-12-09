import argparse
import json
import warnings

from graph.graph_multiagent import build_graph

warnings.filterwarnings("ignore")


def pretty_print_result(state):
    print("\n=== TRANSCRIPT ===\n")
    print(state.get("transcript", ""))

    print("\n=== VOICE STATS (Voice Agent) ===\n")
    print(json.dumps(state.get("voice_stats", {}), indent=2))

    if state.get("asr_error"):
        print("\n=== ASR ERROR ===\n")
        print(state["asr_error"])

    print("\n=== CONTENT ANALYSIS (Master Agent) ===\n")
    print(json.dumps(state.get("content_analysis", {}), indent=2))

    print("\n=== SHARK PANEL ===\n")
    print(json.dumps(state.get("shark_panel", {}), indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: voice -> Groq ASR -> Gemini master content agent -> sharks"
    )
    parser.add_argument("path", help="Path to audio/video file containing the pitch")
    args = parser.parse_args()

    app = build_graph()
    # Groq ASR by default (no backend override). Gemini used only for content.
    initial_state = {"input_path": args.path}
    final_state = app.invoke(initial_state)

    pretty_print_result(final_state)


if __name__ == "__main__":
    main()
