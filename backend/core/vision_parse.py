from __future__ import annotations
import base64
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as a standalone script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.config import GROQ_API_KEY, RAW_DOCS_DIR

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vision_parse")

# Configuration
ASSETS_DIR = RAW_DOCS_DIR / "assets"
_DIAGRAM_ALLOWLIST: set[str] = {
    "action_state_transition.png",
    "handling_multiple_actions.png",
    "information_flow_VDA5050.png",
    "map_distribution_process.png",
    "process_cancel_order.png",
    "process_order_update.png",
    "request_coordinated_replanning_zone_replanning.png",
    "request_release_zone_access.png",
    "request_state_transitions.png",
    "states_during_order_handling.png",
    "order_information_state_topic.png",
    "graph_representation_transmission.png",
    "driving_route_horizon.png",
    "update_order_extension.png",
    "update_order_stitching_node.png",
}

_VISION_PROMPT = """\
You are an expert technical documentation writer analyzing VDA 5050 specification diagrams.

Analyze this diagram image and produce a comprehensive, structured text description. Your output will be embedded into a vector database for retrieval, so include ALL relevant technical details.

Format your response as follows:
State whether this is a sequence diagram, activity diagram, state machine, flowchart, etc.

### Participants / Actors
List all participants, actors, or components shown in the diagram.

### Process Flow (Step by Step)
Describe every step, message, decision point, and transition in the diagram in sequential order. Be extremely detailed. Include:
- All message names and their directions (e.g., "Fleet Control sends /order to Mobile Robot via MQTT Broker")
- All decision points and their conditions
- All states and transitions
- All action types and blocking behaviors

### Key Technical Terms
List all VDA 5050 specific terms, field names, enum values, states, topics, or error codes visible in the diagram.

### Summary
A 2-3 sentence summary of what this diagram shows in the context of VDA 5050 fleet communication.
"""

def _encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def _describe_diagram(image_path: Path) -> str:

    # pyrefly: ignore [missing-import]
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    image_b64 = _encode_image_base64(image_path)
    diagram_name = image_path.stem.replace("_", " ").title()

    prompt = _VISION_PROMPT.format(diagram_name=diagram_name)

    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    return response.choices[0].message.content

# Main
def main() -> None:
    print("\n" + "=" * 70)
    print("  VDA-5050 RAG — Vision Diagram Parser")
    print("=" * 70)
    print(f"  Assets dir : {ASSETS_DIR}")
    print(f"  Diagrams   : {len(_DIAGRAM_ALLOWLIST)}")
    print("=" * 70)

    if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set or invalid. Cannot use vision model.")
        sys.exit(1)

    png_files = sorted(ASSETS_DIR.glob("*.png"))
    diagrams = [f for f in png_files if f.name in _DIAGRAM_ALLOWLIST]

    if not diagrams:
        logger.error("No diagrams found in allowlist at %s", ASSETS_DIR)
        sys.exit(1)

    print(f"\n  Found {len(diagrams)} diagrams to process:\n")

    for idx, diagram_path in enumerate(diagrams, 1):
        output_path = diagram_path.with_suffix(".description.md")

        # Skip if already generated (unless you want to regenerate)
        if output_path.exists():
            print(f"  [{idx}/{len(diagrams)}] SKIP (exists): {diagram_path.name}")
            continue

        print(f"  [{idx}/{len(diagrams)}] Processing: {diagram_path.name} ...")

        try:
            description = _describe_diagram(diagram_path)

            # Save the description
            output_path.write_text(description, encoding="utf-8")
            logger.info("Saved description to %s (%d chars)", output_path.name, len(description))
            print(f"           ✓ Saved: {output_path.name} ({len(description)} chars)")

        except Exception as e:
            logger.error("Failed to process %s: %s", diagram_path.name, e)
            print(f"           ✗ Error: {e}")

        if idx < len(diagrams):
            time.sleep(20)

    print("\n" + "=" * 70)
    print("  VISION PARSING COMPLETE")
    print(f"  Description files saved in: {ASSETS_DIR}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
