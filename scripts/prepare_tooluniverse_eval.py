#!/Users/razvan/mamba/envs/tooluni/bin/python
import argparse
import json
import sys
from pathlib import Path

DATASET = Path(__file__).resolve().parents[1] / "48-submissions-clean.json"


def load_items():
    with DATASET.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_item(items, item_id=None, index=None):
    if item_id is not None:
        for item in items:
            if item.get("id") == item_id:
                return item
        raise SystemExit(f"No submission found with id: {item_id}")
    if index is not None:
        if index < 1 or index > len(items):
            raise SystemExit(f"Index out of range: {index}. Valid range is 1..{len(items)}")
        return items[index - 1]
    raise SystemExit("Provide either --id or --index")


def compact_json(value):
    return json.dumps(value, indent=2, ensure_ascii=True)


def build_prompt(item, include_existing=False):
    parts = []
    parts.append("Question:")
    parts.append(item.get("prompt", "").strip())

    criteria = item.get("refinedCriteria") or item.get("criteria") or {}
    if criteria:
        parts.append("")
        parts.append("Criteria:")
        parts.append(compact_json(criteria))

    clarity = item.get("claritySelections") or {}
    if clarity:
        parts.append("")
        parts.append("Clarity selections:")
        parts.append(compact_json(clarity))

    if include_existing and item.get("modelResponse"):
        parts.append("")
        parts.append("Existing non-ToolUniverse response for comparison:")
        parts.append(item["modelResponse"].strip())

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare one evaluation item for the ToolUniverse Router agent."
    )
    parser.add_argument("--index", type=int, help="1-based submission index in 47-submissions-clean.json")
    parser.add_argument("--id", dest="item_id", help="Submission UUID")
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include the existing modelResponse for side-by-side comparison",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all submissions as index, id, and prompt preview",
    )
    args = parser.parse_args()

    items = load_items()

    if args.list:
        for idx, item in enumerate(items, start=1):
            prompt = " ".join(item.get("prompt", "").split())
            preview = prompt[:120] + ("..." if len(prompt) > 120 else "")
            print(f"{idx:02d}\t{item.get('id', '')}\t{preview}")
        return

    item = find_item(items, item_id=args.item_id, index=args.index)
    print(build_prompt(item, include_existing=args.include_existing))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
