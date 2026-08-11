from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from geco_cf.pilot import PilotConfig, run_pilot, summarize
from geco_cf.results import write_results


def load_factory(path: str):
    module_name, separator, function_name = path.partition(":")
    if not separator:
        raise ValueError("factory must use package.module:function syntax")
    return getattr(importlib.import_module(module_name), function_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GECo-CF community pilot")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    raw_config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot_config = PilotConfig(
        **{key: raw_config[key] for key in PilotConfig.__dataclass_fields__ if key in raw_config}
    )
    dataset, predictor = load_factory(args.factory)(raw_config)
    results = run_pilot(dataset, predictor, pilot_config, raw_config.get("max_graphs"))
    summary = summarize(results, pilot_config)
    write_results(args.output_dir, results, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
