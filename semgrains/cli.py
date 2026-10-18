import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import pandas as pd
from .config import Config
from .io import image_paths, load_image
from .pipeline import segment_all, METHODS
from .export import export_result, safe_stem


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semgrains: пакетная сегментация SEM-снимков")
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", default="output/batch")
    batch.add_argument("--config")
    batch.add_argument("--method", choices=["all", *METHODS], default="all")
    batch.add_argument("--scales")
    args = parser.parse_args(argv)
    try:
        config = Config.load(args.config) if args.config else Config()
        paths = image_paths(args.input)
        if not paths:
            raise ValueError("В папке нет поддерживаемых изображений")
        if len(paths) > 1 and config.unit != "px" and not args.scales:
            raise ValueError("Для пакета масштаб каждого файла задаётся через --scales")
        scales = json.loads(Path(args.scales).read_text(encoding="utf-8")) if args.scales else {}
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        config.save(output / "config.json")
        rows, failed = [], 0
        for path in paths:
            try:
                current = replace(config, **scales[path.name]) if path.name in scales else config
                if args.scales and path.name not in scales:
                    current = replace(current, scale_length_px=None, scale_length_value=None)
                results = segment_all(load_image(path), current)
                for method, result in results.items():
                    if args.method != "all" and args.method != method:
                        continue
                    metadata = export_result(result, output / safe_stem(path.name) / method)
                    rows.append({"source": path.name, "method": method,
                                 "count_included": metadata["count_included"], "seconds": result.seconds, "error": ""})
                print(f"OK {path.name}", flush=True)
            except Exception as exc:
                failed += 1
                rows.append({"source": path.name, "error": str(exc)})
                print(f"ERROR {path.name}: {exc}", file=sys.stderr)
        pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
        return 1 if failed else 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
