from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from .io import read_jsonl, write_jsonl, inference_call


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and run the CPU CompactFact extractor")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", default="data/train.jsonl")
    train.add_argument("--catalog", default="data/catalog.json")
    train.add_argument("--config", choices=["tiny", "balanced", "quality"], default="balanced")
    train.add_argument("--output", default="models/balanced")
    train.add_argument("--seed", type=int, default=42)
    predict = commands.add_parser("predict")
    predict.add_argument("--model", default="models/balanced")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--validate", action="store_true")
    serve = commands.add_parser("serve")
    serve.add_argument("--model", default="models/balanced")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from .model import FactExtractor
    if args.command == "train":
        data = list(read_jsonl(args.data))
        model = FactExtractor(json.loads(Path(args.catalog).read_text()), args.config, args.seed).fit(data)
        import sklearn, numpy, scipy
        model.metadata.update({"input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                               "catalog_sha256": hashlib.sha256(Path(args.catalog).read_bytes()).hexdigest(),
                               "python": platform.python_version(), "sklearn": sklearn.__version__,
                               "numpy": numpy.__version__, "scipy": scipy.__version__})
        model.save(args.output)
        print(json.dumps(model.metadata, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        model = FactExtractor.load(args.model)
        started = time.perf_counter()
        def predictions():
            from .schemas import ExtractionResponse
            for i, call in enumerate(read_jsonl(args.input), 1):
                pred = model.predict(inference_call(call))
                if args.validate:
                    ExtractionResponse.model_validate(pred)
                if i % 100 == 0:
                    print(f"{i} calls, {time.perf_counter() - started:.1f}s", file=sys.stderr)
                yield pred
        write_jsonl(args.output, predictions())
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(model_dir=args.model), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())