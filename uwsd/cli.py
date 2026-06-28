"""Command-line interface for the uwsd benchmark platform.

Examples
--------
    uwsd list-methods
    uwsd run --method mfs --output results/mfs.json
    uwsd run --method bert --model all-MiniLM-L6-v2 --output results/bert.json
    uwsd report results/*.json --format markdown
    uwsd compare results/bert.json results/mfs.json
    uwsd predict --method hashing \
        --sentence "I deposited cash at the bank ." --target bank \
        --senses "bank_(financial)=financial institution" "bank_(geography)=river side"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _print_summary(manifest: dict) -> None:
    m = manifest["metrics"]
    method = manifest["method"]
    label = method.get("name")
    if method.get("encoder"):
        label += f" [{method['encoder']}]"
    ci = m["accuracy_ci95"]
    print(f"\n=== {label} on {manifest['dataset']['name']} ===")
    print(f"  instances : {m['n']:,}")
    print(f"  hits      : {m['hits']:,}")
    print(f"  accuracy  : {m['micro_accuracy']*100:.2f}%  "
          f"(95% CI [{ci[0]*100:.2f}, {ci[1]*100:.2f}])")
    print(f"  macro-F1  : {m['macro_f1_over_words']*100:.2f}")
    print(f"  weighted-F1: {m['weighted_f1_over_words']*100:.2f}")
    commit = manifest["environment"].get("git_commit")
    print(f"  traceable : commit={commit}, ts={manifest['timestamp_utc']}")


def cmd_list_methods(args) -> int:
    from .methods import list_methods

    print("Available methods:")
    for m in list_methods():
        print(f"  {m['name']:<12} {m['description']}")
    return 0


def cmd_run(args) -> int:
    from .data import load_coarsewsd20
    from .evaluate import evaluate
    from .methods import get_method

    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    method = get_method(args.method, **kwargs)
    dataset = load_coarsewsd20(root=args.data_root, words=args.words)
    print(f"Loaded {dataset.name}: {len(dataset)} words, "
          f"{dataset.num_test_instances:,} test instances.")
    print(f"Running method '{args.method}'...")
    manifest = evaluate(
        method, dataset,
        limit=args.limit,
        keep_predictions=args.keep_predictions,
        progress=not args.quiet,
    )
    if not args.keep_correct:
        manifest.pop("_correct_vector", None)
    _print_summary(manifest)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nWrote manifest -> {out}")
    return 0


def cmd_report(args) -> int:
    from .report import load_manifests, to_latex, to_markdown

    manifests = load_manifests(args.inputs)
    if args.format == "latex":
        text = to_latex(manifests, caption=args.caption or "")
    else:
        text = to_markdown(manifests)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.format} table -> {args.output}")
    else:
        print(text)
    return 0


def cmd_compare(args) -> int:
    from .metrics import mcnemar_test, paired_bootstrap_test

    def load_vec(path):
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if "_correct_vector" not in m:
            raise SystemExit(
                f"{path} has no _correct_vector; re-run with --keep-correct."
            )
        return m["_correct_vector"], m["method"].get("name", path)

    a, na = load_vec(args.a)
    b, nb = load_vec(args.b)
    boot = paired_bootstrap_test(a, b)
    mc = mcnemar_test(a, b)
    print(f"Comparing A={na} vs B={nb} on {len(a):,} paired instances")
    print(f"  delta accuracy (A-B): {boot['delta_accuracy']*100:+.2f} pts "
          f"(95% CI [{boot['ci_low']*100:+.2f}, {boot['ci_high']*100:+.2f}])")
    print(f"  paired bootstrap p-value: {boot['p_value']:.4f}")
    print(f"  McNemar exact p-value   : {mc['p_value']:.4f} "
          f"(discordant={mc['discordant']})")
    return 0


def cmd_predict(args) -> int:
    from .data import Instance, WordTask
    from .methods import get_method

    classes = {}
    for i, spec in enumerate(args.senses):
        if "=" in spec:
            key, label = spec.split("=", 1)
        else:
            key, label = f"sense_{i}", spec
        classes[i] = label.strip()
    tokens = args.sentence.split()
    if args.target_index is not None:
        idx = args.target_index
        target = tokens[idx]
    else:
        target = args.target
        idx = next((j for j, t in enumerate(tokens) if t.lower() == target.lower()), 0)
    word = target
    task = WordTask(word=word, classes=classes)
    inst = Instance(word=word, token_index=idx, sentence=args.sentence)
    kwargs = {}
    if args.model:
        kwargs["model"] = args.model
    method = get_method(args.method, **kwargs)
    pred = method.predict(inst, task)
    print(f"Sentence : {args.sentence}")
    print(f"Target   : '{target}' (token {idx})")
    print(f"Predicted: {pred.label}  (confidence {pred.confidence:.1%})")
    if pred.low_confidence:
        print("  ⚠ low confidence: candidate senses are close; treat with caution.")
    print(f"Why      : {pred.explanation}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uwsd", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"uwsd {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("list-methods", help="List registered WSD methods.")
    s.set_defaults(func=cmd_list_methods)

    s = sub.add_parser("run", help="Run a method on a dataset and write a manifest.")
    s.add_argument("--method", required=True)
    s.add_argument("--model", default=None, help="Encoder/model name (method-specific).")
    s.add_argument("--data-root", default=None, help="Path to CoarseWSD-20 dir.")
    s.add_argument("--words", nargs="*", default=None, help="Subset of words.")
    s.add_argument("--limit", type=int, default=None,
                   help="Max test instances per word (for quick runs).")
    s.add_argument("--output", default=None, help="Where to write the JSON manifest.")
    s.add_argument("--keep-predictions", action="store_true",
                   help="Store per-instance predictions + explanations in manifest.")
    s.add_argument("--keep-correct", action="store_true",
                   help="Keep the per-instance correctness vector (for `compare`).")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("report", help="Build a markdown/LaTeX table from manifests.")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    s.add_argument("--caption", default=None)
    s.add_argument("--output", default=None)
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("compare", help="Significance test between two manifests.")
    s.add_argument("a")
    s.add_argument("b")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("predict", help="Disambiguate a single sentence (interactive).")
    s.add_argument("--method", default="hashing")
    s.add_argument("--model", default=None)
    s.add_argument("--sentence", required=True)
    s.add_argument("--target", default=None, help="The ambiguous word.")
    s.add_argument("--target-index", type=int, default=None,
                   help="Token index of the ambiguous word (overrides --target).")
    s.add_argument("--senses", nargs="+", required=True,
                   help="Candidate senses as 'key=phrase' or just 'phrase'.")
    s.set_defaults(func=cmd_predict)

    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
