"""Turn one or more result manifests into publication-ready tables.

Supports Markdown (for the README / GitHub) and LaTeX booktabs (for papers).
Numbers are read straight from the manifests produced by :mod:`uwsd.evaluate`,
so a table can never disagree with the run that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _row_from_manifest(m: dict) -> dict:
    method = m.get("method", {})
    name = method.get("name", "?")
    enc = method.get("encoder")
    label = f"{name} ({enc})" if enc else name
    met = m["metrics"]
    ci = met.get("accuracy_ci95", [None, None])
    return {
        "method": label,
        "hits": met.get("hits"),
        "n": met.get("n"),
        "accuracy": met.get("micro_accuracy"),
        "macro_f1": met.get("macro_f1_over_words"),
        "weighted_f1": met.get("weighted_f1_over_words"),
        "ci_low": ci[0],
        "ci_high": ci[1],
    }


def load_manifests(paths: List[str]) -> List[dict]:
    out = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def to_markdown(manifests: List[dict]) -> str:
    rows = [_row_from_manifest(m) for m in manifests]
    rows.sort(key=lambda r: (r["accuracy"] is None, -(r["accuracy"] or 0)))
    lines = [
        "| Method | Hits | Accuracy | 95% CI | Macro-F1 | Weighted-F1 |",
        "| ------ | ---- | -------- | ------ | -------- | ----------- |",
    ]
    for r in rows:
        ci = (
            f"[{r['ci_low']*100:.2f}, {r['ci_high']*100:.2f}]"
            if r["ci_low"] is not None
            else "-"
        )
        lines.append(
            f"| {r['method']} | {r['hits']:,}/{r['n']:,} | "
            f"{r['accuracy']*100:.2f}% | {ci} | "
            f"{r['macro_f1']*100:.2f} | {r['weighted_f1']*100:.2f} |"
        )
    return "\n".join(lines)


def to_latex(manifests: List[dict], caption: str = "", label: str = "tab:wsd") -> str:
    rows = [_row_from_manifest(m) for m in manifests]
    rows.sort(key=lambda r: (r["accuracy"] is None, -(r["accuracy"] or 0)))
    out = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Hits & Accuracy & Macro-F1 & Weighted-F1 \\\\",
        "\\midrule",
    ]
    for r in rows:
        out.append(
            f"{r['method']} & {r['hits']}/{r['n']} & "
            f"{r['accuracy']*100:.2f}\\% & {r['macro_f1']*100:.2f} & "
            f"{r['weighted_f1']*100:.2f} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}"]
    if caption:
        out.append(f"\\caption{{{caption}}}")
    out.append(f"\\label{{{label}}}")
    out.append("\\end{table}")
    return "\n".join(out)


def per_word_markdown(manifest: dict) -> str:
    pw = manifest["metrics"]["per_word"]
    lines = [
        "| Word | Senses | Hits | Accuracy | Macro-F1 |",
        "| ---- | ------ | ---- | -------- | -------- |",
    ]
    for word in sorted(pw):
        r = pw[word]
        lines.append(
            f"| {word} | {r['num_senses']} | {r['hits']}/{r['n']} | "
            f"{r['accuracy']*100:.2f}% | {r['macro_f1']*100:.2f} |"
        )
    return "\n".join(lines)
