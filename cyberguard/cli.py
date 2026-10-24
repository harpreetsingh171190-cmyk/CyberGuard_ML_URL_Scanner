"""Command line interface:  python -m cyberguard train|predict|scan"""
from __future__ import annotations

import argparse
import sys

import matplotlib
import pandas as pd

from . import __version__
from .data import demo_dataset, load_dataset, split_data
from .features import features_frame
from .modeling import (TEXT_MODELS, dump_report, error_analysis, explain, load_bundle, pick_threshold, predict_urls,
                       save_bundle, train_and_compare)

MODEL_PATH = "models/cyberguard_model.joblib"


def cmd_train(a):
    matplotlib.use("Agg")
    from . import plots
    demo = a.data is None
    df = demo_dataset() if demo else load_dataset(a.data, a.max_rows)
    if demo:
        print("DEMO MODE: tiny hand-written sample. Scores below are NOT meaningful.")
    print(f"{len(df):,} URLs | malicious share {df['label'].mean():.1%}")
    train_df, test_df = split_data(df, group_by_domain=not a.no_group_split, time_split=a.time_split)
    results, fitted, probs, y_te = train_and_compare(train_df, test_df)
    print(results.round(4).to_string(index=False))
    best = results.loc[0, "Model"]
    thr = pick_threshold(y_te, probs[best], a.min_recall)
    print(f"\nBest model: {best} | decision threshold for recall >= {a.min_recall:.0%}: {thr:.3f}")
    plots.confusion(y_te, probs[best], thr, best, a.report_dir)
    plots.roc_and_pr(y_te, probs, a.report_dir)
    plots.feature_importance(fitted[best], a.report_dir)
    errs = error_analysis(test_df, probs[best], thr)
    errs.to_csv(f"{a.report_dir}/error_analysis.csv", index=False)
    print(f"\nError analysis saved ({len(errs)} rows). Most confident mistakes:")
    print(errs.head(5)[["error_type", "malicious_probability", "url"]].to_string(index=False) if len(errs) else "none")
    metrics = results.loc[0].drop("Model").round(4).to_dict()
    dump_report(results, {"best_model": best, "threshold": thr, "demo": demo, "n_urls": len(df)},
                f"{a.report_dir}/metrics.json")
    baseline = features_frame(train_df["url"]).median().to_dict()
    print("Saved:", save_bundle(fitted[best], best, thr, metrics, a.model, demo, baseline, best in TEXT_MODELS))


def cmd_predict(a):
    bundle = load_bundle(a.model)
    if bundle.get("demo"):
        print("Warning: this model was trained on the demo set. Do not trust it.\n", file=sys.stderr)
    for url in a.urls:
        r = explain(bundle, url)
        print(f"{r['verdict'].upper():<11} {r['malicious_probability']:.1%}  [{r['risk_level']}]  {url}")
        for s in r["signals"]:
            print(f"    - {s}")
        if a.why:
            for t in r["top_features"]:
                print(f"    * {t['feature']}={t['value']:.2f} (typical {t['typical']:.2f}) -> {t['impact']:+.3f}")


def cmd_scan(a):
    bundle = load_bundle(a.model)
    urls = pd.read_csv(a.input)[a.column] if a.input.endswith(".csv") else pd.Series(open(a.input).read().split())
    out = predict_urls(bundle, urls)
    out.to_csv(a.output, index=False)
    print(f"Scored {len(out):,} URLs -> {a.output}\n{out['verdict'].value_counts().to_string()}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cyberguard", description="ML malicious URL scanner (educational).")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train and compare models")
    t.add_argument("--data", help="CSV with url + label/type columns (omit for demo set)")
    t.add_argument("--max-rows", type=int, default=100_000)
    t.add_argument("--min-recall", type=float, default=0.85)
    t.add_argument("--no-group-split", action="store_true")
    t.add_argument("--time-split", action="store_true", help="train on older URLs, test on newest (needs a date column)")
    t.add_argument("--model", default=MODEL_PATH)
    t.add_argument("--report-dir", default="reports")
    t.set_defaults(fn=cmd_train)

    q = sub.add_parser("predict", help="check one or more URLs")
    q.add_argument("urls", nargs="+")
    q.add_argument("--model", default=MODEL_PATH)
    q.add_argument("--why", action="store_true", help="show top feature impacts")
    q.set_defaults(fn=cmd_predict)

    s = sub.add_parser("scan", help="score a CSV (column) or text file of URLs")
    s.add_argument("input")
    s.add_argument("--column", default="url")
    s.add_argument("--output", default="scan_results.csv")
    s.add_argument("--model", default=MODEL_PATH)
    s.set_defaults(fn=cmd_scan)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
