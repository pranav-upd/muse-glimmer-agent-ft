#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LM Evaluation Harness driver for Muse Glimmer (single benchmark).

Runs EleutherAI GPQA Diamond (zero-shot) on Modal GPU via the deployed
``lm_harness`` function from starter.py.

Usage::

    python -m modal deploy starter.py   # required once (publishes lm_harness)
    python eval.py --lm-limit 10        # smoke: 10 samples
    python eval.py                      # full GPQA Diamond, both weights
    python eval.py --lm-source hub      # finetuned LoRA only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("glimmer-lm-eval")

MODAL_APP = "muse-glimmer-grpo"
MODAL_FN = "lm_harness"


def run_lm_eval(tasks: list[str], source: str, limit: int | None, temperature: float = 0.3) -> int:
    """Score base and/or finetuned weights with the LM Evaluation Harness.

    Executes remotely on Modal GPU via the deployed ``lm_harness`` function.
    Prints a per-task comparison table. Returns 0 on success.
    """
    import modal  # local Modal client — lookup only, compute stays remote

    log.info("Looking up deployed lm_harness (deploy starter.py first if missing) ...")
    try:
        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
    except Exception as exc:
        log.error("Lookup failed: %s. Deploy with: python -m modal deploy starter.py", exc)
        return 2
    log.info("Running lm-eval tasks=%s source=%s limit=%s temperature=%s ...",
             tasks, source, limit, temperature)
    try:
        result = fn.remote(tasks, source, limit, temperature)
    except Exception as exc:
        log.error("Remote lm-eval failed: %s", exc)
        return 1
    sources = result.get("sources", {})
    names = sorted({t for s in sources.values() if isinstance(s, dict) for t in s if t != "error"})
    if result.get("error") or not names:
        blob = json.dumps(result)
        log.error("lm-eval error head: %.800s", blob[:800])
        log.error("lm-eval error tail: %.1500s", blob[-1500:])
        return 1
    header = f"{'task':36s}" + "".join(f"{s:>22s}" for s in sorted(sources))
    print(header)
    for task in names:
        row = f"{task:36s}"
        for src in sorted(sources):
            vals = sources[src].get(task, {})
            if not isinstance(vals, dict) or "error" in vals:
                row += f"{'ERR':>22s}"
                continue
            pick = vals.get("acc,none") or vals.get("acc_norm,none") or vals.get("acc")
            row += f"{(round(float(pick), 4) if pick is not None else 'n/a'):>22}"
        print(row)
    for src in sorted(sources):
        src_vals = sources[src]
        if isinstance(src_vals, dict):
            if "error" in src_vals:
                log.error("%s error: %.500s", src, src_vals["error"])
            for task, vals in src_vals.items():
                if isinstance(vals, dict) and "error" in vals:
                    log.error("%s/%s error: %.400s", src, task, vals["error"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPQA Diamond eval driver for Muse Glimmer")
    parser.add_argument("--lm", action="store_true",
                        help="Run the benchmark (explicit switch; running without flags does the same).")
    parser.add_argument("--lm-tasks", default="gpqa_diamond_zeroshot",
                        help="lm-eval task (single benchmark: GPQA Diamond zero-shot)")
    parser.add_argument("--lm-source", default="both", choices=["hub", "base", "both"],
                        help="Which weights to score: finetuned LoRA (hub), base, or both")
    parser.add_argument("--lm-limit", type=int, default=None,
                        help="Cap samples per task (e.g. 10 for a smoke test)")
    parser.add_argument("--lm-temp", type=float, default=0.3,
                        help="Sampling temperature (no-op for loglikelihood tasks like GPQA)")
    args = parser.parse_args(argv)
    return run_lm_eval(
        tasks=[t.strip() for t in args.lm_tasks.split(",") if t.strip()],
        source=args.lm_source,
        limit=args.lm_limit,
        temperature=args.lm_temp,
    )


if __name__ == "__main__":
    sys.exit(main())
