"""Experiment definitions. Each experiment yields jobs: one prompt plus the metadata needed to
score it. Items are shared across languages and templates so that comparisons are paired."""
from .data import build_item, render_context, render_question, spread

POSITIONS = [0, 3, 6, 9, 13, 16, 19]          # 0-indexed patch of the target (patches 1..20)
EDGE, MIDDLE = {0, 19}, {6, 9, 13}

PRESETS = {
    # n = items per cell
    "smoke":    {"main": 2,  "oracle": 4,   "length": 2,   "crosslang": 2,  "D": [0, 4]},
    "standard": {"main": 30, "oracle": 100, "length": 70,  "crosslang": 50, "D": [0, 4]},
    "full":     {"main": 50, "oracle": 150, "length": 135, "crosslang": 50, "D": [0, 2, 4, 8]},
}

LENGTH_POS = [0, 9, 19]
LENGTH_PAD = 4          # padding sentences per patch in the base context of the length experiment
MAIN_PAD = 1


def item_seed(exp, i, pos, d):
    base = {"main": 1, "oracle": 2, "length": 3, "crosslang": 4}[exp]
    return base * 10_000_000 + i * 10_000 + pos * 100 + d


def _job(exp, cond, i, item, ctx_lang, q_lang, template, context, **extra):
    q = render_question(item, q_lang)
    prompt = f"{context}\n\n{q}" if context else q
    return {"exp": exp, "cond": cond, "i": i, "pos": item["target_patch"], "d": item["n_distractors"],
            "template": template, "ctx_lang": ctx_lang, "q_lang": q_lang, "prompt": prompt,
            "pair_id": f"{exp}|{i}|{item['target_patch']}|{item['n_distractors']}", "item": item, **extra}


def main_jobs(n, D):
    for i in range(n):
        for pos in POSITIONS:
            for d in D:
                item = build_item(item_seed("main", i, pos, d), target_patch=pos, n_distractors=d,
                                  pad_per_patch=MAIN_PAD)
                for template in ("native", "arrow"):
                    for lang in ("en", "fr"):
                        yield _job("main", template, i, item, lang, lang, template,
                                   render_context(item, lang, template))


def oracle_jobs(n):
    for i in range(n):
        item = build_item(item_seed("oracle", i, 9, 0), target_patch=9, n_distractors=0, pad_per_patch=0)
        for lang in ("en", "fr"):
            for template in ("native", "arrow"):
                yield _job("oracle", f"oracle_{template}", i, item, lang, lang, template,
                           render_context(item, lang, template, keep=[9]))
            yield _job("oracle", "closed_book", i, item, lang, lang, "native", None)


def _closest(lo, hi, f, target):
    """Smallest s in [lo, hi] with f(s) >= target (f increasing); hi if none."""
    while lo < hi:
        mid = (lo + hi) // 2
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def length_jobs(n, ntok):
    """Content-matched (en, fr), token-matched (en_long, fr_short), length sweep (en_plus50) and the
    fixed-budget baseline used in prior work (fr_budget: French patches dropped to fit the English length)."""
    for i in range(n):
        for pos in LENGTH_POS:
            item = build_item(item_seed("length", i, pos, 0), target_patch=pos, n_distractors=0,
                              pad_per_patch=LENGTH_PAD)
            P = len(item["patches"])
            base = LENGTH_PAD * P
            ctx = lambda lang, total: render_context(item, lang, "native", pad_counts=spread(total, P))
            en, fr = ctx("en", base), ctx("fr", base)
            L_en, L_fr = ntok(en), ntok(fr)
            meta = {"L_en": L_en, "L_fr": L_fr}
            yield _job("length", "en", i, item, "en", "en", "native", en, ctx_tokens=L_en, **meta)
            yield _job("length", "fr", i, item, "fr", "fr", "native", fr, ctx_tokens=L_fr, **meta)
            # EN-long: more padding until English reaches the French length
            s = _closest(base, base * 4, lambda t: ntok(ctx("en", t)), L_fr)
            c = ctx("en", s)
            yield _job("length", "en_long", i, item, "en", "en", "native", c, ctx_tokens=ntok(c), **meta)
            # length sweep: English +50%
            s = _closest(base, base * 8, lambda t: ntok(ctx("en", t)), int(1.5 * L_en))
            c = ctx("en", s)
            yield _job("length", "en_plus50", i, item, "en", "en", "native", c, ctx_tokens=ntok(c), **meta)
            # FR-short: less padding until French matches the English length (largest s with ntok <= L_en)
            s = _closest(0, base, lambda t: ntok(ctx("fr", t)), L_en + 1) - 1
            s = max(s, 0)
            c = ctx("fr", s)
            yield _job("length", "fr_short", i, item, "fr", "fr", "native", c, ctx_tokens=ntok(c),
                       matched=ntok(c) <= L_en, **meta)
            # Baseline protocol (fixed token budget): drop non-target French patches, evenly spaced
            cands = [p for p in range(P) if p != pos]
            for m in range(1, len(cands) + 1):
                step = len(cands) / m
                drop = {cands[int(k * step)] for k in range(m)}
                keep = [p for p in range(P) if p not in drop]
                c = render_context(item, "fr", "native", keep=keep)
                if ntok(c) <= L_en:
                    break
            yield _job("length", "fr_budget", i, item, "fr", "fr", "native", c, ctx_tokens=ntok(c),
                       patches_kept=len(keep), **meta)


def crosslang_jobs(n):
    for i in range(n):
        for pos in LENGTH_POS:
            item = build_item(item_seed("crosslang", i, pos, 0), target_patch=pos, n_distractors=0,
                              pad_per_patch=MAIN_PAD)
            for ctx_lang in ("en", "fr"):
                context = render_context(item, ctx_lang, "native")
                for q_lang in ("en", "fr"):
                    yield _job("crosslang", f"ctx_{ctx_lang}_q_{q_lang}", i, item, ctx_lang, q_lang,
                               "native", context)


def build_jobs(experiments, preset, ntok, overrides=None):
    cfg = dict(PRESETS[preset]); cfg.update(overrides or {})
    jobs = []
    for exp in experiments:
        if exp == "main":
            jobs += list(main_jobs(cfg["main"], cfg["D"]))
        elif exp == "oracle":
            jobs += list(oracle_jobs(cfg["oracle"]))
        elif exp == "length":
            jobs += list(length_jobs(cfg["length"], ntok))
        elif exp == "crosslang":
            jobs += list(crosslang_jobs(cfg["crosslang"]))
        else:
            raise ValueError(f"unknown experiment {exp}")
    for j in jobs:
        j["job_id"] = f"{j['pair_id']}|{j['cond']}|{j['ctx_lang']}|{j['q_lang']}"
    return jobs, cfg
