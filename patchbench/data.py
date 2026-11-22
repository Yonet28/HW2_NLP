"""Synthetic patch-note items rendered with identical content in English and French.

An *item* is language independent: a list of patches, each holding balance changes
(champion, ability key, statistic, previous value, new value) and a number of padding
sentences. Rendering turns an item into text for a given language and template.
"""
import random
import re

CHAMPIONS = ["Ahri", "Garen", "Lux", "Darius", "Jinx", "Ezreal", "Yasuo", "Thresh", "Leona", "Morgana",
             "Vayne", "Annie", "Ashe", "Brand", "Caitlyn", "Diana", "Ekko", "Fiora", "Graves", "Irelia",
             "Janna", "Karma", "Katarina", "Lissandra", "Malphite", "Nami", "Orianna", "Pyke", "Riven", "Sona",
             "Syndra", "Talon", "Teemo", "Veigar", "Viktor", "Xayah", "Zed", "Ziggs", "Zoe", "Zyra"]
KEYS = ["Q", "W", "E", "R"]

# stat id -> value range (min, max, step) and unit
STAT_RANGE = {
    "damage":   (40, 300, 5, ""),
    "cooldown": (4, 20, 0.5, " s"),
    "mana":     (30, 120, 5, ""),
    "range":    (300, 1200, 25, ""),
}
STAT_EN = {"damage": "base damage", "cooldown": "cooldown", "mana": "mana cost", "range": "range"}
# French: noun phrase with article, bare noun phrase, "de" form, plural verb
STAT_FR = {
    "damage":   ("les dégâts de base", "dégâts de base", "des dégâts de base", True),
    "cooldown": ("le temps de recharge", "temps de recharge", "du temps de recharge", False),
    "mana":     ("le coût en mana", "coût en mana", "du coût en mana", False),
    "range":    ("la portée", "portée", "de la portée", False),
}

# Parallel padding sentences: patch-style commentary with no number, no champion,
# no ability key and no statistic name, so that padding cannot act as a distractor.
PADDING = [
    ("This update continues our work on overall game balance.",
     "Cette mise à jour poursuit notre travail sur l'équilibre général du jeu."),
    ("We have read your feedback carefully and will keep listening.",
     "Nous avons lu attentivement vos retours et nous continuerons à vous écouter."),
    ("Some adjustments target the early game, while others focus on later stages.",
     "Certains ajustements visent le début de partie, d'autres se concentrent sur les phases plus tardives."),
    ("Our goal is to make more strategies viable in competitive play.",
     "Notre objectif est de rendre davantage de stratégies viables en compétition."),
    ("We will monitor the impact of these changes over the coming weeks.",
     "Nous suivrons l'impact de ces changements au cours des prochaines semaines."),
    ("Several visual effects have been polished for better readability.",
     "Plusieurs effets visuels ont été retravaillés pour une meilleure lisibilité."),
    ("Thank you to everyone who reported issues on the public test server.",
     "Merci à toutes les personnes qui ont signalé des problèmes sur le serveur de test public."),
    ("The team also fixed a number of minor bugs reported by players.",
     "L'équipe a également corrigé plusieurs petits bugs signalés par les joueurs."),
    ("These changes aim to reward good decisions rather than raw reflexes.",
     "Ces changements visent à récompenser les bonnes décisions plutôt que les simples réflexes."),
    ("Ranked games will use these settings as soon as the update is live.",
     "Les parties classées utiliseront ces réglages dès que la mise à jour sera en ligne."),
    ("We know that frequent changes can be hard to follow, so we keep these notes short.",
     "Nous savons que des changements fréquents peuvent être difficiles à suivre, nous gardons donc ces notes courtes."),
    ("Professional teams were consulted before finalizing this update.",
     "Les équipes professionnelles ont été consultées avant de finaliser cette mise à jour."),
    ("If something feels wrong after this update, please let us know.",
     "Si quelque chose vous semble anormal après cette mise à jour, n'hésitez pas à nous le signaler."),
    ("The overall pace of matches should remain largely unchanged.",
     "Le rythme général des parties devrait rester globalement inchangé."),
    ("Some interface texts were rewritten to be clearer for new players.",
     "Certains textes de l'interface ont été réécrits pour être plus clairs pour les nouveaux joueurs."),
    ("This round of changes is smaller than usual by design.",
     "Cette série de changements est volontairement plus réduite que d'habitude."),
    ("We are also preparing larger updates for later in the season.",
     "Nous préparons aussi des mises à jour plus importantes pour la suite de la saison."),
    ("As always, the full list of fixes is available on our website.",
     "Comme toujours, la liste complète des corrections est disponible sur notre site."),
    ("Matchmaking will be watched closely during the first days.",
     "La recherche de partie sera surveillée de près pendant les premiers jours."),
    ("Players should not need to change their habits after this update.",
     "Les joueurs ne devraient pas avoir besoin de changer leurs habitudes après cette mise à jour."),
    ("Audio cues were reviewed to make important moments easier to notice.",
     "Les signaux sonores ont été revus pour rendre les moments importants plus faciles à repérer."),
    ("Our designers tested these changes in many different situations.",
     "Nos concepteurs ont testé ces changements dans de nombreuses situations différentes."),
    ("Some of these adjustments may be reverted if they do not work as intended.",
     "Certains de ces ajustements pourront être annulés s'ils ne fonctionnent pas comme prévu."),
    ("We appreciate the patience of the community during this period.",
     "Nous remercions la communauté pour sa patience pendant cette période."),
]

LANGS = ("en", "fr")
TEMPLATES = ("native", "arrow")


def rand_val(rng, stat):
    lo, hi, step, _ = STAT_RANGE[stat]
    return lo + step * rng.randint(0, int(round((hi - lo) / step)))


def fmt(v, lang):
    s = str(int(v)) if float(v).is_integer() else f"{v:.1f}"
    return s.replace(".", ",") if lang == "fr" else s


def version(p):
    return f"14.{p + 1}"


# --------------------------------------------------------------------------- items

def build_item(seed, n_patches=20, changes_per_patch=12, target_patch=9, n_distractors=0, pad_per_patch=1):
    """Language-independent item. target_patch is 0-indexed. Distractors are changes of the
    same (champion, key, statistic) in other patches."""
    rng = random.Random(seed)
    target = (rng.choice(CHAMPIONS), rng.choice(KEYS), rng.choice(list(STAT_RANGE)))
    others = [i for i in range(n_patches) if i != target_patch]
    target_patches = set([target_patch] + rng.sample(others, n_distractors))
    current = {}
    patches = []
    for p in range(n_patches):
        triples, used = [], set()
        if p in target_patches:
            triples.append(target); used.add(target)
        while len(triples) < changes_per_patch:
            t = (rng.choice(CHAMPIONS), rng.choice(KEYS), rng.choice(list(STAT_RANGE)))
            if t == target or t in used:
                continue
            used.add(t); triples.append(t)
        rng.shuffle(triples)
        changes = []
        for t in triples:
            old = current.get(t, rand_val(rng, t[2]))
            new = old
            while new == old:
                new = rand_val(rng, t[2])
            current[t] = new
            changes.append({"champ": t[0], "key": t[1], "stat": t[2], "old": old, "new": new,
                            "is_target": t == target and p == target_patch})
        patches.append({"version": version(p), "changes": changes})
    target_change = next(c for c in patches[target_patch]["changes"] if c["is_target"])
    other_vals = {c[k] for pt in patches for c in pt["changes"]
                  if (c["champ"], c["key"], c["stat"]) == target for k in ("old", "new")}
    other_vals.discard(target_change["new"])
    # padding sentence ids, fixed per item so that every rendering shares them
    pad_order = list(range(len(PADDING)))
    rng.shuffle(pad_order)
    return {
        "seed": seed, "target": list(target), "target_patch": target_patch, "n_distractors": n_distractors,
        "patches": patches, "answer": target_change["new"], "previous": target_change["old"],
        "distractor_values": sorted(other_vals), "pad_order": pad_order,
        "pad_counts": [pad_per_patch] * n_patches,
    }


def spread(total, n):
    """Distribute `total` padding sentences over n patches as evenly as possible."""
    base, extra = divmod(total, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


# --------------------------------------------------------------------------- rendering

def render_change(c, lang, template):
    unit = STAT_RANGE[c["stat"]][3]
    o, n = fmt(c["old"], lang) + unit, fmt(c["new"], lang) + unit
    if template == "arrow":
        name = STAT_EN[c["stat"]] if lang == "en" else STAT_FR[c["stat"]][1]
        return f"{c['champ']} {c['key']}, {name}: {o} → {n}"
    if lang == "en":
        return f"{c['champ']}: {c['key']} {STAT_EN[c['stat']]} changed from {o} to {n}."
    art, _, _, plural = STAT_FR[c["stat"]]
    return f"{c['champ']} : {art} du {c['key']} {'passent' if plural else 'passe'} de {o} à {n}."


def patch_header(v, lang):
    if lang == "en":
        return f"=== Patch {v} notes ===\nHere are the champion balance changes for patch {v}."
    return f"=== Notes du patch {v} ===\nVoici les changements d'équilibrage des champions pour le patch {v}."


def render_patch(item, p, lang, template, n_pad, pad_cursor):
    lines = [patch_header(item["patches"][p]["version"], lang)]
    li = 0 if lang == "en" else 1
    pads = []
    for _ in range(n_pad):
        pads.append(PADDING[item["pad_order"][pad_cursor % len(PADDING)]][li])
        pad_cursor += 1
    if pads:
        lines.append(" ".join(pads))
    lines += [render_change(c, lang, template) for c in item["patches"][p]["changes"]]
    return "\n".join(lines), pad_cursor


def render_context(item, lang, template="native", pad_counts=None, keep=None):
    """pad_counts: padding sentences per patch (defaults to the item's).
    keep: indices of patches to keep (oracle and budget conditions)."""
    pad_counts = pad_counts if pad_counts is not None else item["pad_counts"]
    keep = range(len(item["patches"])) if keep is None else keep
    blocks, cursor = [], 0
    for p in keep:
        text, cursor = render_patch(item, p, lang, template, pad_counts[p], cursor)
        blocks.append(text)
    return "\n\n".join(blocks)


def render_question(item, lang, style="marker"):
    """style 'marker' (final prompt) asks for "Answer: <number>"; style 'plain' is the pilot prompt."""
    champ, key, stat = item["target"]
    v = item["patches"][item["target_patch"]]["version"]
    if lang == "en":
        q = (f"According to the patch notes above, in patch {v}, what is the new value of {champ}'s "
             f"{key} {STAT_EN[stat]}? Answer with the number only")
        return q + (", in the format \"Answer: <number>\"." if style == "marker" else ".")
    q = (f"D'après les notes de patch ci-dessus, dans le patch {v}, quelle est la nouvelle valeur "
         f"{STAT_FR[stat][2]} du {key} de {champ} ? Réponds uniquement par le nombre")
    return q + (", au format « Réponse : <nombre> »." if style == "marker" else ".")


# --------------------------------------------------------------------------- scoring

MARKER = re.compile(r"(?:answer|r[ée]ponse)\s*:\s*(-?\d+(?:[.,]\d+)?)", re.IGNORECASE)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_answer(text):
    """Returns (value or None, marker_found)."""
    m = MARKER.search(text)
    if m:
        return float(m.group(1).replace(",", ".")), True
    m = NUMBER.search(text)
    return (float(m.group().replace(",", ".")) if m else None), False


def score(item, output):
    pred, marker = parse_answer(output)
    eq = lambda a, b: a is not None and abs(a - b) < 1e-6
    return {
        "pred": pred, "marker_found": marker, "parsed": pred is not None,
        "correct": eq(pred, item["answer"]),
        "trap": any(eq(pred, v) for v in item["distractor_values"]),
        "trap_previous": eq(pred, item["previous"]),
    }
