"""
hardneg.py - a synthetic, templated test set for the failure mode that
motivated this project: questions that look almost identical but ask for
different things ("capital of India" vs "capital of Japan").

Public duplicate-question datasets do not test this directly, so we build
a set where every pair is labelled by construction and the labels are
double-checked by an independent lookup table (see meaning_of()).

Pair types
  paraphrase      label 1  same meaning, different wording
  entity_swap     label 0  same question, different entity   (capital of India / Japan)
  number_swap     label 0  same question, different number   (refund in 30 / 60 days)
  attribute_swap  label 0  same entity, different thing asked (price / weight of the X1 laptop)
  polarity_swap   label 0  opposite action                   (enable / disable dark mode)
  unrelated       label 0  different topics entirely (the easy case, for realism)

LIMITS - read before trusting any result on this set:
  * It is synthetic and written by us. Language is far more repetitive than
    real chat traffic, and a model could pick up template regularities.
  * Different seeds draw from the SAME finite pool, so results across seeds
    are correlated, not independent replications.
  * It tests one failure mode. Doing well here does not show general quality.
"""

import itertools
import random
import re

# ----------------------------------------------------------------------------
# Question families: an attribute asked about an entity.
# ----------------------------------------------------------------------------

FAMILIES = {
    "countries": {
        "entities": ["India", "Japan", "France", "Germany", "Brazil", "Canada", "Egypt",
                     "Kenya", "Peru", "Norway", "Italy", "Spain", "Mexico", "Turkey"],
        "attributes": {
            "capital": ["What is the capital of {x}?", "Which city is the capital of {x}?",
                        "Tell me the capital city of {x}.", "{x}'s capital is which city?"],
            "currency": ["What currency does {x} use?", "Which currency is used in {x}?",
                         "What is the official currency of {x}?", "Tell me what money people use in {x}."],
            "language": ["What language do people speak in {x}?", "Which language is spoken in {x}?",
                         "What is the main language of {x}?", "Tell me the primary language used in {x}."],
        },
    },
    "products": {
        "entities": ["X1 laptop", "Pro tablet", "Air phone", "Max headphones", "Lite speaker",
                     "Edge watch", "Nova camera"],
        "attributes": {
            "price": ["How much does the {x} cost?", "What is the price of the {x}?",
                      "How expensive is the {x}?"],
            "warranty": ["How long is the warranty on the {x}?", "What warranty does the {x} come with?",
                         "How many years of warranty does the {x} have?"],
            "battery": ["How long does the {x} battery last?", "What is the battery life of the {x}?",
                        "How many hours does the {x} run on one charge?"],
            "weight": ["How much does the {x} weigh?", "What is the weight of the {x}?",
                       "How heavy is the {x}?"],
        },
    },
    "companies": {
        "entities": ["Apple", "Microsoft", "Amazon", "Tesla", "Google", "Netflix", "Spotify",
                     "Nvidia", "Adobe", "Intel"],
        "attributes": {
            "founder": ["Who founded {x}?", "Who is the founder of {x}?", "Who started the company {x}?"],
            "headquarters": ["Where is {x} headquartered?", "Where are the headquarters of {x}?",
                             "In which city is the head office of {x}?"],
            "founding_year": ["When was {x} founded?", "In what year was {x} established?",
                              "What year did {x} start?"],
        },
    },
    "services": {
        "entities": ["Gmail", "Netflix", "Instagram", "Zoom", "Dropbox", "Spotify", "Slack", "PayPal"],
        "attributes": {
            "reset_password": ["How do I reset my {x} password?",
                               "I forgot my {x} password, how can I get a new one?",
                               "What are the steps to recover my {x} password?"],
            "delete_account": ["How do I delete my {x} account?",
                               "What is the process to permanently close my {x} account?",
                               "I want to remove my {x} account, how do I do that?"],
            "change_email": ["How do I change the email on my {x} account?",
                             "Can I update the email address linked to my {x} account?",
                             "What are the steps to switch my {x} account email?"],
        },
    },
}

# ----------------------------------------------------------------------------
# Number intents: the same question with a different number is a different question.
# ----------------------------------------------------------------------------

NUMBER_INTENTS = {
    "refund_window_days": {
        "values": [7, 14, 30, 60, 90],
        "templates": ["Can I get a refund within {x} days?",
                      "Is a refund possible within {x} days of purchase?",
                      "Do you refund purchases made in the last {x} days?"],
    },
    "warranty_years": {
        "values": [2, 3, 5, 10],
        "templates": ["Does the product come with a {x}-year warranty?",
                      "Is there a {x} year warranty on this item?",
                      "Do you offer {x} years of warranty coverage?"],
    },
    "delivery_days": {
        "values": [2, 3, 5, 7, 10],
        "templates": ["Will my order arrive in {x} days?", "Can you deliver within {x} days?",
                      "Is delivery possible in {x} days?"],
    },
    "password_length": {
        "values": [6, 8, 10, 12, 16],
        "templates": ["Must my password be at least {x} characters long?",
                      "Is a minimum of {x} characters required for passwords?",
                      "Does the password need {x} or more characters?"],
    },
}

# ----------------------------------------------------------------------------
# Polarity: opposite actions on the same object.
# ----------------------------------------------------------------------------

POLARITY = [
    ("enable", "disable", ["two-factor authentication", "dark mode", "auto-renewal", "email alerts"]),
    ("add", "remove", ["a payment method", "a profile photo", "a shipping address", "a backup email"]),
    ("lock", "unlock", ["my account", "my card", "my phone screen"]),
    ("open", "close", ["a support ticket", "a savings account", "a chat window"]),
    ("increase", "decrease", ["the font size", "the volume", "the brightness"]),
    ("turn on", "turn off", ["notifications", "location sharing", "airplane mode"]),
]
POLARITY_TEMPLATES = ["How do I {v} {o}?", "What is the way to {v} {o}?",
                      "Can you tell me how to {v} {o}?", "I want to {v} {o}, how do I do that?"]

NEGATIVE_QUOTAS = {"entity_swap": 0.25, "number_swap": 0.20, "attribute_swap": 0.25,
                   "polarity_swap": 0.15, "unrelated": 0.15}
CATEGORIES = ["paraphrase"] + list(NEGATIVE_QUOTAS)


# ----------------------------------------------------------------------------
# Independent label oracle
# ----------------------------------------------------------------------------

def _norm(s):
    s = re.sub(r"[^a-z0-9' ]+", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _all_questions():
    """Every (text, meaning) the generator can produce. `meaning` identifies
    exactly what is being asked, independent of wording."""
    for fam, spec in FAMILIES.items():
        for attr, templates in spec["attributes"].items():
            for ent in spec["entities"]:
                for t in templates:
                    yield t.format(x=ent), ("fam", fam, attr, ent)
    for intent, spec in NUMBER_INTENTS.items():
        for v in spec["values"]:
            for t in spec["templates"]:
                yield t.format(x=v), ("num", intent, v)
    for v1, v2, objs in POLARITY:
        for verb in (v1, v2):
            for o in objs:
                for t in POLARITY_TEMPLATES:
                    yield t.format(v=verb, o=o), ("pol", verb, o)


def build_meaning_table():
    """Map normalised text -> meaning. Raises if two different meanings share
    the same text, which would make a label ambiguous."""
    table = {}
    for text, meaning in _all_questions():
        key = _norm(text)
        if key in table and table[key] != meaning:
            raise ValueError(f"Ambiguous question text {text!r}: {table[key]} vs {meaning}")
        table[key] = meaning
    return table


_MEANING = build_meaning_table()


def meaning_of(text):
    """What this question asks, looked up independently of how pairs are built.
    Casing and punctuation are ignored. None if the text is not from this set."""
    return _MEANING.get(_norm(text))


# ----------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------

def _positive_pool():
    """Every distinct paraphrase pair: same meaning, two different wordings."""
    pool = []

    def add(texts):
        pool.extend(itertools.combinations(texts, 2))

    for spec in FAMILIES.values():
        for templates in spec["attributes"].values():
            for ent in spec["entities"]:
                add([t.format(x=ent) for t in templates])
    for spec in NUMBER_INTENTS.values():
        for v in spec["values"]:
            add([t.format(x=v) for t in spec["templates"]])
    for v1, v2, objs in POLARITY:
        for verb in (v1, v2):
            for o in objs:
                add([t.format(v=verb, o=o) for t in POLARITY_TEMPLATES])
    return pool


def max_pairs(positive_fraction=0.4):
    """Largest n this generator can supply at the given positive fraction."""
    return int(len(_positive_pool()) / positive_fraction)


def _draw_entity_swap(rng):
    spec = FAMILIES[rng.choice(list(FAMILIES))]
    templates = spec["attributes"][rng.choice(list(spec["attributes"]))]
    e1, e2 = rng.sample(spec["entities"], 2)
    return rng.choice(templates).format(x=e1), rng.choice(templates).format(x=e2)


def _draw_number_swap(rng):
    spec = NUMBER_INTENTS[rng.choice(list(NUMBER_INTENTS))]
    v1, v2 = rng.sample(spec["values"], 2)
    return rng.choice(spec["templates"]).format(x=v1), rng.choice(spec["templates"]).format(x=v2)


def _draw_attribute_swap(rng):
    spec = FAMILIES[rng.choice(list(FAMILIES))]
    entity = rng.choice(spec["entities"])
    a1, a2 = rng.sample(list(spec["attributes"]), 2)
    return (rng.choice(spec["attributes"][a1]).format(x=entity),
            rng.choice(spec["attributes"][a2]).format(x=entity))


def _draw_polarity_swap(rng):
    v1, v2, objs = rng.choice(POLARITY)
    o = rng.choice(objs)
    return (rng.choice(POLARITY_TEMPLATES).format(v=v1, o=o),
            rng.choice(POLARITY_TEMPLATES).format(v=v2, o=o))


def _random_question(group, rng):
    if group == "family":
        spec = FAMILIES[rng.choice(list(FAMILIES))]
        templates = spec["attributes"][rng.choice(list(spec["attributes"]))]
        return rng.choice(templates).format(x=rng.choice(spec["entities"]))
    if group == "number":
        spec = NUMBER_INTENTS[rng.choice(list(NUMBER_INTENTS))]
        return rng.choice(spec["templates"]).format(x=rng.choice(spec["values"]))
    v1, v2, objs = rng.choice(POLARITY)
    return rng.choice(POLARITY_TEMPLATES).format(v=rng.choice([v1, v2]), o=rng.choice(objs))


def _draw_unrelated(rng):
    g1, g2 = rng.sample(["family", "number", "polarity"], 2)  # different topics
    return _random_question(g1, rng), _random_question(g2, rng)


_DRAWERS = {"entity_swap": _draw_entity_swap, "number_swap": _draw_number_swap,
            "attribute_swap": _draw_attribute_swap, "polarity_swap": _draw_polarity_swap,
            "unrelated": _draw_unrelated}


def _largest_remainder(weights, total):
    raw = {k: w * total for k, w in weights.items()}
    base = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[: total - sum(base.values())]:
        base[k] += 1
    return base


def _casual(s):
    """Mimic quick typing: lowercase, no punctuation. Applied to either label
    with the same probability so formatting cannot reveal the answer."""
    return re.sub(r"[?.!,]", "", s).lower()


def _key(a, b):
    return frozenset((_norm(a), _norm(b)))


def generate_pairs(n, seed=0, positive_fraction=0.4, casual_prob=0.3):
    """
    Returns n tuples (question_a, question_b, label, category), deterministic
    for a given seed. Raises ValueError if the pool cannot supply n pairs.
    """
    rng = random.Random(seed)
    n_pos = round(n * positive_fraction)
    n_neg = n - n_pos

    pool = _positive_pool()
    if n_pos > len(pool):
        raise ValueError(
            f"hardneg can supply at most {len(pool)} distinct paraphrase pairs but n={n} "
            f"needs {n_pos}. Maximum n at positive_fraction={positive_fraction} is "
            f"{max_pairs(positive_fraction)}.")
    rng.shuffle(pool)
    positives = pool[:n_pos]
    seen = {_key(a, b) for a, b in positives}

    negatives = []
    for category, count in _largest_remainder(NEGATIVE_QUOTAS, n_neg).items():
        draw, got, attempts = _DRAWERS[category], 0, 0
        while got < count:
            attempts += 1
            if attempts > 200 * count + 1000:
                raise ValueError(f"hardneg cannot supply {count} distinct '{category}' pairs; lower n.")
            a, b = draw(rng)
            k = _key(a, b)
            if k in seen:
                continue
            seen.add(k)
            negatives.append((a, b, 0, category))
            got += 1

    rows = [(a, b, 1, "paraphrase") for a, b in positives] + negatives
    rng.shuffle(rows)
    out = []
    for a, b, label, category in rows:
        if rng.random() < 0.5:
            a, b = b, a
        if rng.random() < casual_prob:
            b = _casual(b)
        out.append((a, b, label, category))
    return out
