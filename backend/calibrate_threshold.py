"""
calibrate_threshold.py

Measures real similarity scores from BOTH the original bi-encoder
(cosine similarity) and the cross-encoder re-ranker, against the same
golden set of paraphrase and non-paraphrase question pairs. This lets
you see, with real numbers, whether the cross-encoder actually improves
separation between the two groups - rather than assuming it does.

Run with:  python3 calibrate_threshold.py
"""

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim

bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
cross_encoder = CrossEncoder('cross-encoder/quora-distilroberta-base')

paraphrase_pairs = [
    ("What is the capital of India?", "Which city is the capital of India?"),
    ("Who is the current president of BJP?", "Who leads BJP right now?"),
    ("How do I reset my password?", "I forgot my password, what do I do?"),
    ("What are your business hours?", "When are you open?"),
    ("How much does this cost?", "What is the price of this?"),
    ("Can I get a refund?", "Is a refund possible for this?"),
    ("What is the capital of France?", "Tell me France's capital city"),
    ("How do I contact support?", "What's the best way to reach customer service?"),
    ("Is there a free trial?", "Can I try this for free?"),
    ("What time does the store close?", "When does the shop shut for the day?"),
]

non_paraphrase_pairs = [
    ("What is the capital of India?", "What is the capital of Japan?"),
    ("14+7=?", "20+1="),
    ("How do I reset my password?", "How do I delete my account?"),
    ("What are your business hours?", "What is your refund policy?"),
    ("How much does this cost?", "How do I contact support?"),
    ("Can I get a refund?", "Can I upgrade my plan?"),
    ("What is the capital of France?", "What is the capital of Germany?"),
    ("Is there a free trial?", "Is there a student discount?"),
    ("What time does the store close?", "Where is the store located?"),
    ("Who is the current president of BJP?", "Who is the current CEO of Google?"),
]


def cosine_score(a, b):
    emb = bi_encoder.encode([a, b])
    return float(cos_sim([emb[0]], [emb[1]])[0][0])


def cross_score(a, b):
    return float(cross_encoder.predict([(a, b)])[0])


print(f"{'COSINE':>8}  {'CROSS-ENC':>10}   PAIR")
print("=" * 90)
print("--- PARAPHRASE PAIRS (should score HIGH on both) ---")
para_cos, para_cross = [], []
for a, b in paraphrase_pairs:
    c = cosine_score(a, b)
    x = cross_score(a, b)
    para_cos.append(c)
    para_cross.append(x)
    print(f"{c:8.4f}  {x:10.4f}   {a}  <->  {b}")

print("\n--- NON-PARAPHRASE PAIRS (should score LOW on both) ---")
non_cos, non_cross = [], []
for a, b in non_paraphrase_pairs:
    c = cosine_score(a, b)
    x = cross_score(a, b)
    non_cos.append(c)
    non_cross.append(x)
    print(f"{c:8.4f}  {x:10.4f}   {a}  <->  {b}")

print("\n" + "=" * 90)
print("SUMMARY: separation between paraphrase and non-paraphrase groups")
print("=" * 90)

cos_gap = min(para_cos) - max(non_cos)
cross_gap = min(para_cross) - max(non_cross)

print(f"\nCOSINE (bi-encoder):")
print(f"  Lowest paraphrase score:      {min(para_cos):.4f}")
print(f"  Highest non-paraphrase score: {max(non_cos):.4f}")
print(f"  Separation gap:               {cos_gap:+.4f}  {'(overlap!)' if cos_gap < 0 else '(clean separation)'}")

print(f"\nCROSS-ENCODER (quora-distilroberta-base):")
print(f"  Lowest paraphrase score:      {min(para_cross):.4f}")
print(f"  Highest non-paraphrase score: {max(non_cross):.4f}")
print(f"  Separation gap:               {cross_gap:+.4f}  {'(overlap!)' if cross_gap < 0 else '(clean separation)'}")

print(f"\nImprovement in separation gap: {cross_gap - cos_gap:+.4f}")
print("(Positive = cross-encoder separates the two groups better than cosine alone)")

if cross_gap > 0:
    suggested_auto_hit = round((min(para_cross) + max(non_cross)) / 2 + 0.05, 3)
    suggested_borderline = round(max(non_cross) + 0.02, 3)
else:
    suggested_auto_hit = round(min(para_cross) - 0.02, 3)
    suggested_borderline = round(max(non_cross) + 0.02, 3)

print(f"\nSuggested cross-encoder thresholds for config.json 'default' policy:")
print(f'  "auto_hit": {suggested_auto_hit}, "borderline_low": {suggested_borderline}')
