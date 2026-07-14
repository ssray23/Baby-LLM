"""
Turns a short list of facts into a data/*.txt file in the [Q]/[A] format
your pipeline already expects, automatically generating 3-4 paraphrased
questions per fact.

Why this exists: your retrieval step matches on literal word overlap
(TF-IDF), not meaning. A single phrasing per fact misses any user who asks
it differently. Multiple phrasings per fact is what actually buys you
"world knowledge" coverage, not just adding more distinct facts.

Usage: edit the FACTS list below, then run:
    python3 generate_knowledge.py > data/knowledge_generated.txt
"""

def _cap(s):
    return s[0].upper() + s[1:] if s else s

def _block(question, answer, punctuation="?"):
    q = _cap(question.strip())
    if not q.endswith(("?", ".")):
        q += punctuation
    return f"[Q]: {q}\n[A]: {answer.strip()}"


# --- One template function per fact "shape". Add more if a fact doesn't
# fit these (a definition, a person, a number, a place, a process).
# Commands ("tell me about X") end in a period, real questions end in "?". ---

def definition(subject, answer):
    qs = [
        (f"what is {subject}", "?"),
        (f"can you explain {subject}", "?"),
        (f"tell me about {subject}", "."),
        (f"define {subject}", "."),
    ]
    return [_block(q, answer, p) for q, p in qs]

def person(subject, answer):
    qs = [
        (f"who is {subject}", "?"),
        (f"what did {subject} do", "?"),
        (f"tell me about {subject}", "."),
        (f"what is {subject} known for", "?"),
    ]
    return [_block(q, answer, p) for q, p in qs]

def numeric(phrasings, answer):
    # Numeric facts vary too much in sentence structure to auto-rewrite
    # safely ("how many moons" vs "what year" vs "how fast"), so this one
    # takes fully-written phrasings instead of building them from a stem.
    # Still just 2-3 short lines to write per fact.
    return [_block(q, answer) for q in phrasings]

def location(subject, answer):
    qs = [
        (f"where is {subject}", "?"),
        (f"where is {subject} located", "?"),
        (f"tell me about the location of {subject}", "."),
    ]
    return [_block(q, answer, p) for q, p in qs]

def process(subject, answer):
    qs = [
        (f"how does {subject} work", "?"),
        (f"explain how {subject} works", "."),
        (f"what happens during {subject}", "?"),
    ]
    return [_block(q, answer, p) for q, p in qs]


# --- Add facts here. Keep casing consistent for the same entity across
# every fact you add (always "Mount Kilimanjaro", never mix in
# "mount kilimanjaro") since the tokenizer treats different casings as
# different words. ---

FACTS = [
    (definition, "photosynthesis",
     "Photosynthesis is the process plants use to convert sunlight into energy."),
    (person, "Marie Curie",
     "Marie Curie discovered radioactivity and won two Nobel Prizes."),
    (numeric, ["how many moons does Mars have", "what is the number of moons orbiting Mars"],
     "Mars has two moons, named Phobos and Deimos."),
    (location, "Mount Kilimanjaro",
     "Mount Kilimanjaro is located in Tanzania and is the highest mountain in Africa."),
    (process, "photosynthesis",
     "Photosynthesis works by using chlorophyll to capture sunlight and convert carbon dioxide and water into glucose and oxygen."),
]


def build():
    blocks = []
    for fn, subject, answer in FACTS:
        blocks.extend(fn(subject, answer))
    return "\n\n".join(blocks) + "\n"


if __name__ == "__main__":
    print(build())
