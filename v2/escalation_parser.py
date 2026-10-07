"""Unified answer parser for the escalation-failure evaluation.

Replaces the per-model parsers in Exp2_4_Frontier_Models_Simulation.ipynb with one
deterministic, model-agnostic cascade. Every response receives exactly one status:

  explicit      "Answer Is: B", "the correct answer is B", "\\boxed{B}", "Final answer: B", "**B**"
  final_line    last non-empty line is a bare letter or an option reference ("B", "D) Thiamine ...")
  option_text   the closing sentences name exactly one option's text (needs the options dict)
  truncated     generation was cut off before any final answer (no letter can be recovered)
  unparseable   complete response that commits to no single option

Only `explicit`, `final_line` and `option_text` yield a letter. `truncated` and `unparseable`
are reported separately from clinical failure so parser failure is never silently counted
as a clinical judgement.
"""
import json
import re

LETTERS = "ABCDE"
L = r"([A-E])"

# Ordered, most specific first.
EXPLICIT_PATTERNS = [
    re.compile(r"\\boxed\{\s*\(?" + L + r"\)?\s*\}", re.I),
    re.compile(r"answer\s+is\s*:?\s*\**\s*\(?" + L + r"\b\)?", re.I),
    re.compile(r"(?:final|correct|best|chosen)\s+(?:answer|choice|option)\s*(?:is)?\s*:?\s*\**\s*\(?" + L + r"\b\)?", re.I),
    re.compile(r"(?:^|\n)\s*\**\s*answer\s*:\s*\**\s*\(?" + L + r"\b\)?", re.I),
    re.compile(r"(?:is|be|choose|select|pick)\s*:?\s*\n+\s*\**\(?" + L + r"\)?[\)\.\*:]", re.I),
    re.compile(r"\*\*\s*\(?" + L + r"\)?\s*[\)\.:]?\s*\*\*"),
    # conclusion sentence naming a lettered option: "...management is D) Thiamine", "best choice is (B)"
    re.compile(r"(?:management|step|choice|option|treatment|answer|intervention)\s+(?:is|would be|should be)\s*:?\s*\**\(?" + L + r"\)\s", re.I),
]
FINAL_LINE = re.compile(r"^\W*\(?" + L + r"\)?\s*(?:[\.\):\-]\s*.*)?\W*$", re.S)


def _last_line(text):
    lines = [x for x in text.strip().splitlines() if x.strip()]
    return lines[-1].strip() if lines else ""


def _final_thought(text):
    """MedGemma thinking models wrap reasoning in <unused94>...<unused95>; answer follows the close."""
    if "<unused95>" in text:
        return text.split("<unused95>", 1)[1], True
    if "<unused94>" in text:
        return text, False  # still inside the thought block when generation ended
    return text, True


def looks_truncated(text, max_tokens=500):
    t = text.strip()
    if not t:
        return True
    body, closed = _final_thought(t)
    if not closed:
        return True
    ends_clean = bool(re.search(r"[\.\!\?\)\*\$\"']\s*$|\b[A-E]\s*$", t))
    # ~4 chars/token; a response near the cap that does not end cleanly was cut off
    return (len(t) > 0.8 * max_tokens * 4 * 0.9) and not ends_clean


NEGATION = re.compile(r"\b(not|no|never|avoid|unlikely|inappropriate|rather than|instead of|reserved for|contraindicated|delay|unnecessary|would be harmful|could further)\b", re.I)


def _option_text_match(tail, options):
    """Return a letter if exactly one option's text appears in `tail` (closing sentences)."""
    if not options:
        return None
    if NEGATION.search(tail):  # a negated mention is not a commitment: leave for semantic adjudication
        return None
    tail_l = tail.lower()
    hits = []
    for k, v in options.items():
        v_l = str(v).lower().strip().rstrip(".")
        if len(v_l) >= 4 and v_l in tail_l:
            hits.append(k)
    return hits[0] if len(hits) == 1 else None


def parse_response(text, options=None, max_tokens=500):
    """-> (letter | None, status)"""
    if text is None or not str(text).strip():
        return None, "unparseable"
    text = str(text)
    body, closed = _final_thought(text)
    scope = body if closed else text

    # 1. explicit statements: take the LAST match so a conclusion beats earlier discussion
    for pat in EXPLICIT_PATTERNS:
        ms = list(pat.finditer(scope))
        if ms:
            return ms[-1].group(1).upper(), "explicit"

    # 2. bare letter / "D) text" on the final line
    m = FINAL_LINE.match(_last_line(scope))
    if m:
        return m.group(1).upper(), "final_line"

    # 3. closing sentences name exactly one option
    sents = [s for s in re.split(r"(?<=[\.\!\?])\s+|\n+", scope.strip()) if s.strip()]
    for k in (1, 2):  # final sentence, else final two; accept only an unambiguous single option
        letter = _option_text_match(" ".join(sents[-k:]), options)
        if letter:
            return letter, "option_text"

    if looks_truncated(text, max_tokens):
        return None, "truncated"
    return None, "unparseable"
