"""Shared helpers for the judge-filtered self-training experiment (GSM8K and TriviaQA)."""
import json, re, string

GSM_PROMPT = ("Solve the following math problem. Reason step by step, then give the final answer on the last line in the "
              "form '#### <number>'.\n\n{q}")
TQ_PROMPT = ("Answer the following trivia question. Reply with only the answer (a few words), no explanation.\n\n"
             "Question: {q}")
GSM_JUDGE = ("Below is a math problem and a proposed solution. Decide whether the proposed solution's final answer is "
             "correct.\n\nProblem:\n{q}\n\nProposed solution:\n{sol}\n\nIs the final answer correct? Reply with only one "
             "word: Yes or No.")
TQ_JUDGE = ("Here is a trivia question and a proposed answer. Is the proposed answer correct?\n\nQuestion: {q}\n"
            "Proposed answer: {a}\n\nReply with only one word: Yes or No.")

TQ_JUDGE_STRICT = ("Here is a trivia question and a proposed answer. Be strict: answer Yes only if the proposed answer is fully correct and you are "
                   "certain of it; otherwise answer No.\n\nQuestion: {q}\nProposed answer: {a}\n\nReply with only one word: Yes or No.")   # 2026-11-15

TQ_PROMPT_IDK = ("Answer the following trivia question. Reply with only the answer (a few words), no explanation. "
                 "If you do not know the answer, reply exactly: I don't know.\n\nQuestion: {q}")
TQ_JUDGE_REF = ("Here is a trivia question, its reference answer and a proposed answer. Is the proposed answer correct?\n\n"
                "Question: {q}\nReference answer: {ref}\nProposed answer: {a}\n\nReply with only one word: Yes or No.")
TQ_JUDGE_OWN = ("Here is a trivia question and a proposed answer. Is the proposed answer correct?\n\nQuestion: {q}\n"
                "Your own answer to this question: {own}\nProposed answer: {a}\n\nReply with only one word: Yes or No.")
TQ_PROMPT_FLAG = ("Answer the following trivia question. Reply with only the answer (a few words), no explanation. "
                  "If you are not sure that your answer is correct, add (unsure) after it.\n\nQuestion: {q}")
FLAG_RE = re.compile(r"[\s,;:.\-]*[\(\[]?\s*unsure\s*[\)\]]?\s*[.!]?\s*$", re.I)
IDK_RE = re.compile(r"^(i don t know|i do not know|i dont know|idk|i m not sure|not sure|i cannot answer|cannot answer|no idea|i am not sure)\b|^unknown$")


def is_idk(x):
    """x: normalized answer (parse_tq output). True for an explicit abstention."""
    return bool(x) and bool(IDK_RE.match(x))


NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
BOX = re.compile(r"\\boxed\{([^{}]*)\}")
THINK = re.compile(r"<think>.*?</think>", re.S)


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def write_jsonl(p, rows):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def strip_think(text):
    t = THINK.sub("", text)
    if "</think>" in t:                      # an unclosed or leading think block
        t = t.split("</think>")[-1]
    return t.strip()


def to_float(s):
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_gsm(text):
    """Number after the last '####', else the last \\boxed{...} number, else the last number in the text."""
    text = strip_think(text)
    if "####" in text:
        m = NUM.search(text.split("####")[-1])
        if m:
            return to_float(m.group())
    b = BOX.findall(text)
    if b:
        m = NUM.search(b[-1])
        if m:
            return to_float(m.group())
    nums = NUM.findall(text)
    return to_float(nums[-1]) if nums else None


def normalize(s):
    s = s.lower()
    s = "".join(ch if ch not in set(string.punctuation) else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tq_line(text):
    t = strip_think(text).split("\n")[0].strip()
    return t[:200]


def split_flag(text):
    """Answer-then-flag format: returns (first line without a trailing '(unsure)' flag, flagged?)."""
    t = tq_line(text)
    m = FLAG_RE.search(t)
    if not m:
        return t, False
    return t[:m.start()].strip(), True


def parse_tq(text):
    t = tq_line(text)
    t = re.sub(r"^(final answer|answer)\s*[:\-]\s*", "", t, flags=re.I)
    return normalize(t.strip().strip("*").strip())


def tq_correct(x, aliases):
    """Alias match, alias (>= 3 chars) as a whole-word substring, or the looser rules used in the pilot (SJ_LENIENT)."""
    if not x:
        return False
    if x in aliases or any(len(t) >= 3 and re.search(r"\b" + re.escape(t) + r"\b", x) for t in aliases):
        return True
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[a-z]{1,12}", x)
    y = m.group(1) if m else x
    return (y in aliases or any(t and re.search(r"\b" + re.escape(t) + r"\b", y) for t in aliases)
            or (len(y) >= 4 and any(re.search(r"\b" + re.escape(y) + r"\b", t) for t in aliases)))


def gsm_correct(x, gold):
    return x is not None and abs(x - gold) < 1e-4


def is_vlm(model):
    """True when the checkpoint is a vision-language model (text-only use needs the image/video inputs switched off)."""
    try:
        from huggingface_hub import hf_hub_download
        c = json.load(open(hf_hub_download(model, "config.json")))
        return any("ConditionalGeneration" in a for a in c.get("architectures", []))
    except Exception:
        return False


_AUDIT = None


def tq_ok(x, idx, aliases):
    """tq_correct, plus (item, answer) pairs that a reference-aware grader marked correct (file in $TQ_AUDIT)."""
    global _AUDIT
    if _AUDIT is None:
        import json as _j, os as _o
        _AUDIT = set()
        f = _o.environ.get("TQ_AUDIT")
        if f and _o.path.exists(f):
            for l in open(f):
                r = _j.loads(l)
                if r.get("verdict") == "correct":
                    _AUDIT.add((r["idx"], r["answer"]))
    return tq_correct(x, aliases) or (x is not None and (idx, x) in _AUDIT)


def tq_strict(x, aliases):
    """Strict TriviaQA match for RL rewards: the normalized answer equals an alias, or contains one as whole words with
    at most two extra words (so a hedged list of guesses does not count)."""
    if not x:
        return False
    if x in aliases:
        return True
    n = len(x.split())
    for t in aliases:
        if len(t) >= 3 and n <= len(t.split()) + 2 and re.search(r"\b" + re.escape(t) + r"\b", x):
            return True
    return False
