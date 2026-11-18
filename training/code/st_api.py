"""Per-answer frontier judge through the Anthropic Messages API, used as an RL reward (st_grpo.py --reward api).
One call per NEW (question, answer) pair; verdicts are cached per run. The judge sees only the question and the proposed answer
(no reference answer), writes at most two short sentences and ends with a line that holds one digit 1-5 (5 = certainly correct
... 1 = certainly wrong); the trainer accepts an answer at level >= --api_yes (default 4), the same rule as the earlier batch
measurements. Only a digit that stands alone on the final line counts; a reply without one is level 3 (rejected) and is counted
in `noparse`. A pair for which the API returns "Output blocked by content filtering policy" (HTTP 400; seen for about 1 pair in
1,000) is also level 3 and is counted in `blocked` and in `errors`. max_tokens is 800: when the judge is unsure it writes far more
than two sentences, and a limit of 200 cut 5.5% of 24,340 replies before the digit (all of 60 such pairs end with a digit at 700). (A format test on 2026-11-12 showed why: with up-front thinking off the model reasons in its visible reply before
the digit, a 16-token limit cut 74 of 120 replies, and reading "the first digit" would have picked digits out of years.)
The API key is read once from Secret Manager into memory (never printed, never written to disk, never put in the environment).
Pure standard library, so it can be tested without a GPU.

Request format (official docs read on 2026-11-12: models overview, Messages "create" reference, versioning, migration to
2026-10-21, authentication, thinking pages on platform.claude.com):
  - header `anthropic-version: 2026-10-21`; key as `Authorization: Bearer <key>` (x-api-key is the legacy form);
  - `temperature` / `top_p` / `top_k` are no longer accepted (400 error), so none is sent: the judge samples at the default;
  - Claude Sonnet 5.5 thinks before answering by default and thinking counts toward max_tokens; `"thinking": {"type":
    "between_tools"}` is the documented way to switch up-front thinking off for a request without tools (pass thinking=None to
    leave the model's default);
  - an API key that is not scoped to a workspace needs the header `anthropic-workspace-id` (taken from the environment variable
    ANTHROPIC_WORKSPACE_ID when it is set; this id is not a secret)."""
import json, subprocess, time, urllib.request, urllib.error, re, os
from concurrent.futures import ThreadPoolExecutor

SYSTEM = ("You are a closed-book grader of short answers to trivia questions. Decide from your own knowledge only whether the proposed "
          "answer is a correct answer to the question. Small spelling variants, extra words, or a full sentence that contains the right "
          "answer are fine. A different person, place, thing or number, or a reply that is not an answer, is wrong. "
          "First write at most two short sentences of reasoning. Then end with a final line that contains only one digit: "
          "5 = certainly correct, 4 = probably correct, 3 = cannot tell, 2 = probably wrong, 1 = certainly wrong.")
FINAL = re.compile(r"(?:^|\n)\s*([1-5])\s*\.?\s*$")     # a digit alone on the final line


def read_key(secret, project):
    env = {k: v for k, v in os.environ.items() if k != "CLOUDSDK_AUTH_ACCESS_TOKEN"}
    out = subprocess.run(["gcloud", "secrets", "versions", "access", "latest", "--secret", secret, "--project", project],
                         env=env, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError("could not read the API key secret (is the secret-access permission in place?)")
    return out.stdout.strip()


class ApiJudge:
    def __init__(self, key, model, url="https://api.anthropic.com/v1/messages", version="2026-10-21", workers=8, max_retries=8, timeout=60,
                 thinking="between_tools", max_tokens=800):
        self._key = key; self.model = model; self.url = url; self.version = version
        self.workers = workers; self.max_retries = max_retries; self.timeout = timeout
        self.thinking = thinking; self.max_tokens = max_tokens
        self.workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
        self.calls = 0; self.errors = 0; self.in_tokens = 0; self.out_tokens = 0; self.noparse = 0; self.blocked = 0

    def _safe(self, s):
        return str(s).replace(self._key, "<key>")[:300]

    def one(self, question, answer):
        """Level 1-5 for one (question, answer) pair; 3 (cannot tell = rejected) after repeated failures."""
        b = {"model": self.model, "max_tokens": self.max_tokens, "system": SYSTEM,
             "messages": [{"role": "user", "content": f"Question: {question}\nProposed answer: {answer}"}]}
        if self.thinking:
            b["thinking"] = {"type": self.thinking}
        body = json.dumps(b).encode()
        headers = {"Authorization": "Bearer " + self._key, "anthropic-version": self.version, "content-type": "application/json"}
        if self.workspace:
            headers["anthropic-workspace-id"] = self.workspace
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(self.url, data=body, method="POST", headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode())
                self.calls += 1
                u = d.get("usage", {}); self.in_tokens += int(u.get("input_tokens", 0)); self.out_tokens += int(u.get("output_tokens", 0))
                text = "".join(c.get("text", "") for c in d.get("content", []) if c.get("type") == "text")
                m = FINAL.search(text.strip())
                if not m:
                    self.noparse += 1
                return int(m.group(1)) if m else 3
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404):          # wrong request, key or model: retrying cannot help
                    self.errors += 1
                    try:
                        msg = json.loads(e.read().decode()).get("error", {}); msg = f"{msg.get('type', '')}: {msg.get('message', '')}"
                    except Exception:
                        msg = ""
                    if e.code == 400 and "content filter" in msg.lower():   # the API refuses to answer for this one pair
                        self.blocked += 1                                    # ("Output blocked by content filtering policy"):
                        return 3                                             # counted, and judged "cannot tell" (rejected)
                    raise RuntimeError(f"API refused the request (HTTP {e.code}) {self._safe(msg)}")
                time.sleep(min(60, 2 ** attempt))
            except Exception:
                time.sleep(min(60, 2 ** attempt))
        self.errors += 1
        return 3

    def many(self, pairs):
        """pairs: list of (question, answer). Returns the list of levels, in order."""
        if not pairs:
            return []
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda p: self.one(p[0], p[1]), pairs))
