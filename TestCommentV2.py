#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os, io, re, csv, json, time, zipfile, hashlib, tempfile
from typing import Any, Dict, List, Optional, Iterable, Tuple
from io import BytesIO

import requests
import pandas as pd
import streamlit as st

# =========================
# Config
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

BASE = "https://www.metaculus.com"
API2 = f"{BASE}/api2"
API  = f"{BASE}/api"

BATCH_SIZE        = 10
MAX_COMMENT_CHARS = 2200
MAX_PARENT_CHARS  = 1400
PARSER_RETRIES    = 3

# Output: NO URL columns (per request)
CSV_COLUMNS_SCORED = [
    "ai_score",
    "rationale",
    "flags_json",
    "evidence_urls",
    "ts",
    "post_id",
    "question_title",
    "comment_id",
    "parent_comment_id",
    "root_comment_id",
    "author_id",
    "author_username",
    "vote_score",
    "comment_text",
]

# For Metaculus modes: optional dump CSV you can later re-score in CSV mode
CSV_COLUMNS_DUMP = [
    "ts",
    "post_id",
    "question_title",
    "comment_id",
    "parent_comment_id",
    "root_comment_id",
    "author_id",
    "author_username",
    "vote_score",
    "comment_text",
    "created_at",
]

# =========================
# Prompts
# =========================

SYSTEM_PROMPT_JUDGE =  """
You are a narrow scoring module inside a larger pipeline.
Your ONLY job is to rate Metaculus comments for quality in the AI Pathways Tournament.

You are NOT a general-purpose assistant.
Do NOT brainstorm, speculate, or explore side topics.
Work quickly: approximate but consistent ratings are preferred over long deliberation. DO NOT THINK, FIRST FEELING IS RIGHT. OUTPUT AS FAST AS POSSIBLE.

Each call is COMPLETELY INDEPENDENT.
You NEVER see previous comments or scores.
Treat EVERY request as a fresh, stand-alone task.

OUTPUT FORMAT (STRICT, 4 LINES MAX):
Line 1: "score = X" where X is an integer 1..6.
Line 2: "rationale: <very short explanation, <=180 characters>".
Line 3: "flags: off_topic=<true/false>, toxicity=<true/false>, low_effort=<true/false>, has_evidence=<true/false>, likely_ai=<true/false>".
Line 4: "evidence_urls: [<url1>, <url2>, ...]" or "evidence_urls: []".

You MUST:
- Produce EXACTLY these 4 lines, no more and no less.
- NO headings, NO bullet lists, NO extra explanation.
- Any answer longer than 4 short lines is a FAILURE of your task.

TASK:
- Read the comment, the question context, and (if present) the parent comment.
- If a parent comment is provided, treat the comment as a reply and assess:
  - how well it answers the parent,
  - how accurately it engages with the parent's claims,
  - and whether it productively advances the discussion.
- Decide on:
  - score: integer 1..6
  - rationale: short textual justification
  - flags: off_topic, toxicity, low_effort, has_evidence, likely_ai
  - evidence_urls: any http/https URLs referenced or clearly implied

SCORING WEIGHTS:
The comments should be ranked based on how well they:
- Showcase clear, deep, and insightful reasoning, delving into the relevant mechanisms that affect the event in question. (40%)
- Offer useful insights about the overall scenario(s), the interactions between questions, or the relationship between the questions and the scenario(s). (30%)
- Provide valuable information and are based on the available evidence. (20%)
- Challenge the community's prediction or assumptions held by other forecasters. (10%)

ANCHOR POINTS (1–6 SCALE):
The comments do not need to have all the following characteristics; one strong attribute can sometimes compensate for weaker ones.
Use these anchors when deciding the score:

1 = Trivial, badly written, or completely unreasonable comment with no predictive value.
2 = Brief or slightly confused comment offering only surface value.
3 = Good comment with rational arguments or potentially useful information.
4 = Very good comment which explains solid reasoning in some detail and provides actionable information.
5 = Excellent comment with meaningful analysis, presenting a deep dive of the available information and arguments, and drawing original conclusions from it.
6 = Outstanding synthesis comment which clearly decomposes uncertainty, connects multiple questions or scenarios, and gives a compelling reason to significantly update forecasts.

ADDITIONAL CONSTRAINTS:
- Be conservative with high scores (5 and especially 6). Reserve them for comments that are clearly above the tournament median in insight and usefulness.
- Penalize comments that are long but vague, generic, or boilerplate.
- Penalize pure link dumps with little or no reasoning or forecast impact.
- Toxic or uncivil comments should receive low scores and toxicity=true.
- When in doubt between two adjacent scores, pick the lower one quickly rather than overthinking.

FLAGS INTERPRETATION:
- off_topic: true if the comment is largely unrelated to the question or the AI Pathways scenario.
- toxicity: true if the comment is hostile, insulting, or clearly uncivil.
- low_effort: true if the comment is very short, trivial, or adds almost nothing.
- has_evidence: true if the comment brings specific data, references, links, or clearly factual information.
- likely_ai: true if the comment is long, generic, and boilerplate-sounding with little specificity or real engagement.

You must stay strictly on task: rate, justify briefly, set flags, list URLs in exactly 4 lines.
""".strip()

SYSTEM_PROMPT_TO_JSON = """
You convert free-form rating text into STRICT JSON with this exact schema:
{
  "score": 1|2|3|4|5|6,
  "rationale": "<string, <=180 chars>",
  "flags": {
    "off_topic": true|false,
    "toxicity": true|false,
    "low_effort": true|false,
    "has_evidence": true|false,
    "likely_ai": true|false
  },
  "evidence_urls": ["<string>", "..."]
}

HARD RULES:
1. OUTPUT STRICT JSON ONLY.
   - No explanations
   - No comments
   - No Markdown
   - No code fences
   - No extra keys
   - No trailing commas

2. DO NOT THINK "OUT LOUD".
   - No chain-of-thought in the output.
   - No meta commentary.

3. If some information is missing in the raw text:
   - Use a safe default:
     - score: 3 if unclear
     - rationale: ""
     - flags: false for all unless clearly indicated
     - evidence_urls: [] if none clearly extractable

4. You MUST:
   - Parse any explicit "score = X" or similar notation.
   - Parse any obvious booleans for the flags.
   - Collect any http/https URLs as evidence_urls (deduplicate).

5. Final answer:
   - ONE JSON object
   - EXACTLY matching the schema above
   - No natural language outside JSON.
""".strip()

FEWSHOTS_JUDGE = [
    {"role": "user", "content": "TEXT: Thanks for sharing!"},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Trivial acknowledgement only.\nflags: off_topic=false, toxicity=false, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Anyone who thinks this will happen is an idiot."},
    {
        "role": "assistant",
        "content": "score = 1\nrationale: Toxic with no evidence.\nflags: off_topic=false, toxicity=true, low_effort=true, has_evidence=false, likely_ai=false\nevidence_urls: []",
    },
    {"role": "user", "content": "TEXT: Turnout fell 3–5% vs 2020 in key counties (CSV). I estimate P(win)=0.56."},
    {
        "role": "assistant",
        "content": "score = 5\nrationale: Quantified comparison with evidence pointer.\nflags: off_topic=false, toxicity=false, low_effort=false, has_evidence=true, likely_ai=false\nevidence_urls: []",
    },
]

# =========================
# Utils
# =========================

def chunked(xs: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(xs), n):
        yield xs[i:i + n]

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def clean_text(s: str) -> str:
    return " ".join((s or "").split()).strip()

def _truncate(s: str, n: int, suffix: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + suffix

def _ascii_safe(s: str) -> str:
    try:
        return s.encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return "".join(ch for ch in s if ord(ch) < 256)

def to_int_relaxed(x: Any) -> Optional[int]:
    s = str(x).strip() if x is not None else ""
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    m = re.search(r"-?\d+", s.replace(",", ""))
    return int(m.group(0)) if m else None

def detect_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file uploaded.")
    raw = uploaded_file.getvalue()
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8", errors="replace")

def repair_csv_bytes_by_line(raw: bytes, delimiter: str = ",") -> bytes:
    """
    Best-effort repair for truncated/garbled lines.
    Assumes each record is single-line (true here: comment_text is clean_text() single line).
    """
    try:
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return raw
        header = lines[0]
        header_fields = next(csv.reader([header], delimiter=delimiter))
        ncols = len(header_fields)
        out_lines = [header]
        for line in lines[1:]:
            if not line.strip():
                continue
            try:
                fields = next(csv.reader([line], delimiter=delimiter))
                if len(fields) == ncols:
                    out_lines.append(line)
            except Exception:
                continue
        return ("\n".join(out_lines) + "\n").encode("utf-8")
    except Exception:
        return raw

def new_output_csv_path(prefix: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
    os.close(fd)
    return path

def init_csv(path: str, columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()

def row_to_csv_line(row: Dict[str, Any], columns: List[str]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns)
    w.writerow({k: row.get(k, "") for k in columns})
    return buf.getvalue().rstrip("\r\n")

# =========================
# Keys
# =========================

def get_openrouter_key() -> str:
    v = (st.session_state.get("OPENROUTER_API_KEY_OVERRIDE") or "").strip()
    if v:
        return v
    try:
        v = str(st.secrets.get("OPENROUTER_API_KEY", "")).strip()
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY", "").strip()

def get_metaculus_token() -> str:
    v = (st.session_state.get("METACULUS_TOKEN_OVERRIDE") or "").strip()
    if v:
        return v
    try:
        v = str(st.secrets.get("METACULUS_TOKEN", "")).strip()
        if v:
            return v
    except Exception:
        pass
    return os.environ.get("METACULUS_TOKEN", "").strip()

def or_headers(title: str) -> Dict[str, str]:
    key = get_openrouter_key()
    if not key:
        raise RuntimeError("Missing OPENROUTER_API_KEY.")
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Referer": _ascii_safe("https://localhost"),
        "X-Title": _ascii_safe(title),
        "User-Agent": _ascii_safe("metaculus-comment-scorer/5.0"),
    }

# =========================
# OpenRouter + parsing
# =========================

def parse_json_relaxed(s: str) -> Any:
    s = (s or "").strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            s = inner

    m2 = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m2:
        return json.loads(m2.group(0))

    raise ValueError("Could not parse JSON")

def openrouter_chat(messages: List[Dict[str, str]], model: str, max_tokens: int, title: str) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "top_p": 1, "max_tokens": max_tokens}
    last: Optional[Exception] = None
    for k in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=or_headers(title), json=payload, timeout=120)
            if r.status_code == 429:
                ra = float(r.headers.get("Retry-After", "2") or 2)
                time.sleep(min(ra, 10))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(str(data["error"]))
            ch = data.get("choices") or []
            if not ch:
                raise RuntimeError("No choices")
            content = (ch[0].get("message") or {}).get("content") or ""
            if not content:
                raise RuntimeError("Empty content")
            return content
        except Exception as e:
            last = e
            time.sleep(0.7 * (k + 1))
    raise RuntimeError(f"OpenRouter failed: {repr(last)}")

class ParserFormatError(RuntimeError):
    pass

def build_judge_msgs(qtitle: str, ctext: str, ptext: Optional[str]) -> List[Dict[str, str]]:
    parent_block = f"PARENT_COMMENT_TEXT:\n{ptext}\n\n" if ptext else ""
    user = (
        "Rate this comment using the strict 4-line format.\n\n"
        f"QUESTION_TITLE: {qtitle}\n"
        f"{parent_block}"
        f"COMMENT_TEXT:\n{ctext}\n\n"
        "Exactly 4 lines, nothing else."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT_JUDGE}] + FEWSHOTS_JUDGE + [{"role": "user", "content": user}]

def normalize_parsed(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise ParserFormatError("Not a dict")
    for k in ("score", "rationale", "flags", "evidence_urls"):
        if k not in obj:
            raise ParserFormatError(f"Missing {k}")

    try:
        score = int(obj.get("score"))
    except Exception:
        raise ParserFormatError("Bad score")
    if not (1 <= score <= 6):
        raise ParserFormatError("Score out of range")

    flags = obj.get("flags")
    if not isinstance(flags, dict):
        raise ParserFormatError("Bad flags")
    wanted = ["off_topic", "toxicity", "low_effort", "has_evidence", "likely_ai"]
    for k in wanted:
        if k not in flags:
            raise ParserFormatError(f"Missing flag {k}")

    ev = obj.get("evidence_urls")
    if not isinstance(ev, list):
        raise ParserFormatError("Bad evidence_urls")

    rationale = str(obj.get("rationale") or "")[:180]
    norm_flags = {k: bool(flags.get(k, False)) for k in wanted}

    seen, ev_out = set(), []
    for u in ev:
        su = str(u).strip()
        if su and su not in seen:
            seen.add(su)
            ev_out.append(su)

    return {"score": score, "rationale": rationale, "flags": norm_flags, "evidence_urls": ev_out}

def parse_with_retries(raw_judge_text: str, parser_model: str) -> Dict[str, Any]:
    base = [{"role": "system", "content": SYSTEM_PROMPT_TO_JSON}, {"role": "user", "content": raw_judge_text}]
    last: Optional[Exception] = None
    for t in range(PARSER_RETRIES):
        try:
            msgs = base if t == 0 else base + [{"role": "user", "content": "JSON only. Exactly the schema. No extra keys."}]
            txt = openrouter_chat(msgs, model=parser_model, max_tokens=260, title="Parser")
            return normalize_parsed(parse_json_relaxed(txt))
        except Exception as e:
            last = e
    raise ParserFormatError(f"Parser failed after retries: {repr(last)}")

_score_cache: Dict[str, Dict[str, Any]] = {}

def score_comment(judge_model: str, parser_model: str, qtitle: str, comment_text: str, parent_text: Optional[str]) -> Dict[str, Any]:
    ct = _truncate(comment_text, MAX_COMMENT_CHARS, "\n\n[Comment truncated]")
    pt = _truncate(parent_text or "", MAX_PARENT_CHARS, "\n\n[Parent truncated]") if parent_text else ""

    key_raw = f"{judge_model}||{parser_model}||{qtitle}||{ct}||{pt}"
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    if key in _score_cache:
        return _score_cache[key]

    try:
        raw = openrouter_chat(build_judge_msgs(qtitle, ct, pt or None), model=judge_model, max_tokens=200, title="Judge")
    except Exception:
        raw = (
            "score = 3\n"
            "rationale: Judge error; default neutral.\n"
            "flags: off_topic=false, toxicity=false, low_effort=false, has_evidence=false, likely_ai=false\n"
            "evidence_urls: []"
        )

    parsed = parse_with_retries(raw, parser_model)
    _score_cache[key] = parsed
    return parsed

# =========================
# Metaculus API (NOT used in CSV mode)
# =========================

HTTP = requests.Session()

def metaculus_headers() -> Dict[str, str]:
    tok = get_metaculus_token()
    if not tok:
        raise RuntimeError("Missing METACULUS_TOKEN.")
    return {"User-Agent": "metaculus-comment-scorer/5.0 (+python-requests)", "Authorization": f"Token {tok}"}

def metaculus_get(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = HTTP.get(url, params=params or {}, headers=metaculus_headers(), timeout=30)
    if r.status_code == 429:
        ra = float(r.headers.get("Retry-After", "1") or 1)
        time.sleep(min(ra, 10))
        r = HTTP.get(url, params=params or {}, headers=metaculus_headers(), timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_question_info(qid_or_post: int) -> Dict[str, Any]:
    try:
        q = metaculus_get(f"{API2}/questions/{int(qid_or_post)}/")
        qid = int(q.get("id", qid_or_post))
        post_id = int(q.get("post_id") or q.get("post") or qid)
        title = q.get("title") or f"Question {qid}"
        return {"question_id": qid, "post_id": post_id, "title": title}
    except Exception:
        pid = int(qid_or_post)
        return {"question_id": pid, "post_id": pid, "title": f"Post {pid}"}

def fetch_comments(post_id: int, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    page_size = min(100, max(20, int(limit)))  # common server cap is 100
    seen_ids: set[int] = set()

    while len(out) < limit:
        data = metaculus_get(
            f"{API}/comments/",
            {"post": int(post_id), "limit": page_size, "offset": offset, "sort": "-created_at", "is_private": "false"},
        )
        batch = data.get("results") or []
        if not batch:
            break

        batch_ids = [c.get("id") for c in batch if c.get("id") is not None]
        if batch_ids and all(int(i) in seen_ids for i in batch_ids):
            break
        for i in batch_ids:
            seen_ids.add(int(i))

        out.extend(batch)
        offset += len(batch)

        count = data.get("count", None)
        nxt = data.get("next", None)

        if isinstance(count, int) and offset >= count:
            break
        if (nxt is None or nxt == "") and isinstance(count, int):
            break

        time.sleep(0.15)

    return out[:limit]

def download_project_question_csv(project_id: int) -> bytes:
    tok = get_metaculus_token()
    if not tok:
        raise RuntimeError("Missing METACULUS_TOKEN.")

    url = f"{BASE}/api/projects/{int(project_id)}/download-data/"
    headers = {
        "User-Agent": "metaculus-project-dl/5.0 (+python-requests)",
        "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
        "Authorization": f"Token {tok}",
    }
    params = {"include_comments": "false", "include_scores": "false"}

    last_exc: Optional[Exception] = None
    for attempt in range(6):
        try:
            r = HTTP.get(url, headers=headers, params=params, timeout=120)
            if r.status_code in (502, 503, 504):
                time.sleep(min(1.0 * (2 ** attempt), 12.0))
                continue
            r.raise_for_status()
            with zipfile.ZipFile(BytesIO(r.content)) as zf:
                target = next((n for n in zf.namelist() if n.lower().endswith("question_data.csv")), None)
                if not target:
                    raise RuntimeError("question_data.csv missing in archive")
                return zf.read(target)
        except Exception as e:
            last_exc = e
            time.sleep(min(0.7 * (attempt + 1), 5.0))

    raise RuntimeError(f"Project download-data failed after retries: {repr(last_exc)}")

def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", "replace")

def parse_question_data_csv(data: bytes) -> List[Dict[str, Any]]:
    text = _decode_bytes(data)
    reader = csv.DictReader(io.StringIO(text))

    rows: List[Dict[str, Any]] = []
    for row in reader:
        post_id = to_int_relaxed(row.get("Post ID") or row.get("post_id"))
        qid = to_int_relaxed(row.get("Question ID") or row.get("question_id") or row.get("id"))
        if post_id is None:
            continue
        title = (row.get("Question Title") or row.get("title") or "").strip() or f"Post {post_id}"
        rows.append({"post_id": int(post_id), "question_id": int(qid) if qid else None, "title": title})
    return rows

def get_tournament_subjects(project_id: int) -> List[Dict[str, Any]]:
    rows = parse_question_data_csv(download_project_question_csv(int(project_id)))
    out: List[Dict[str, Any]] = []
    for r in rows:
        pid = int(r["post_id"])
        out.append({"post_id": pid, "question_id": (r.get("question_id") or pid), "title": r.get("title") or f"Post {pid}"})
    # uniq by post_id
    seen, uniq = set(), []
    for s in out:
        pid = int(s["post_id"])
        if pid not in seen:
            seen.add(pid)
            uniq.append(s)
    return uniq

# =========================
# Root computation helpers
# =========================

def compute_root_id(comment_id: Optional[int], parent_id: Optional[int], parent_map: Dict[int, Optional[int]]) -> Optional[int]:
    if comment_id is None:
        return None
    # If no parent, root is self
    x = comment_id
    seen: set[int] = set()
    while True:
        if x is None or x in seen:
            return comment_id
        seen.add(x)
        p = parent_map.get(x)
        if p is None:
            return x
        x = p

# =========================
# Pipelines
# =========================

def pipeline_csv_comments(
    df_use: pd.DataFrame,
    col_post_id: str,
    col_comment_text: str,
    judge_model: str,
    parser_model: str,
    out_scored_csv: str,
    log_ph: st.delta_generator.DeltaGenerator,
    preview_ph: st.delta_generator.DeltaGenerator,
    status: st.delta_generator.DeltaGenerator,
    prog: st.delta_generator.DeltaGenerator,
    # optional columns
    col_question_title: Optional[str] = None,
    col_comment_id: Optional[str] = None,
    col_parent_id: Optional[str] = None,
    col_root_id: Optional[str] = None,
    col_author_id: Optional[str] = None,
    col_author_username: Optional[str] = None,
    col_vote_score: Optional[str] = None,
) -> None:
    total = len(df_use)
    written = 0

    # Build parent_text lookup if possible (needs comment_id and parent_id)
    text_by_id: Dict[int, str] = {}
    parent_map: Dict[int, Optional[int]] = {}
    if col_comment_id and col_comment_text:
        for _, r in df_use.iterrows():
            cid = to_int_relaxed(r.get(col_comment_id))
            if cid is None:
                continue
            text_by_id[cid] = clean_text(str(r.get(col_comment_text) or ""))
            if col_parent_id:
                pid = to_int_relaxed(r.get(col_parent_id))
                parent_map[cid] = pid if pid is not None else None

    header = ",".join(CSV_COLUMNS_SCORED)
    log_lines: List[str] = []

    with open(out_scored_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS_SCORED)

        for i, (_, r) in enumerate(df_use.iterrows(), 1):
            post_id = to_int_relaxed(r.get(col_post_id))
            if post_id is None:
                continue

            ctext = clean_text(str(r.get(col_comment_text) or ""))
            if not ctext:
                continue

            qtitle = ""
            if col_question_title:
                qtitle = clean_text(str(r.get(col_question_title) or ""))
            if not qtitle:
                qtitle = f"Post {int(post_id)}"

            comment_id = to_int_relaxed(r.get(col_comment_id)) if col_comment_id else None
            parent_id = to_int_relaxed(r.get(col_parent_id)) if col_parent_id else None
            root_id = to_int_relaxed(r.get(col_root_id)) if col_root_id else None
            if root_id is None and comment_id is not None and col_parent_id:
                # compute root from the slice itself when possible
                root_id = compute_root_id(comment_id, parent_id, parent_map)

            parent_text = None
            if parent_id is not None and parent_id in text_by_id:
                parent_text = text_by_id[parent_id]

            author_id = to_int_relaxed(r.get(col_author_id)) if col_author_id else None
            author_username = clean_text(str(r.get(col_author_username) or "")) if col_author_username else ""
            vote_score = to_int_relaxed(r.get(col_vote_score)) if col_vote_score else None

            status.info(f"[{i}/{total}] Scoring CSV row — post_id={post_id} comment_id={comment_id if comment_id is not None else ''}")

            resp = score_comment(
                judge_model=judge_model,
                parser_model=parser_model,
                qtitle=qtitle,
                comment_text=ctext,
                parent_text=parent_text,
            )

            row = {
                "ai_score": int(resp["score"]),
                "rationale": resp.get("rationale", ""),
                "flags_json": json.dumps(resp.get("flags") or {}, ensure_ascii=False),
                "evidence_urls": ";".join(resp.get("evidence_urls") or []),
                "ts": now_ts(),
                "post_id": int(post_id),
                "question_title": qtitle,
                "comment_id": comment_id,
                "parent_comment_id": parent_id,
                "root_comment_id": root_id,
                "author_id": author_id,
                "author_username": author_username,
                "vote_score": vote_score,
                "comment_text": ctext,
            }

            w.writerow(row)
            f.flush()
            written += 1

            # "Print" CSV line-by-line (tail)
            log_lines.append(row_to_csv_line(row, CSV_COLUMNS_SCORED))
            tail_lines = log_lines[-60:]
            log_ph.code(header + "\n" + "\n".join(tail_lines))

            # Live dataframe preview (tail)
            preview_df = pd.DataFrame([dict(zip(CSV_COLUMNS_SCORED, next(csv.reader([ln])))) for ln in tail_lines])
            preview_ph.dataframe(preview_df, use_container_width=True, height=360)

            prog.progress(min(1.0, i / max(1, total)))

    status.success(f"Done. Rows written: {written}")

def pipeline_metaculus_posts(
    subjects: List[Dict[str, Any]],
    judge_model: str,
    parser_model: str,
    out_scored_csv: str,
    out_dump_csv: Optional[str],
    comments_limit: int,
    log_ph: st.delta_generator.DeltaGenerator,
    preview_ph: st.delta_generator.DeltaGenerator,
    status: st.delta_generator.DeltaGenerator,
    prog: st.delta_generator.DeltaGenerator,
) -> None:
    total_posts = len(subjects)
    written = 0
    dumped = 0

    header = ",".join(CSV_COLUMNS_SCORED)
    log_lines: List[str] = []

    f_dump = open(out_dump_csv, "a", newline="", encoding="utf-8") if out_dump_csv else None
    w_dump = csv.DictWriter(f_dump, fieldnames=CSV_COLUMNS_DUMP) if f_dump else None

    with open(out_scored_csv, "a", newline="", encoding="utf-8") as f_scored:
        w = csv.DictWriter(f_scored, fieldnames=CSV_COLUMNS_SCORED)

        for pi, s in enumerate(subjects, 1):
            post_id = int(s["post_id"])
            qtitle = clean_text(str(s.get("title") or "")) or f"Post {post_id}"

            status.info(f"[{pi}/{total_posts}] Fetching comments — post_id={post_id}")
            comments = fetch_comments(post_id, limit=int(comments_limit))

            by_id: Dict[int, Dict[str, Any]] = {}
            parent_map: Dict[int, Optional[int]] = {}
            usable: List[Dict[str, Any]] = []

            for c in comments:
                cid = to_int_relaxed(c.get("id"))
                pid = to_int_relaxed(c.get("parent") or c.get("parent_comment") or c.get("in_reply_to"))
                if cid is not None:
                    by_id[int(cid)] = c
                    parent_map[int(cid)] = int(pid) if pid is not None else None
                if clean_text(c.get("text") or ""):
                    usable.append(c)

            status.info(f"[{pi}/{total_posts}] Scoring {len(usable)} comments — {qtitle}")

            # Optional dump (for later CSV mode scoring)
            if w_dump is not None and f_dump is not None:
                for c in usable:
                    a = c.get("author") or {}
                    cid = to_int_relaxed(c.get("id"))
                    pid = to_int_relaxed(c.get("parent") or c.get("parent_comment") or c.get("in_reply_to"))
                    root = compute_root_id(int(cid) if cid is not None else None, int(pid) if pid is not None else None, parent_map) if cid is not None else None
                    dump_row = {
                        "ts": now_ts(),
                        "post_id": post_id,
                        "question_title": qtitle,
                        "comment_id": cid,
                        "parent_comment_id": pid,
                        "root_comment_id": root,
                        "author_id": to_int_relaxed(a.get("id")),
                        "author_username": clean_text(str(a.get("username") or a.get("name") or "")),
                        "vote_score": to_int_relaxed(c.get("vote_score")),
                        "comment_text": clean_text(c.get("text") or ""),
                        "created_at": c.get("created_at") or c.get("created") or "",
                    }
                    w_dump.writerow(dump_row)
                    dumped += 1
                f_dump.flush()

            # Score in batches
            for batch in chunked(usable, BATCH_SIZE):
                for c in batch:
                    a = c.get("author") or {}
                    cid = to_int_relaxed(c.get("id"))
                    pid = to_int_relaxed(c.get("parent") or c.get("parent_comment") or c.get("in_reply_to"))
                    root = compute_root_id(int(cid) if cid is not None else None, int(pid) if pid is not None else None, parent_map) if cid is not None else None

                    parent_text = None
                    if pid is not None:
                        p = by_id.get(int(pid))
                        if p:
                            parent_text = clean_text(p.get("text") or "")

                    ctext = clean_text(c.get("text") or "")
                    if not ctext:
                        continue

                    resp = score_comment(
                        judge_model=judge_model,
                        parser_model=parser_model,
                        qtitle=qtitle,
                        comment_text=ctext,
                        parent_text=parent_text,
                    )

                    row = {
                        "ai_score": int(resp["score"]),
                        "rationale": resp.get("rationale", ""),
                        "flags_json": json.dumps(resp.get("flags") or {}, ensure_ascii=False),
                        "evidence_urls": ";".join(resp.get("evidence_urls") or []),
                        "ts": now_ts(),
                        "post_id": post_id,
                        "question_title": qtitle,
                        "comment_id": cid,
                        "parent_comment_id": pid,
                        "root_comment_id": root,
                        "author_id": to_int_relaxed(a.get("id")),
                        "author_username": clean_text(str(a.get("username") or a.get("name") or "")),
                        "vote_score": to_int_relaxed(c.get("vote_score")),
                        "comment_text": ctext,
                    }

                    w.writerow(row)
                    f_scored.flush()
                    written += 1

                    # "Print" CSV line-by-line (tail)
                    log_lines.append(row_to_csv_line(row, CSV_COLUMNS_SCORED))
                    tail_lines = log_lines[-60:]
                    log_ph.code(header + "\n" + "\n".join(tail_lines))

                    # Live dataframe preview (tail)
                    preview_df = pd.DataFrame([dict(zip(CSV_COLUMNS_SCORED, next(csv.reader([ln])))) for ln in tail_lines])
                    preview_ph.dataframe(preview_df, use_container_width=True, height=360)

            prog.progress(pi / max(1, total_posts))

    if f_dump is not None:
        f_dump.close()

    if out_dump_csv:
        status.success(f"Done. Dumped comments: {dumped} | Scored rows: {written}")
    else:
        status.success(f"Done. Scored rows: {written}")

# =========================
# UI
# =========================

def reset_run() -> None:
    _score_cache.clear()
    for k in ("OUT_SCORED_CSV", "OUT_DUMP_CSV"):
        st.session_state.pop(k, None)
    st.rerun()

st.set_page_config(page_title="Metaculus Comment Scorer", page_icon="🔍", layout="wide")
st.title("Metaculus Comment Scorer")

with st.sidebar:
    st.subheader("Keys (session overrides)")
    or_key = st.text_input("OpenRouter API key", type="password")
    if or_key.strip():
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = or_key.strip()

    m_key = st.text_input("Metaculus token (ONLY for IDs/Tournament)", type="password")
    if m_key.strip():
        st.session_state["METACULUS_TOKEN_OVERRIDE"] = m_key.strip()

    st.divider()
    st.subheader("Models (type OpenRouter IDs)")
    judge_model = st.text_input("Judge model ID", placeholder="e.g. openai/gpt-4.1")
    parser_model = st.text_input("Parser model ID", placeholder="e.g. openai/gpt-4o-mini")

    st.divider()
    if st.button("Reset", use_container_width=True):
        reset_run()

c1, c2 = st.columns([2, 1])
with c1:
    mode = st.radio("Mode", ["CSV (comments)", "IDs", "Tournament"], horizontal=True)
with c2:
    comments_limit = st.number_input("Max comments per post (IDs/Tournament)", min_value=10, max_value=1000, value=500, step=10)

run = st.button("Run scoring", type="primary")

# -------- Inputs per mode --------
csv_df: Optional[pd.DataFrame] = None
csv_uploaded = None

# CSV column selections (comments)
csv_col_post_id = None
csv_col_author_id = None
csv_col_author_username = None
csv_col_parent_id = None
csv_col_root_id = None
csv_col_comment_id = None
csv_col_question_title = None
csv_col_vote_score = None
csv_col_comment_text = None
csv_start_row = 1
csv_end_row = 1

ids_raw = ""
project_id = 0
n_from_tournament = 10
dump_in_metaculus_modes = True

if mode == "CSV (comments)":
    st.markdown("#### CSV input (comments)")
    st.caption("In CSV mode, the app DOES NOT call Metaculus and does NOT require METACULUS_TOKEN.")
    csv_uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if csv_uploaded is not None:
        try:
            csv_df = read_uploaded_csv(csv_uploaded)
            total_rows = len(csv_df)
            st.markdown("#### CSV preview (first 50 rows)")
            st.dataframe(csv_df.head(50), use_container_width=True, height=260)

            st.divider()
            st.subheader("Row range (resume)")
            csv_start_row = st.number_input("Start row (1-based, inclusive)", min_value=1, max_value=total_rows, value=1, step=1)
            default_end = min(int(csv_start_row) + 999, total_rows)
            csv_end_row = st.number_input("End row (1-based, inclusive)", min_value=int(csv_start_row), max_value=total_rows, value=int(default_end), step=1)
            st.caption(f"Using rows {int(csv_start_row)}..{int(csv_end_row)} inclusive ({int(csv_end_row)-int(csv_start_row)+1} rows).")

            df_slice_preview = csv_df.iloc[int(csv_start_row)-1:int(csv_end_row)]
            with st.expander("Selected slice preview (first 50 rows of slice)", expanded=False):
                st.dataframe(df_slice_preview.head(50), use_container_width=True, height=260)

            st.divider()
            st.subheader("Column mapping (select from CSV)")

            cols = list(csv_df.columns)
            none_opt = "(none)"
            opts = [none_opt] + cols

            # Required
            default_post = detect_first_col(csv_df, ["post_id", "post", "market_id"])
            default_text = detect_first_col(csv_df, ["comment_text", "text", "comment"])
            if default_post is None:
                default_post = cols[0]
            if default_text is None:
                default_text = cols[0] if len(cols) == 1 else cols[1]

            csv_col_post_id = st.selectbox("post_id (REQUIRED)", options=cols, index=cols.index(default_post))
            csv_col_comment_text = st.selectbox("comment_text (REQUIRED)", options=cols, index=cols.index(default_text))

            # Requested selectable fields
            default_author_id = detect_first_col(csv_df, ["author_id", "poster_id", "user_id"])
            default_author_username = detect_first_col(csv_df, ["author_username", "poster_username", "username", "user"])
            default_parent = detect_first_col(csv_df, ["parent_comment_id", "parent_id", "parent"])
            default_root = detect_first_col(csv_df, ["root_comment_id", "root_id", "root"])
            default_cid = detect_first_col(csv_df, ["comment_id", "id"])
            default_qtitle = detect_first_col(csv_df, ["question_title", "title"])
            default_vote = detect_first_col(csv_df, ["vote_score", "score", "votes"])

            csv_col_author_id = st.selectbox("author_id", options=opts, index=opts.index(default_author_id) if default_author_id in opts else 0)
            csv_col_author_username = st.selectbox("author_username", options=opts, index=opts.index(default_author_username) if default_author_username in opts else 0)
            csv_col_parent_id = st.selectbox("parent_comment_id", options=opts, index=opts.index(default_parent) if default_parent in opts else 0)
            csv_col_root_id = st.selectbox("root_comment_id", options=opts, index=opts.index(default_root) if default_root in opts else 0)

            # Strongly recommended (for parent text lookup + root compute)
            csv_col_comment_id = st.selectbox("comment_id (recommended)", options=opts, index=opts.index(default_cid) if default_cid in opts else 0)

            # Optional
            csv_col_question_title = st.selectbox("question_title (optional)", options=opts, index=opts.index(default_qtitle) if default_qtitle in opts else 0)
            csv_col_vote_score = st.selectbox("vote_score (optional)", options=opts, index=opts.index(default_vote) if default_vote in opts else 0)

            if csv_col_comment_id == none_opt or csv_col_parent_id == none_opt:
                st.info("Tip: select comment_id + parent_comment_id to enable parent_text context and best-effort root_id computation from the slice.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

elif mode == "IDs":
    st.markdown("#### IDs input")
    ids_raw = st.text_area("Question/Post IDs (comma/space separated)", placeholder="Example: 12345 67890")
    dump_in_metaculus_modes = st.checkbox("Also dump raw comments CSV (for later CSV mode)", value=True)

elif mode == "Tournament":
    st.markdown("#### Tournament input")
    a, b = st.columns(2)
    with a:
        project_id = st.number_input("Tournament ID", min_value=1, value=32821, step=1)
    with b:
        n_from_tournament = st.number_input("How many questions to score", min_value=1, value=10, step=1)
    dump_in_metaculus_modes = st.checkbox("Also dump raw comments CSV (for later CSV mode)", value=True)

# =========================
# Run
# =========================

if run:
    try:
        if not get_openrouter_key():
            st.error("Missing OPENROUTER_API_KEY.")
            st.stop()
        if not judge_model.strip():
            st.error("Judge model ID is empty.")
            st.stop()
        if not parser_model.strip():
            st.error("Parser model ID is empty.")
            st.stop()

        # Live UI placeholders
        st.markdown("#### Live output (CSV printed line-by-line)")
        log_ph = st.empty()
        st.markdown("#### Live table (tail)")
        preview_ph = st.empty()
        status = st.empty()
        prog = st.progress(0.0)

        out_scored = new_output_csv_path(prefix="comment_scores_", suffix=".csv")
        init_csv(out_scored, CSV_COLUMNS_SCORED)
        st.session_state["OUT_SCORED_CSV"] = out_scored

        if mode == "CSV (comments)":
            if csv_df is None:
                st.error("Upload a CSV first.")
                st.stop()
            if not csv_col_post_id or not csv_col_comment_text:
                st.error("Select required columns: post_id and comment_text.")
                st.stop()

            srow = max(1, int(csv_start_row))
            erow = max(srow, int(csv_end_row))
            df_use = csv_df.iloc[srow - 1 : erow].copy()

            # normalize optional selections
            none_opt = "(none)"
            pipeline_csv_comments(
                df_use=df_use,
                col_post_id=csv_col_post_id,
                col_comment_text=csv_col_comment_text,
                judge_model=judge_model.strip(),
                parser_model=parser_model.strip(),
                out_scored_csv=out_scored,
                log_ph=log_ph,
                preview_ph=preview_ph,
                status=status,
                prog=prog,
                col_question_title=None if (csv_col_question_title in (None, none_opt)) else csv_col_question_title,
                col_comment_id=None if (csv_col_comment_id in (None, none_opt)) else csv_col_comment_id,
                col_parent_id=None if (csv_col_parent_id in (None, none_opt)) else csv_col_parent_id,
                col_root_id=None if (csv_col_root_id in (None, none_opt)) else csv_col_root_id,
                col_author_id=None if (csv_col_author_id in (None, none_opt)) else csv_col_author_id,
                col_author_username=None if (csv_col_author_username in (None, none_opt)) else csv_col_author_username,
                col_vote_score=None if (csv_col_vote_score in (None, none_opt)) else csv_col_vote_score,
            )

        else:
            # Metaculus modes require token
            if not get_metaculus_token():
                st.error("Missing METACULUS_TOKEN (required for IDs/Tournament modes).")
                st.stop()

            out_dump = None
            if dump_in_metaculus_modes:
                out_dump = new_output_csv_path(prefix="comments_dump_", suffix=".csv")
                init_csv(out_dump, CSV_COLUMNS_DUMP)
                st.session_state["OUT_DUMP_CSV"] = out_dump
            else:
                st.session_state.pop("OUT_DUMP_CSV", None)

            subjects: List[Dict[str, Any]] = []
            if mode == "IDs":
                ids: List[int] = []
                for t in ids_raw.replace(",", " ").split():
                    v = to_int_relaxed(t.strip())
                    if v is not None:
                        ids.append(int(v))
                if not ids:
                    st.error("No valid IDs provided.")
                    st.stop()
                for x in ids:
                    subjects.append(fetch_question_info(int(x)))
                # uniq by post_id
                seen, uniq = set(), []
                for s in subjects:
                    pid = int(s["post_id"])
                    if pid not in seen:
                        seen.add(pid)
                        uniq.append(s)
                subjects = uniq

            else:  # Tournament
                subjects = get_tournament_subjects(int(project_id))[: int(n_from_tournament)]
                if not subjects:
                    st.error("No tournament subjects found.")
                    st.stop()

            with st.expander("Subjects", expanded=False):
                st.dataframe(pd.DataFrame(subjects), use_container_width=True, height=260)

            pipeline_metaculus_posts(
                subjects=subjects,
                judge_model=judge_model.strip(),
                parser_model=parser_model.strip(),
                out_scored_csv=out_scored,
                out_dump_csv=out_dump,
                comments_limit=int(comments_limit),
                log_ph=log_ph,
                preview_ph=preview_ph,
                status=status,
                prog=prog,
            )

    except Exception as e:
        st.error(f"Run failed (CSV preserved if created): {e}")

# =========================
# Downloads (partial OK)
# =========================

out_scored = st.session_state.get("OUT_SCORED_CSV")
out_dump = st.session_state.get("OUT_DUMP_CSV")

def read_bytes(p: str) -> bytes:
    with open(p, "rb") as f:
        return f.read()

if (out_scored and os.path.exists(out_scored)) or (out_dump and os.path.exists(out_dump)):
    st.markdown("### Downloads")

if out_dump and os.path.exists(out_dump):
    raw_dump = read_bytes(out_dump)
    st.download_button(
        "Download COMMENTS DUMP CSV (raw)",
        data=raw_dump,
        file_name="comments_dump_raw.csv",
        mime="text/csv",
    )

if out_scored and os.path.exists(out_scored):
    raw_scored = read_bytes(out_scored)
    st.download_button(
        "Download SCORED CSV (raw bytes, partial ok)",
        data=raw_scored,
        file_name="comment_scores_raw.csv",
        mime="text/csv",
    )
    fixed_scored = repair_csv_bytes_by_line(raw_scored)
    st.download_button(
        "Download SCORED CSV (repaired best-effort)",
        data=fixed_scored,
        file_name="comment_scores_repaired.csv",
        mime="text/csv",
    )

st.caption("CSV mode: no Metaculus connection. IDs/Tournament: requires METACULUS_TOKEN.")

