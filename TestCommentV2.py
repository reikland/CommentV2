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

RECENT_LIMIT      = 10
BATCH_SIZE        = 10
MAX_COMMENT_CHARS = 2200
MAX_PARENT_CHARS  = 1400
PARSER_RETRIES    = 3

CSV_COLUMNS = [
    "ai_score",
    "comment_text",
    "ts",
    "market_id",
    "question_title",
    "question_url",
    "comment_id",
    "parent_id",
    "poster_id",
    "poster_username",
    "vote_score",
    "rationale",
    "flags_json",
    "evidence_urls",
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
        "User-Agent": _ascii_safe("metaculus-comment-scorer/4.0"),
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

def build_judge_msgs(qtitle: str, qurl: str, ctext: str, ptext: Optional[str]) -> List[Dict[str, str]]:
    parent_block = f"PARENT_COMMENT_TEXT:\n{ptext}\n\n" if ptext else ""
    user = (
        "Rate this comment using the strict 4-line format.\n\n"
        f"QUESTION_TITLE: {qtitle}\nQUESTION_URL: {qurl}\n\n"
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

def score_comment(judge_model: str, parser_model: str, qtitle: str, qurl: str, comment_text: str, parent_text: Optional[str]) -> Dict[str, Any]:
    ct = _truncate(comment_text, MAX_COMMENT_CHARS, "\n\n[Comment truncated]")
    pt = _truncate(parent_text or "", MAX_PARENT_CHARS, "\n\n[Parent truncated]") if parent_text else ""

    key_raw = f"{judge_model}||{parser_model}||{qtitle}||{qurl}||{ct}||{pt}"
    key = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()
    if key in _score_cache:
        return _score_cache[key]

    try:
        raw = openrouter_chat(build_judge_msgs(qtitle, qurl, ct, pt or None), model=judge_model, max_tokens=200, title="Judge")
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
# Metaculus API
# =========================

HTTP = requests.Session()

def metaculus_headers() -> Dict[str, str]:
    tok = get_metaculus_token()
    if not tok:
        raise RuntimeError("Missing METACULUS_TOKEN.")
    return {"User-Agent": "metaculus-comment-scorer/4.0 (+python-requests)", "Authorization": f"Token {tok}"}

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
        url = q.get("page_url") or q.get("url") or f"{BASE}/questions/{qid}/"
        return {"question_id": qid, "post_id": post_id, "title": title, "url": url}
    except Exception:
        pid = int(qid_or_post)
        return {"question_id": pid, "post_id": pid, "title": f"Post {pid}", "url": f"{BASE}/posts/{pid}/"}

@st.cache_data(show_spinner=False, ttl=120)
def fetch_recent_open_questions(limit: int) -> List[Dict[str, Any]]:
    data = metaculus_get(f"{API2}/questions/", {"status": "open", "limit": max(limit, 20)})
    qs = data.get("results") or []

    def ts(q: Dict[str, Any]) -> str:
        return q.get("open_time") or q.get("created_at") or q.get("scheduled_close_time") or ""
    qs.sort(key=ts, reverse=True)

    out: List[Dict[str, Any]] = []
    for q in qs[:limit]:
        qid = q.get("id")
        if qid is None:
            continue
        post_id = q.get("post_id") or q.get("post")
        if post_id:
            out.append({
                "question_id": int(qid),
                "post_id": int(post_id),
                "title": q.get("title") or f"Question {qid}",
                "url": q.get("page_url") or q.get("url") or f"{BASE}/questions/{qid}/",
            })
        else:
            out.append(fetch_question_info(int(qid)))

    seen, uniq = set(), []
    for s in out:
        pid = int(s["post_id"])
        if pid not in seen:
            seen.add(pid)
            uniq.append(s)
    return uniq

# -------- fixed comment pagination (server cap-safe) --------
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
# -----------------------------------------------------------

def _decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", "replace")

# -------- robust project export with retries on 503 --------
def download_project_question_csv(project_id: int) -> bytes:
    tok = get_metaculus_token()
    if not tok:
        raise RuntimeError("Missing METACULUS_TOKEN.")

    url = f"{BASE}/api/projects/{int(project_id)}/download-data/"
    headers = {
        "User-Agent": "metaculus-project-dl/4.0 (+python-requests)",
        "Accept": "application/zip,application/octet-stream,*/*;q=0.8",
        "Authorization": f"Token {tok}",
    }
    params = {"include_comments": "false", "include_scores": "false"}

    last_exc: Optional[Exception] = None
    for attempt in range(6):
        try:
            r = HTTP.get(url, headers=headers, params=params, timeout=120)

            # Bulk export infra can be flaky
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
# -----------------------------------------------------------

def parse_question_data_csv(data: bytes) -> List[Dict[str, Any]]:
    text = _decode_bytes(data)
    reader = csv.DictReader(io.StringIO(text))

    def to_int(x: Any) -> Optional[int]:
        s = str(x).strip() if x is not None else ""
        if not s:
            return None
        m = re.search(r"\d+", s.replace(",", ""))
        return int(m.group(0)) if m else None

    rows: List[Dict[str, Any]] = []
    for row in reader:
        post_id = to_int(row.get("Post ID") or row.get("post_id"))
        qid = to_int(row.get("Question ID") or row.get("question_id") or row.get("id"))
        if post_id is None:
            continue
        title = (row.get("Question Title") or row.get("title") or "").strip() or f"Post {post_id}"
        url = (row.get("Question URL") or row.get("url") or "").strip()
        if not url and qid:
            url = f"{BASE}/questions/{qid}/"
        if not url:
            url = f"{BASE}/posts/{post_id}/"
        rows.append({"post_id": int(post_id), "question_id": int(qid) if qid else None, "title": title, "url": url})
    return rows

# -------- fallback: list project questions via API2 (no ZIP) --------
def _parse_questions_page(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    batch = data.get("results") or []
    out: List[Dict[str, Any]] = []
    for q in batch:
        qid = q.get("id")
        post_id = q.get("post_id") or q.get("post") or qid
        if post_id is None:
            continue
        title = q.get("title") or (f"Question {qid}" if qid is not None else f"Post {post_id}")
        url = q.get("page_url") or q.get("url") or (f"{BASE}/questions/{qid}/" if qid is not None else f"{BASE}/posts/{post_id}/")
        out.append(
            {
                "question_id": int(qid) if qid is not None else int(post_id),
                "post_id": int(post_id),
                "title": title,
                "url": url,
            }
        )
    return out

def get_tournament_subjects_via_api2(project_id: int) -> List[Dict[str, Any]]:
    """
    Fallback when /api/projects/{id}/download-data/ is unavailable (503) or restricted.
    Tries a few known patterns to list questions in a project/tournament.
    """
    out: List[Dict[str, Any]] = []

    # Strategy A: /api2/questions/ with filters (common)
    def try_questions_list(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            return metaculus_get(f"{API2}/questions/", params)
        except Exception:
            return None

    offset = 0
    page_size = 200
    while True:
        tried_any = False
        data = None

        # Several param variants; keep it defensive
        variants = [
            {"project": int(project_id), "limit": page_size, "offset": offset},
            {"project_id": int(project_id), "limit": page_size, "offset": offset},
            {"projects": int(project_id), "limit": page_size, "offset": offset},
        ]
        # Best-effort include status=all (some deployments accept it; harmless if ignored)
        expanded: List[Dict[str, Any]] = []
        for v in variants:
            expanded.append({**v, "status": "all"})
            expanded.append(v)

        for p in expanded:
            tried_any = True
            data = try_questions_list(p)
            if isinstance(data, dict) and "results" in data:
                break

        if not tried_any or data is None or not isinstance(data, dict) or "results" not in data:
            break

        batch = data.get("results") or []
        if not batch:
            break

        out.extend(_parse_questions_page(data))
        offset += len(batch)
        if len(batch) < page_size:
            break
        time.sleep(0.15)

    if out:
        seen, uniq = set(), []
        for s in out:
            pid = int(s["post_id"])
            if pid not in seen:
                seen.add(pid)
                uniq.append(s)
        return uniq

    # Strategy B: try a dedicated endpoint (if it exists) /api2/projects/{id}/questions/
    # (Not guaranteed; kept best-effort.)
    try:
        data = metaculus_get(f"{API2}/projects/{int(project_id)}/questions/", {"limit": 200, "offset": 0})
        out2 = _parse_questions_page(data)
        if out2:
            seen, uniq = set(), []
            for s in out2:
                pid = int(s["post_id"])
                if pid not in seen:
                    seen.add(pid)
                    uniq.append(s)
            return uniq
    except Exception:
        pass

    raise RuntimeError(
        "Could not list tournament/project questions via API2 fallback. "
        "The project export may be restricted or temporarily unavailable."
    )
# -------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=300)
def get_tournament_subjects(project_id: int) -> List[Dict[str, Any]]:
    # Prefer ZIP export when available; fall back to API2 listing when export 503/restricted.
    try:
        rows = parse_question_data_csv(download_project_question_csv(int(project_id)))
        return [
            {"question_id": (r.get("question_id") or r["post_id"]), "post_id": r["post_id"], "title": r["title"], "url": r["url"]}
            for r in rows
        ]
    except Exception:
        return get_tournament_subjects_via_api2(int(project_id))

# =========================
# CSV + aggregation
# =========================

def new_output_csv_path() -> str:
    fd, path = tempfile.mkstemp(prefix="metaculus_comment_scores_", suffix=".csv")
    os.close(fd)
    return path

def init_csv(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in CSV_COLUMNS if c in df.columns] + [c for c in df.columns if c not in CSV_COLUMNS]
    return df[cols]

def aggregate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_q = (
        df.groupby(["market_id", "question_title", "question_url"], dropna=False)
        .agg(
            n_comments=("ai_score", "size"),
            avg_score=("ai_score", "mean"),
            p_low=("ai_score", lambda x: (x <= 2).mean()),
            p_high=("ai_score", lambda x: (x >= 4).mean()),
        )
        .reset_index()
        .sort_values(["avg_score", "n_comments"], ascending=[False, False])
    )
    agg_author = (
        df.groupby(["poster_id", "poster_username"], dropna=False)
        .agg(
            n_comments=("ai_score", "size"),
            avg_score=("ai_score", "mean"),
            p_low=("ai_score", lambda x: (x <= 2).mean()),
            p_high=("ai_score", lambda x: (x >= 4).mean()),
        )
        .reset_index()
        .sort_values(["avg_score", "n_comments"], ascending=[False, False])
    )
    return agg_q, agg_author

# =========================
# UI
# =========================

def reset_run() -> None:
    _score_cache.clear()
    st.session_state.pop("OUT_CSV", None)
    st.rerun()

st.set_page_config(page_title="Metaculus Comment Scorer", page_icon="🔍", layout="wide")
st.title("Metaculus Comment Scorer")

with st.sidebar:
    st.subheader("Keys (session overrides)")
    or_key = st.text_input("OpenRouter API key", type="password")
    if or_key.strip():
        st.session_state["OPENROUTER_API_KEY_OVERRIDE"] = or_key.strip()

    m_key = st.text_input("Metaculus token", type="password")
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
    mode = st.radio("Mode", ["Recent", "IDs", "Tournament"], horizontal=True)
with c2:
    comments_limit = st.number_input(
        "Max comments per post",
        min_value=10,
        max_value=1000,
        value=1000,
        step=10,
    )

ids_raw = ""
project_id = 0
n_from_csv = 10

if mode == "IDs":
    ids_raw = st.text_area("IDs (comma/space separated)", placeholder="Example: 12345 67890")
elif mode == "Tournament":
    a, b = st.columns(2)
    with a:
        project_id = st.number_input("Tournament ID", min_value=1, value=32821, step=1)
    with b:
        n_from_csv = st.number_input("How many rows to score", min_value=1, value=10, step=1)

    if get_metaculus_token():
        try:
            subs_all = get_tournament_subjects(int(project_id))
            st.markdown("#### Tournament preview")
            st.dataframe(pd.DataFrame(subs_all[: int(n_from_csv)]), use_container_width=True, height=260)
        except Exception as e:
            st.info(f"Preview unavailable: {e}")
    else:
        st.info("Enter METACULUS_TOKEN to preview tournament questions.")

run = st.button("Run scoring", type="primary")

# =========================
# Pipeline
# =========================

def resolve_subjects() -> List[Dict[str, Any]]:
    if mode == "Recent":
        return fetch_recent_open_questions(RECENT_LIMIT)

    if mode == "IDs":
        ids: List[int] = []
        for t in ids_raw.replace(",", " ").split():
            try:
                ids.append(int(t.strip()))
            except Exception:
                continue
        subs = [fetch_question_info(i) for i in ids]
        seen, out = set(), []
        for s in subs:
            pid = int(s["post_id"])
            if pid not in seen:
                seen.add(pid)
                out.append(s)
        return out

    return get_tournament_subjects(int(project_id))[: int(n_from_csv)]

def pipeline(subjects: List[Dict[str, Any]], out_csv: str, preview_ph: st.delta_generator.DeltaGenerator) -> None:
    prog = st.progress(0.0)
    status = st.empty()
    total = len(subjects)

    preview_rows: List[Dict[str, Any]] = []
    written = 0

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        for i, s in enumerate(subjects, 1):
            post_id = int(s["post_id"])
            title = s.get("title", "")
            url = s.get("url", "")

            status.info(f"[{i}/{total}] Fetching comments — post_id={post_id}")
            comments = fetch_comments(post_id, limit=int(comments_limit))

            by_id: Dict[int, Dict[str, Any]] = {}
            usable: List[Dict[str, Any]] = []
            for c in comments:
                cid = c.get("id")
                if cid is not None:
                    by_id[int(cid)] = c
                if clean_text(c.get("text") or ""):
                    usable.append(c)

            status.info(f"[{i}/{total}] Scoring {len(usable)} comments — {title}")

            skip_post = False
            for batch in chunked(usable, BATCH_SIZE):
                try:
                    for c in batch:
                        a = c.get("author") or {}
                        cid = c.get("id")
                        parent_id = c.get("parent") or c.get("parent_comment") or c.get("in_reply_to")

                        parent_text = None
                        if parent_id:
                            p = by_id.get(int(parent_id))
                            if p:
                                parent_text = (p.get("text") or "").strip()

                        ctext = clean_text(c.get("text") or "")
                        resp = score_comment(
                            judge_model=judge_model.strip(),
                            parser_model=parser_model.strip(),
                            qtitle=title,
                            qurl=url,
                            comment_text=ctext,
                            parent_text=parent_text,
                        )

                        row = {
                            "ai_score": int(resp["score"]),
                            "comment_text": ctext,
                            "ts": now_ts(),
                            "market_id": post_id,
                            "question_title": title,
                            "question_url": url,
                            "comment_id": cid,
                            "parent_id": parent_id,
                            "poster_id": a.get("id"),
                            "poster_username": (a.get("username") or a.get("name") or ""),
                            "vote_score": c.get("vote_score"),
                            "rationale": resp.get("rationale", ""),
                            "flags_json": json.dumps(resp.get("flags") or {}, ensure_ascii=False),
                            "evidence_urls": ";".join(resp.get("evidence_urls") or []),
                        }

                        w.writerow(row)
                        f.flush()
                        written += 1

                        # Live preview (tail) — keep at 40 as requested
                        preview_rows.append(row)
                        tail = preview_rows[-40:]
                        pdf = pd.DataFrame(tail)
                        preview_ph.dataframe(pdf[CSV_COLUMNS], use_container_width=True, height=360)

                        status.info(f"[{i}/{total}] Row {written} written — score={row['ai_score']}")

                except ParserFormatError as e:
                    status.warning(f"[{i}/{total}] Parser formatting failed → skipping this post. {e}")
                    skip_post = True
                except Exception as e:
                    status.error(f"Unexpected error (CSV preserved): {e}")
                    return

                if skip_post:
                    break

            prog.progress(i / total)

    status.success(f"Done. Total rows written: {written}")

# =========================
# Run (CSV preserved even on errors)
# =========================

if run:
    out_csv: Optional[str] = None
    try:
        if not get_openrouter_key():
            st.error("Missing OPENROUTER_API_KEY.")
            st.stop()
        if not get_metaculus_token():
            st.error("Missing METACULUS_TOKEN.")
            st.stop()
        if not judge_model.strip():
            st.error("Judge model ID is empty.")
            st.stop()
        if not parser_model.strip():
            st.error("Parser model ID is empty.")
            st.stop()

        subjects = resolve_subjects()
        if not subjects:
            st.warning("No subjects resolved.")
            st.stop()

        out_csv = new_output_csv_path()
        init_csv(out_csv)
        st.session_state["OUT_CSV"] = out_csv

        with st.expander("Subjects (resolved)", expanded=False):
            st.dataframe(pd.DataFrame(subjects), use_container_width=True, height=260)

        st.markdown("#### Live preview (updates every scored comment)")
        preview_ph = st.empty()

        pipeline(subjects, out_csv, preview_ph)

    except Exception as e:
        st.error(f"Run failed (CSV preserved if created): {e}")

# =========================
# Downloads + results (always if CSV exists)
# =========================

out_csv = st.session_state.get("OUT_CSV")
if out_csv and os.path.exists(out_csv):
    st.markdown("### Downloads")

    def read_bytes(p: str) -> bytes:
        with open(p, "rb") as f:
            return f.read()

    st.download_button(
        "Download RAW CSV (partial ok)",
        data=read_bytes(out_csv),
        file_name="comment_scores_raw.csv",
        mime="text/csv",
    )

    try:
        df = load_df(out_csv)
        st.subheader("Results")
        st.caption(f"Rows: {len(df)} | Posts: {df['market_id'].nunique() if 'market_id' in df else 0}")
        st.dataframe(df, use_container_width=True, height=420)

        agg_q, agg_author = aggregate(df)
        left, right = st.columns(2)
        with left:
            st.markdown("#### By post")
            st.dataframe(agg_q, use_container_width=True, height=320)
        with right:
            st.markdown("#### By commenter")
            st.dataframe(agg_author, use_container_width=True, height=320)

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download by post CSV",
            data=agg_q.to_csv(index=False).encode("utf-8"),
            file_name="comment_scores_by_post.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download by commenter CSV",
            data=agg_author.to_csv(index=False).encode("utf-8"),
            file_name="comment_scores_by_commenter.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Could not load/aggregate CSV (download still works): {e}")
else:
    st.info("No run yet (or CSV not found).")

st.caption("Keys: set OPENROUTER_API_KEY + METACULUS_TOKEN via secrets/env, or use sidebar overrides.")
