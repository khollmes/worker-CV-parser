import os
import json
import re
import time
from typing import Any, Dict, List, Tuple, Optional

import torch
import runpod
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Model config
# -------------------------
REPO_ID = "Alibaba-EI/SmartResume"
SUBFOLDER = "Qwen3-0.6B"

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/workspace/models/smartresume")
MODEL_PATH = f"{MODEL_CACHE_DIR}/{SUBFOLDER}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Per-task defaults (stability-first)
MAX_NEW_TOKENS_BASIC = int(os.getenv("MAX_NEW_TOKENS_BASIC", "220"))
MAX_NEW_TOKENS_EDU = int(os.getenv("MAX_NEW_TOKENS_EDU", "420"))
MAX_NEW_TOKENS_WORK = int(os.getenv("MAX_NEW_TOKENS_WORK", "650"))

REPAIR_ATTEMPTS_DEFAULT = int(os.getenv("REPAIR_ATTEMPTS", "2"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.0"))
REPETITION_PENALTY_DEFAULT = float(os.getenv("REPETITION_PENALTY", "1.01"))

# -------------------------
# JSON helpers (strict + defensive)
# -------------------------
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def _strip_code_fences(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s.strip()

def _basic_json_cleanup(s: str) -> str:
    s = _strip_code_fences(s)
    s = s.strip()
    # common “smart quotes”
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    # remove trailing commas
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s.strip()

def _slice_first_object(s: str) -> Optional[str]:
    """
    Returns the substring from first '{' to last '}' (inclusive), if both exist.
    Helps when model prints accidental prefix/suffix.
    """
    if not s:
        return None
    i = s.find("{")
    j = s.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return None
    return s[i:j+1]

def _try_parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Strict JSON parse with extra guardrails:
    - If output is a quoted JSON string, decode once then parse again.
    - Prefer slicing the first full {...} block.
    - Finally, attempt best-effort extraction of largest {...} block.
    """
    cleaned = _basic_json_cleanup(text)

    # 1) direct parse
    obj = _try_parse_json(cleaned)
    if isinstance(obj, dict):
        return obj

    # 2) handle "JSON as a quoted string" (your failure mode)
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        decoded = _try_parse_json(cleaned)
        if isinstance(decoded, str):
            decoded2 = _basic_json_cleanup(decoded)
            obj2 = _try_parse_json(decoded2)
            if isinstance(obj2, dict):
                return obj2

    # 3) slice from first { to last }
    sliced = _slice_first_object(cleaned)
    if sliced:
        obj3 = _try_parse_json(_basic_json_cleanup(sliced))
        if isinstance(obj3, dict):
            return obj3

    # 4) fallback: pick largest {...} block
    blocks = re.findall(r"\{.*\}", cleaned, flags=re.DOTALL)
    blocks.sort(key=len, reverse=True)
    for b in blocks:
        obj4 = _try_parse_json(_basic_json_cleanup(b))
        if isinstance(obj4, dict):
            return obj4

    raise ValueError("Invalid JSON output")

# -------------------------
# Schema merge
# -------------------------
def empty_final_schema() -> Dict[str, Any]:
    return {
        "basicInfo": {
            "name": "",
            "personalEmail": "",
            "phoneNumber": "",
            "age": "",
            "born": "",
            "gender": "",
            "desiredLocation": [],
            "jobIntention": "",
            "currentLocation": "",
            "placeOfOrigin": ""
        },
        "education": [],
        "workExperience": []
    }

# -------------------------
# Input shaping (3 parts -> globally indexed)
# -------------------------
def _normalize_lines(text: str) -> List[str]:
    lines: List[str] = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            ln = re.sub(r"\s+", " ", ln)
            lines.append(ln)
    return lines

def build_indexed_text_from_parts(part_basic: str, part_edu: str, part_work: str) -> str:
    lines: List[str] = []
    lines += ["=== BASIC PART ==="]
    lines += _normalize_lines(part_basic)
    lines += ["=== EDUCATION PART ==="]
    lines += _normalize_lines(part_edu)
    lines += ["=== WORK PART ==="]
    lines += _normalize_lines(part_work)

    indexed = []
    for i, ln in enumerate(lines):
        indexed.append(f"[{i}]: {ln}")
    return "\n".join(indexed)

# -------------------------
# Prompts (3 prompts + micro-optimizations)
# -------------------------
SYSTEM_STRICT_JSON = """You are a strict JSON generator for resume extraction.

HARD RULES (must follow):
- Output ONLY a single RFC8259-valid JSON object.
- The FIRST character of your entire output must be { and the LAST character must be }.
- Do NOT wrap the JSON in quotes. Do NOT output an escaped JSON string.
- No markdown, no code fences, no comments, no explanations, no extra tokens.
- Use double quotes for all keys and string values.
- No trailing commas anywhere.
- Do not invent facts: use only evidence from the provided indexed text.
- If a field is missing: use "" for strings, [] for arrays.
- Never add new keys. Keys must match the requested schema exactly.

Before you output, silently validate:
1) braces/brackets balance, 2) valid quotes/escapes, 3) top-level is an object.
If you are unsure, output the schema with empty values (still valid JSON).
"""

MICRO_OPT_LINES = """Do NOT output backslashes unless escaping a quote inside a string value.
Do NOT output any character before the first {.
"""

def _apply_template(tok, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system + "\n" + user + "\n"

def prompt_basic(indexed_text: str) -> str:
    user = f"""Extract ONLY basic information from the INDEXED TEXT.

Output EXACTLY this JSON structure (no extra keys):
{{"basicInfo":{{"name":"","personalEmail":"","phoneNumber":"","age":"","born":"","gender":"","desiredLocation":[],"jobIntention":"","currentLocation":"","placeOfOrigin":""}}}}

Rules:
- Use only evidence from text. No guessing.
- gender only "Male"/"Female" if explicitly stated else "".
- desiredLocation must be [] if not explicitly stated.
- Keep phone/email exactly as written.

{MICRO_OPT_LINES}
INDEXED TEXT:
{indexed_text}
"""
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, user)

def prompt_education(indexed_text: str) -> str:
    user = f"""Extract ONLY education experiences from the INDEXED TEXT.

Output EXACTLY this JSON structure:
{{"education":[]}}

Each education item must have EXACT keys:
{{"school":"","major":"","degree":"","startDate":"","endDate":"","location":""}}

Rules:
- Dates: "YYYY-MM" if month exists else "YYYY". endDate may be "present".
- If none found, return {{"education":[]}}.

{MICRO_OPT_LINES}
INDEXED TEXT:
{indexed_text}
"""
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, user)

def prompt_work(indexed_text: str) -> str:
    user = f"""Extract ONLY work experiences from the INDEXED TEXT.

Output EXACTLY this JSON structure:
{{"workExperience":[]}}

Each workExperience item must have EXACT keys:
{{"company":"","position":"","startDate":"","endDate":"","location":"","descriptionSpan":[],"descriptionShort":""}}

Rules:
- Dates: "YYYY-MM" if month exists else "YYYY". endDate may be "present".
- descriptionSpan: [] OR [startLine,endLine] (inclusive) pointing to the lines containing responsibilities/achievements.
- descriptionShort: <= 240 chars, verbatim snippet from those lines. Do NOT paraphrase.
- If none found, return {{"workExperience":[]}}.

{MICRO_OPT_LINES}
INDEXED TEXT:
{indexed_text}
"""
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, user)

def repair_prompt(task_name: str, broken_output: str, expected_top_keys: List[str]) -> str:
    # Repair prompt also uses strict JSON rules + micro-opts.
    # expected_top_keys is a small guard so the model doesn't return the wrong object.
    expected = ", ".join([f"\"{k}\"" for k in expected_top_keys])
    user = f"""Fix the following so it becomes valid JSON (RFC8259).

Constraints:
- Output ONLY a single JSON object.
- First char must be {{ and last char must be }}.
- Do NOT output an escaped JSON string.
- Use double quotes for keys/strings.
- No trailing commas.
- The top-level object MUST contain only these key(s): {expected}.
- Do NOT add any other keys.

{MICRO_OPT_LINES}
TASK: {task_name}
BROKEN OUTPUT:
{broken_output}
"""
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, user)

# -------------------------
# Model init + generation
# -------------------------
tokenizer = None
model = None

def download_model_if_needed() -> None:
    if os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        return
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, local_dir=MODEL_CACHE_DIR, local_dir_use_symlinks=False)

def _get_max_ctx() -> int:
    cfg = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "max_seq_len", "seq_length", "model_max_length"):
        v = getattr(cfg, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    v = getattr(tokenizer, "model_max_length", None)
    if isinstance(v, int) and 0 < v < 10**9:
        return v
    return 8192

def init() -> None:
    global tokenizer, model
    download_model_if_needed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

def generate_text(prompt: str, max_new_tokens: int, temperature: float, repetition_penalty: float) -> str:
    max_ctx = _get_max_ctx()
    safety = 32

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx - safety)
    input_len = int(enc["input_ids"].shape[1])
    allowed_new = min(max_new_tokens, max(0, max_ctx - input_len - safety))
    if allowed_new <= 0:
        raise ValueError(f"prompt_too_long: max_ctx={max_ctx} input_tokens={input_len}")

    inputs = {k: v.to(model.device) for k, v in enc.items()}
    do_sample = temperature > 0.0

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=allowed_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return decoded.strip()

init()

# -------------------------
# Task runner with repair attempts
# -------------------------
def run_task(
    task_name: str,
    prompt_text: str,
    expected_keys: List[str],
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
    repair_attempts: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (parsed_dict_or_none, meta) where meta includes raw_preview on failure.
    """
    t0 = time.time()
    raw = generate_text(prompt_text, max_new_tokens=max_new_tokens, temperature=temperature, repetition_penalty=repetition_penalty)

    # quick hard gate (prevents “mid-array continuation” and quoted-json)
    cleaned = _basic_json_cleanup(raw)
    ok_shape = cleaned.startswith("{") and cleaned.endswith("}")
    parsed: Optional[Dict[str, Any]] = None

    def _keys_ok(d: Dict[str, Any]) -> bool:
        # must be EXACTLY expected top-level keys
        return isinstance(d, dict) and sorted(d.keys()) == sorted(expected_keys)

    # 1) parse attempt
    if ok_shape:
        try:
            d = parse_json_strict(cleaned)
            if _keys_ok(d):
                parsed = d
        except Exception:
            parsed = None

    # 2) repair loop
    attempts_used = 0
    last = raw
    if parsed is None:
        for i in range(1, repair_attempts + 1):
            attempts_used = i
            rep = repair_prompt(task_name, last, expected_keys)
            last = generate_text(rep, max_new_tokens=max(160, max_new_tokens // 2), temperature=0.0, repetition_penalty=repetition_penalty)
            last_clean = _basic_json_cleanup(last)
            if last_clean.startswith("{") and last_clean.endswith("}"):
                try:
                    d = parse_json_strict(last_clean)
                    if _keys_ok(d):
                        parsed = d
                        break
                except Exception:
                    continue

    meta = {
        "attempts": attempts_used,
        "latency_s": round(time.time() - t0, 3),
    }

    if parsed is None:
        meta["raw_preview"] = _basic_json_cleanup(last)[:2000]
        meta["error"] = "model_output_not_valid_json"

    return parsed, meta

# -------------------------
# Main handler (3-call decomposition, stability-first)
# -------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects ONE resume already split upstream into 3 textual parts.
    job["input"] shape:
      {
        "part_basic": "text...",
        "part_education": "text...",
        "part_work": "text...",
        "temperature": 0.0 optional (default: 0.0),
        "repetition_penalty": 1.01 optional (default: 1.01),
        "repair_attempts": 2 optional,
        "return_indexed_text": false optional,

        # optional overrides
        "max_new_tokens_basic": 220,
        "max_new_tokens_education": 420,
        "max_new_tokens_work": 650
      }

    Returns:
      {
        "data": {basicInfo, education, workExperience},
        "meta": {
          "basic": {...},
          "education": {...},
          "work": {...},
          "latency_s": ...
        },
        "indexed_text": ... (optional)
      }
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    part_basic = inp.get("part_basic", "")
    part_edu = inp.get("part_education", "")
    part_work = inp.get("part_work", "")

    if not isinstance(part_basic, str) or not isinstance(part_edu, str) or not isinstance(part_work, str):
        return {"error": "bad_request", "detail": "part_basic/part_education/part_work must be strings"}

    all_text = (part_basic + part_edu + part_work).strip()
    if len(all_text) < 20:
        return {"error": "bad_request", "detail": "parts are empty or too short"}

    temperature = float(inp.get("temperature", TEMPERATURE_DEFAULT))
    repetition_penalty = float(inp.get("repetition_penalty", REPETITION_PENALTY_DEFAULT))
    repair_attempts = int(inp.get("repair_attempts", REPAIR_ATTEMPTS_DEFAULT))
    return_indexed = bool(inp.get("return_indexed_text", False))

    max_basic = int(inp.get("max_new_tokens_basic", MAX_NEW_TOKENS_BASIC))
    max_edu = int(inp.get("max_new_tokens_education", MAX_NEW_TOKENS_EDU))
    max_work = int(inp.get("max_new_tokens_work", MAX_NEW_TOKENS_WORK))

    indexed_text = build_indexed_text_from_parts(part_basic, part_edu, part_work)

    # 3 prompts (sequential for GPU stability; easiest to debug)
    basic_out, basic_meta = run_task(
        task_name="basicInfo",
        prompt_text=prompt_basic(indexed_text),
        expected_keys=["basicInfo"],
        max_new_tokens=max_basic,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
    )

    edu_out, edu_meta = run_task(
        task_name="education",
        prompt_text=prompt_education(indexed_text),
        expected_keys=["education"],
        max_new_tokens=max_edu,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
    )

    work_out, work_meta = run_task(
        task_name="workExperience",
        prompt_text=prompt_work(indexed_text),
        expected_keys=["workExperience"],
        max_new_tokens=max_work,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
    )

    # If any task failed, return a clear failure envelope (with raw previews)
    if basic_out is None or edu_out is None or work_out is None:
        resp = {
            "error": "model_output_not_valid_json",
            "meta": {
                "basic": basic_meta,
                "education": edu_meta,
                "work": work_meta,
                "latency_s": round(time.time() - t0, 3),
            },
        }
        if return_indexed:
            resp["indexed_text"] = indexed_text
        return resp

    # Merge into final schema
    final = empty_final_schema()
    final["basicInfo"] = basic_out.get("basicInfo", final["basicInfo"])
    final["education"] = edu_out.get("education", [])
    final["workExperience"] = work_out.get("workExperience", [])

    resp_ok = {
        "data": final,
        "meta": {
            "basic": basic_meta,
            "education": edu_meta,
            "work": work_meta,
            "latency_s": round(time.time() - t0, 3),
        },
    }
    if return_indexed:
        resp_ok["indexed_text"] = indexed_text
    return resp_ok

runpod.serverless.start({"handler": handler})
