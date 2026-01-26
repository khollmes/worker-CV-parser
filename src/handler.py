import os
import json
import re
import time
from typing import Any, Dict, Optional, Tuple, List

import torch
import runpod
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO_ID = "Alibaba-EI/SmartResume"
SUBFOLDER = "Qwen3-0.6B"

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/workspace/models/smartresume")
MODEL_PATH = f"{MODEL_CACHE_DIR}/{SUBFOLDER}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS", "900"))
REPAIR_ATTEMPTS_DEFAULT = int(os.getenv("REPAIR_ATTEMPTS", "2"))

# ---------- JSON helpers ----------
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = _CODE_FENCE_RE.sub("", s).strip()
    return s.strip()

def _basic_json_cleanup(s: str) -> str:
    s = _strip_code_fences(s)
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s.strip()

def _extract_candidate_json_blocks(text: str) -> list[str]:
    blocks = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    blocks.sort(key=len, reverse=True)
    return blocks

def parse_json_strict(text: str) -> Dict[str, Any]:
    cleaned = _basic_json_cleanup(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    for block in _extract_candidate_json_blocks(cleaned):
        try:
            return json.loads(_basic_json_cleanup(block))
        except Exception:
            continue
    raise ValueError("Invalid JSON output")

# ---------- schema (paper-style) ----------
def default_schema_paper() -> Dict[str, Any]:
    """
    Single unified output schema for ONE model call.
    - Keep descriptionSpan so your server can rehydrate exact text later.
    """
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
        "education": [
            {
                "school": "",
                "major": "",
                "degree": "",
                "startDate": "",
                "endDate": "",
                "location": ""
            }
        ],
        "workExperience": [
            {
                "company": "",
                "position": "",
                "startDate": "",
                "endDate": "",
                "location": "",
                "descriptionSpan": [],
                "descriptionShort": ""
            }
        ]
    }

# ---------- input shaping ----------
def _normalize_lines(text: str) -> List[str]:
    """
    Turns raw text into non-empty, reasonably trimmed lines.
    """
    lines = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln:
            # avoid huge whitespace runs
            ln = re.sub(r"\s+", " ", ln)
            lines.append(ln)
    return lines

def build_indexed_text_from_parts(
    part_basic: str,
    part_edu: str,
    part_work: str,
) -> str:
    """
    User says they will send 3 parts as text.
    We unify them into ONE globally indexed sequence, so spans are stable.
    """
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

# ---------- prompt ----------
def build_prompt_unified(tok, indexed_text: str, schema: Dict[str, Any]) -> str:
    system = (
        "You are a resume information extraction engine.\n"
        "Output MUST be a single valid JSON object (RFC8259) and nothing else.\n"
        "No markdown, no commentary.\n"
        "Do NOT invent facts. Use ONLY evidence from the provided text.\n"
        "If a field is missing, use empty string \"\" or empty list [] (as appropriate).\n"
    )

    user = (
        "Task: Extract resume information from the INDEXED TEXT into EXACTLY the JSON schema below.\n"
        "Rules:\n"
        "- Output EXACTLY the schema keys; do not add extra keys.\n"
        "- Keep phone/email exactly as written.\n"
        "- gender must be \"Male\" or \"Female\" only if explicitly stated; else \"\".\n"
        "- desiredLocation must be an array; if none, return [].\n"
        "- For workExperience.descriptionSpan: return [startLine, endLine] (inclusive) pointing to lines "
        "that contain the job description/responsibilities/achievements. If none, return [].\n"
        "- Do NOT paraphrase long descriptions; use descriptionSpan instead.\n"
        "- Dates: use YYYY-MM if month exists; else YYYY; endDate can be \"present\".\n\n"
        f"JSON SCHEMA (shape):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"INDEXED TEXT:\n{indexed_text}\n"
    )

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system + "\n" + user + "\nJSON:\n"

def repair_prompt_for(broken_output: str) -> str:
    return (
        "Fix the following so it becomes valid JSON (RFC8259).\n"
        "Rules:\n"
        "- Output ONLY JSON. No markdown. No extra text.\n"
        "- Use double quotes for keys/strings.\n"
        "- Remove trailing commas.\n\n"
        f"BROKEN OUTPUT:\n{broken_output}\n"
    )

# ---------- model init ----------
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

def generate_text(prompt: str, max_new_tokens: int, temperature: float) -> str:
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
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    return decoded.strip()

init()

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Expects ONE PDF already split upstream into 3 textual parts.
    job["input"] shape:
      {
        "part_basic": "text...",
        "part_education": "text...",
        "part_work": "text...",
        "schema": {...} optional (defaults to paper-style),
        "max_new_tokens": 900 optional,
        "temperature": 0.0 optional,
        "repair_attempts": 2 optional,
        "return_indexed_text": false optional
      }

    Note: This handler does ONE model call (single prompt) to extract all 3 sections.
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    part_basic = inp.get("part_basic", "")
    part_edu = inp.get("part_education", "")
    part_work = inp.get("part_work", "")

    # minimal validation
    if not isinstance(part_basic, str) or not isinstance(part_edu, str) or not isinstance(part_work, str):
        return {"error": "bad_request", "detail": "part_basic/part_education/part_work must be strings"}
    if len((part_basic + part_edu + part_work).strip()) < 20:
        return {"error": "bad_request", "detail": "parts are empty or too short"}

    schema = inp.get("schema") if isinstance(inp.get("schema"), dict) else default_schema_paper()
    max_new = int(inp.get("max_new_tokens") or MAX_NEW_TOKENS_DEFAULT)
    temperature = float(inp.get("temperature") or 0.0)
    repair_attempts = int(inp.get("repair_attempts") or REPAIR_ATTEMPTS_DEFAULT)
    return_indexed = bool(inp.get("return_indexed_text") or False)

    indexed_text = build_indexed_text_from_parts(part_basic, part_edu, part_work)
    prompt = build_prompt_unified(tokenizer, indexed_text, schema)

    try:
        raw = generate_text(prompt, max_new_tokens=max_new, temperature=temperature)

        try:
            data = parse_json_strict(raw)
            resp = {"data": data, "meta": {"attempts": 0, "latency_s": round(time.time() - t0, 3)}}
            if return_indexed:
                resp["indexed_text"] = indexed_text
            return resp

        except Exception:
            last = raw
            for i in range(1, repair_attempts + 1):
                last = generate_text(repair_prompt_for(last), max_new_tokens=max_new, temperature=0.0)
                try:
                    data = parse_json_strict(last)
                    resp = {"data": data, "meta": {"attempts": i, "latency_s": round(time.time() - t0, 3)}}
                    if return_indexed:
                        resp["indexed_text"] = indexed_text
                    return resp
                except Exception:
                    continue

            resp = {
                "error": "model_output_not_valid_json",
                "meta": {"attempts": repair_attempts, "latency_s": round(time.time() - t0, 3)},
                "raw_preview": _basic_json_cleanup(last)[:2000],
            }
            if return_indexed:
                resp["indexed_text"] = indexed_text
            return resp

    except RuntimeError as e:
        return {"error": "runtime_error", "detail": str(e)[:2000]}
    except Exception as e:
        return {"error": "exception", "detail": str(e)[:2000]}

runpod.serverless.start({"handler": handler})
