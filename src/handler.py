import os
import json
import re
import time
from typing import Any, Dict, Optional

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

# ---------- schema + prompt ----------
def default_schema() -> Dict[str, Any]:
    return {
        "basics": {"full_name": "", "email": "", "phone": "", "location": "", "links": []},
        "headline": "",
        "summary": "",
        "skills": [],
        "experience": [{"company": "", "title": "", "start_date": "", "end_date": "", "location": "", "highlights": []}],
        "education": [{"institution": "", "degree": "", "field": "", "start_date": "", "end_date": ""}],
        "certifications": [],
        "languages": [],
        "projects": [],
        "publications": [],
    }

def build_prompt(tok, cv_text: str, schema: Dict[str, Any]) -> str:
    system = (
        "You are an information extraction engine.\n"
        "Return ONLY valid JSON (RFC8259). No markdown. No commentary.\n"
        "Do not invent facts. If missing, use empty string or empty list.\n"
        "Dates must be YYYY-MM or YYYY-MM-DD when possible.\n"
    )
    user = (
        "Extract this CV into EXACTLY the following JSON shape.\n"
        "Output must be STRICT JSON only (double quotes, no trailing commas).\n\n"
        f"JSON SHAPE:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"CV TEXT:\n{cv_text}\n"
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

# ---------- model init (load once, not per request) ----------
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

# Initialize at import time (best practice for serverless LLM workers) :contentReference[oaicite:3]{index=3}
init()

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job["input"] expected shape:
      {
        "cv_text": "...",
        "schema": {...} (optional),
        "max_new_tokens": 900 (optional),
        "temperature": 0.0 (optional),
        "repair_attempts": 2 (optional)
      }
    """
    t0 = time.time()
    inp = job.get("input", {}) or {}

    cv_text = inp.get("cv_text")
    if not isinstance(cv_text, str) or len(cv_text.strip()) < 10:
        return {"error": "bad_request", "detail": "cv_text must be a non-empty string"}

    schema = inp.get("schema") if isinstance(inp.get("schema"), dict) else default_schema()
    max_new = int(inp.get("max_new_tokens") or MAX_NEW_TOKENS_DEFAULT)
    temperature = float(inp.get("temperature") or 0.0)
    repair_attempts = int(inp.get("repair_attempts") or REPAIR_ATTEMPTS_DEFAULT)

    prompt = build_prompt(tokenizer, cv_text, schema)

    try:
        raw = generate_text(prompt, max_new_tokens=max_new, temperature=temperature)
        try:
            data = parse_json_strict(raw)
            return {"data": data, "meta": {"attempts": 0, "latency_s": round(time.time() - t0, 3)}}
        except Exception:
            last = raw
            for i in range(1, repair_attempts + 1):
                last = generate_text(repair_prompt_for(last), max_new_tokens=max_new, temperature=0.0)
                try:
                    data = parse_json_strict(last)
                    return {"data": data, "meta": {"attempts": i, "latency_s": round(time.time() - t0, 3)}}
                except Exception:
                    continue

            return {
                "error": "model_output_not_valid_json",
                "meta": {"attempts": repair_attempts, "latency_s": round(time.time() - t0, 3)},
                "raw_preview": _basic_json_cleanup(last)[:2000],
            }

    except RuntimeError as e:
        # CUDA device-side assert -> worker likely needs restart
        msg = str(e)
        return {"error": "runtime_error", "detail": msg[:2000]}

    except Exception as e:
        return {"error": "exception", "detail": str(e)[:2000]}

runpod.serverless.start({"handler": handler})
