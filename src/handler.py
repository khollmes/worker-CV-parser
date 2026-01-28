import logging
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
# Logging setup
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_FULL_PAYLOADS = os.getenv("LOG_FULL_PAYLOADS", "true").lower() in ("1", "true", "yes", "on")
LOG_PREVIEW_CHARS = int(os.getenv("LOG_PREVIEW_CHARS", "800"))
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
logging.captureWarnings(True)
logger = logging.getLogger("smartresume")


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=True, default=str)
    except Exception as exc:
        return f"<unserializable:{type(obj).__name__} err={exc}>"


def _preview_text(text: Optional[str]) -> str:
    if text is None:
        return "<none>"
    s = str(text)
    if LOG_FULL_PAYLOADS:
        return s
    if len(s) <= LOG_PREVIEW_CHARS:
        return s
    return f"{s[:LOG_PREVIEW_CHARS]}...<truncated {len(s) - LOG_PREVIEW_CHARS} chars>"


def _log_text(label: str, text: Optional[str], level: int = logging.DEBUG) -> None:
    if not logger.isEnabledFor(level):
        return
    raw = "" if text is None else str(text)
    preview = _preview_text(raw)
    logger.log(level, "%s (%d chars): %s", label, len(raw), preview)


logger.info(
    "Logging configured | level=%s full_payloads=%s preview_chars=%d",
    LOG_LEVEL, LOG_FULL_PAYLOADS, LOG_PREVIEW_CHARS
)

# -------------------------
# Model config
# -------------------------
REPO_ID = "Alibaba-EI/SmartResume"
SUBFOLDER = "Qwen3-0.6B"

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/workspace/models/smartresume")
MODEL_PATH = f"{MODEL_CACHE_DIR}/{SUBFOLDER}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MAX_NEW_TOKENS_BASIC = int(os.getenv("MAX_NEW_TOKENS_BASIC", "220"))
MAX_NEW_TOKENS_EDU = int(os.getenv("MAX_NEW_TOKENS_EDU", "420"))
MAX_NEW_TOKENS_WORK = int(os.getenv("MAX_NEW_TOKENS_WORK", "650"))

REPAIR_ATTEMPTS_DEFAULT = int(os.getenv("REPAIR_ATTEMPTS", "2"))
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE", "0.0"))
REPETITION_PENALTY_DEFAULT = float(os.getenv("REPETITION_PENALTY", "1.01"))

logger.info(
    "Model config | repo=%s subfolder=%s cache_dir=%s model_path=%s device=%s dtype=%s",
    REPO_ID, SUBFOLDER, MODEL_CACHE_DIR, MODEL_PATH, DEVICE, DTYPE
)
logger.info(
    "Defaults | max_basic=%d max_edu=%d max_work=%d repair=%d temp=%.3f rep=%.3f",
    MAX_NEW_TOKENS_BASIC, MAX_NEW_TOKENS_EDU, MAX_NEW_TOKENS_WORK,
    REPAIR_ATTEMPTS_DEFAULT, TEMPERATURE_DEFAULT, REPETITION_PENALTY_DEFAULT
)

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
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s.strip()


def _slice_first_object(s: str) -> Optional[str]:
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
    cleaned = _basic_json_cleanup(text)

    # 1) direct parse
    obj = _try_parse_json(cleaned)
    if isinstance(obj, dict):
        return obj

    # 2) quoted JSON
    if cleaned.startswith("\"") and cleaned.endswith("\""):
        decoded = _try_parse_json(cleaned)
        if isinstance(decoded, str):
            decoded2 = _basic_json_cleanup(decoded)
            obj2 = _try_parse_json(decoded2)
            if isinstance(obj2, dict):
                return obj2

    # 3) slice first { ... }
    sliced = _slice_first_object(cleaned)
    if sliced:
        obj3 = _try_parse_json(_basic_json_cleanup(sliced))
        if isinstance(obj3, dict):
            return obj3

    # 4) largest block
    blocks = re.findall(r"\{.*\}", cleaned, flags=re.DOTALL)
    blocks.sort(key=len, reverse=True)
    for b in blocks:
        obj4 = _try_parse_json(_basic_json_cleanup(b))
        if isinstance(obj4, dict):
            return obj4

    logger.error("parse_json_strict failed; cleaned length=%d", len(cleaned))
    _log_text("parse_json_strict cleaned", cleaned, level=logging.ERROR)
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
# Input shaping
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
# Prompts
# -------------------------
SYSTEM_STRICT_JSON = """You are a strict JSON generator for resume extraction.
... (unchanged)
"""

MICRO_OPT_LINES = """Do NOT output backslashes unless escaping a quote inside a string value.
Do NOT output any character before the first {. """

def _apply_template(tok, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system + "\n" + user + "\n"

# -------------------------
# Prompt builders
# -------------------------
def _build_prompt(task_description: str, indexed_text: str) -> str:
    return "\n".join(
        [
            MICRO_OPT_LINES,
            task_description.strip(),
            "",
            "Indexed text (already normalized, one line per entry):",
            indexed_text or "",
        ]
    )


def prompt_basic(indexed_text: str) -> str:
    task = """
Return a strict JSON object with exactly one key, "basicInfo".
Inside "basicInfo" provide the following properties: name, personalEmail,
phoneNumber, age, born, gender, desiredLocation, jobIntention,
currentLocation, placeOfOrigin. Use strings for scalar values and arrays
for the multi-valued fields (e.g., desiredLocation). If information is
missing, return an empty string or empty array as appropriate. Do not
wrap responses in markdown code fences or add any extra commentary.
"""
    prompt = _build_prompt(task, indexed_text)
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, prompt)


def prompt_education(indexed_text: str) -> str:
    task = """
Return a strict JSON object with exactly one key, "education".
The value must be an array of education records ordered newest first.
Each record should contain: institution, degree, major, startDate,
endDate, and summary. Use ISO-style dates when available, empty strings
when unknown, and a list of short summary sentences if you can infer any
coursework or honors. Only output the JSON object (no prose or markup).
"""
    prompt = _build_prompt(task, indexed_text)
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, prompt)


def prompt_work(indexed_text: str) -> str:
    task = """
Return a strict JSON object with exactly one key, "workExperience".
The value must be an array of job records ordered newest first.
Each record should include: company, title, startDate, endDate, and summary.
Populate summary with a list of short sentences describing responsibilities
or accomplishments. Use empty strings or [] when details are absent, and
only output the JSON object.
"""
    prompt = _build_prompt(task, indexed_text)
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, prompt)


def repair_prompt(
    task_name: str,
    previous_output: str,
    expected_keys: List[str],
    indexed_text: str,
) -> str:
    task = f"""
We previously asked for a JSON object with exactly the keys {expected_keys}.
The last model output was:
{previous_output or "<empty>"}
Please re-read the indexed text and return a clean JSON object that
matches the expected schema (one top-level key {expected_keys}),
contains only valid JSON characters, and keeps values for any fields you
can recover. Do not add any explanation or markdown.
"""
    prompt = _build_prompt(task, indexed_text or "")
    return _apply_template(tokenizer, SYSTEM_STRICT_JSON, prompt)

# prompt_basic / prompt_education / prompt_work / repair_prompt unchanged...

# -------------------------
# Model init + generation
# -------------------------
tokenizer = None
model = None


def download_model_if_needed() -> None:
    cfg_path = os.path.join(MODEL_PATH, "config.json")
    if os.path.isdir(MODEL_PATH) and os.path.exists(cfg_path):
        logger.info("Model already present at %s", MODEL_PATH)
        return
    logger.info("Model not found at %s (config: %s)", MODEL_PATH, cfg_path)
    logger.info("Downloading model %s to %s", REPO_ID, MODEL_CACHE_DIR)
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_CACHE_DIR,
        local_dir_use_symlinks=False
    )
    logger.info("Model download complete")


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
    logger.info("Initializing model...")
    t0 = time.time()
    download_model_if_needed()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    logger.info(
        "Tokenizer loaded | class=%s vocab=%s model_max_length=%s",
        tokenizer.__class__.__name__,
        getattr(tokenizer, "vocab_size", None),
        getattr(tokenizer, "model_max_length", None),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True,
    )
    logger.info("Model loaded | class=%s", model.__class__.__name__)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg_payload = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
        _log_text("Model config", _safe_json_dumps(cfg_payload))
    model.eval()
    logger.info("Model initialized on device=%s in %.2fs", DEVICE, time.time() - t0)


def generate_text(prompt: str, max_new_tokens: int, temperature: float, repetition_penalty: float) -> str:
    max_ctx = _get_max_ctx()
    safety = 32

    logger.debug("Prompt chars=%d", len(prompt))
    _log_text("Prompt", prompt)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_ctx - safety)
    input_len = int(enc["input_ids"].shape[1])
    allowed_new = min(max_new_tokens, max(0, max_ctx - input_len - safety))
    if allowed_new <= 0:
        logger.error("prompt_too_long: max_ctx=%s input_tokens=%s", max_ctx, input_len)
        raise ValueError(f"prompt_too_long: max_ctx={max_ctx} input_tokens={input_len}")

    inputs = {k: v.to(model.device) for k, v in enc.items()}
    do_sample = temperature > 0.0

    logger.debug(
        "Generating text | max_new=%d allowed_new=%d temp=%.3f rep=%.3f input_tokens=%d max_ctx=%d",
        max_new_tokens, allowed_new, temperature, repetition_penalty, input_len, max_ctx
    )

    t0 = time.time()
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
    logger.debug("Generation done in %.3fs", time.time() - t0)

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]
    logger.debug("Decoded chars=%d", len(decoded))
    _log_text("Decoded output", decoded)
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
    context_text: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    logger.info(
        "Running task=%s | max_new=%d temp=%.3f rep=%.3f repair_attempts=%d",
        task_name, max_new_tokens, temperature, repetition_penalty, repair_attempts
    )
    logger.debug("Task=%s expected_keys=%s", task_name, expected_keys)
    _log_text(f"Task {task_name} prompt", prompt_text)

    raw = generate_text(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )

    _log_text(f"Raw generation {task_name}", raw)

    cleaned = _basic_json_cleanup(raw)
    ok_shape = cleaned.startswith("{") and cleaned.endswith("}")
    parsed: Optional[Dict[str, Any]] = None

    def _keys_ok(d: Dict[str, Any]) -> bool:
        return isinstance(d, dict) and sorted(d.keys()) == sorted(expected_keys)

    if ok_shape:
        try:
            d = parse_json_strict(cleaned)
            if _keys_ok(d):
                parsed = d
                logger.info("Task=%s parsed successfully on first attempt", task_name)
        except Exception as e:
            logger.warning("Task=%s initial parse failed: %s", task_name, e)

    attempts_used = 0
    last = raw
    if parsed is None:
        logger.info("Task=%s entering repair loop", task_name)
        for i in range(1, repair_attempts + 1):
            attempts_used = i
            rep = repair_prompt(task_name, last, expected_keys, context_text)
            _log_text(f"Repair prompt {task_name} attempt={i}", rep)
            last = generate_text(
                rep,
                max_new_tokens=max(160, max_new_tokens // 2),
                temperature=0.0,
                repetition_penalty=repetition_penalty,
            )
            last_clean = _basic_json_cleanup(last)
            _log_text(f"Repair output {task_name} attempt={i}", last_clean)

            if last_clean.startswith("{") and last_clean.endswith("}"):
                try:
                    d = parse_json_strict(last_clean)
                    if _keys_ok(d):
                        parsed = d
                        logger.info("Task=%s repaired successfully on attempt=%d", task_name, i)
                        break
                except Exception as e:
                    logger.warning("Task=%s repair attempt=%d failed: %s", task_name, i, e)
                    continue

    meta = {
        "attempts": attempts_used,
        "latency_s": round(time.time() - t0, 3),
    }

    if parsed is None:
        meta["raw_preview"] = _basic_json_cleanup(last)[:2000]
        meta["error"] = "model_output_not_valid_json"
        logger.error("Task=%s failed after %d attempts", task_name, attempts_used)

    return parsed, meta

# -------------------------
# Main handler
# -------------------------
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    logger.debug("Raw job payload: %s", _safe_json_dumps(job))
    job_id = job.get("id") or job.get("job_id")
    logger.info("Received job id=%s", job_id)

    inp = job.get("input", {}) or {}
    logger.debug("Job input keys: %s", list(inp.keys()))
    logger.debug("Job input payload: %s", _safe_json_dumps(inp))

    part_basic = inp.get("part_basic", "")
    part_edu = inp.get("part_education", "")
    part_work = inp.get("part_work", "")
    logger.info(
        "Job id=%s parts length | basic=%d edu=%d work=%d",
        job_id, len(part_basic), len(part_edu), len(part_work)
    )
    _log_text("part_basic", part_basic)
    _log_text("part_education", part_edu)
    _log_text("part_work", part_work)

    if not isinstance(part_basic, str) or not isinstance(part_edu, str) or not isinstance(part_work, str):
        logger.warning("Bad request: non-string parts")
        return {"error": "bad_request", "detail": "part_basic/part_education/part_work must be strings"}

    all_text = (part_basic + part_edu + part_work).strip()
    if len(all_text) < 20:
        logger.warning("Bad request: text too short (len=%d)", len(all_text))
        return {"error": "bad_request", "detail": "parts are empty or too short"}

    temperature = float(inp.get("temperature", TEMPERATURE_DEFAULT))
    repetition_penalty = float(inp.get("repetition_penalty", REPETITION_PENALTY_DEFAULT))
    repair_attempts = int(inp.get("repair_attempts", REPAIR_ATTEMPTS_DEFAULT))
    return_indexed = bool(inp.get("return_indexed_text", False))

    max_basic = int(inp.get("max_new_tokens_basic", MAX_NEW_TOKENS_BASIC))
    max_edu = int(inp.get("max_new_tokens_education", MAX_NEW_TOKENS_EDU))
    max_work = int(inp.get("max_new_tokens_work", MAX_NEW_TOKENS_WORK))

    logger.info(
        "Job id=%s | temp=%.3f rep=%.3f repair=%d max_basic=%d max_edu=%d max_work=%d",
        job_id, temperature, repetition_penalty, repair_attempts, max_basic, max_edu, max_work
    )

    indexed_text = build_indexed_text_from_parts(part_basic, part_edu, part_work)
    logger.debug("Indexed text length: %d chars", len(indexed_text))
    _log_text("indexed_text", indexed_text)

    basic_out, basic_meta = run_task(
        task_name="basicInfo",
        prompt_text=prompt_basic(indexed_text),
        expected_keys=["basicInfo"],
        max_new_tokens=max_basic,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
        context_text=indexed_text,
    )

    edu_out, edu_meta = run_task(
        task_name="education",
        prompt_text=prompt_education(indexed_text),
        expected_keys=["education"],
        max_new_tokens=max_edu,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
        context_text=indexed_text,
    )

    work_out, work_meta = run_task(
        task_name="workExperience",
        prompt_text=prompt_work(indexed_text),
        expected_keys=["workExperience"],
        max_new_tokens=max_work,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        repair_attempts=repair_attempts,
        context_text=indexed_text,
    )

    if basic_out is None or edu_out is None or work_out is None:
        logger.error(
            "Job id=%s failed | basic_ok=%s edu_ok=%s work_ok=%s",
            job_id, basic_out is not None, edu_out is not None, work_out is not None
        )
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
        _log_text("response_error", _safe_json_dumps(resp))
        return resp

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

    _log_text("response_ok", _safe_json_dumps(resp_ok))
    logger.info("Job id=%s completed in %.3fs", job_id, time.time() - t0)
    return resp_ok


runpod.serverless.start({"handler": handler})
