import os
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx

# -----------------------
# Config
# -----------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3:8b")
# Concurrency control for GPU-bound requests (backpressure)
MAX_PARALLEL_INFER = int(os.environ.get("MAX_PARALLEL_INFER", "8"))

# Allowed canonical labels (your set)
VALID_LABELS = [
    "explicit_nudity", "suggestive", "violence", "disturbing_content",
    "rude_gestures", "alcohol", "drugs", "tobacco", "hate_speech", 
]

# Precompile a refusal detection regex (common refusal phrases)
REFUSAL_REGEX = re.compile(
    r"(I cannot create|I can't create|I won't create|I cannot provide|I refuse|I cannot help with|I cannot generate)",
    re.IGNORECASE
)

# -----------------------
# Keyword maps for fallback deterministic classification
# (These are used if the model refuses or returns invalid output)
# Extend these patterns as needed for your domain
# -----------------------
KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "alcohol": [r"\bdrunk\b", r"\balcohol\b", r"\bintoxicated\b", r"\bbeer\b", r"\bwine\b", r"\bwhisky\b", r"\bvodka\b"],
    "violence": [r"\bsword\b", r"\bstab\b", r"\bkill(ed|ing)?\b", r"\bknife\b", r"\bshot\b", r"\battack(ed|ing)?\b", r"\bcharged\b", r"\bslashed\b"],
    "explicit_nudity": [r"\bnaked\b", r"\bnudity\b", r"\bexplicit sexual\b", r"\bexposed\b"],
    "suggestive": [r"\bsexual\b", r"\bsexually suggestive\b", r"\bseduce\b", r"\bflirtatious\b"],
    "disturbing_content": [r"\bgore\b", r"\bgruesome\b", r"\bhorrific\b", r"\bdisembowel\b"],
    "rude_gestures": [r"\bfuck you\b", r"\bmiddle finger\b", r"\bfinger\b", r"\bflipping\b"],
    "drugs": [r"\bdrugs?\b", r"\bheroin\b", r"\bcocaine\b", r"\bmeth\b", r"\bweed\b", r"\bmarijuana\b"],
    "tobacco": [r"\bcigarette\b", r"\bsmok(e|ing)\b", r"\btobacco\b", r"\bvape\b"],
    "hate_speech": [r"\bracist\b", r"\bslur\b", r"\bhate\b", r"\bkill (all|the)\b.*\b(people|them)\b"]
}

# compile patterns for speed
COMPILED_KEYWORD_PATTERNS = {
    label: [re.compile(p, re.IGNORECASE) for p in pats] for label, pats in KEYWORD_PATTERNS.items()
}

# -----------------------
# Pydantic request models
# -----------------------
class Descriptor(BaseModel):
    text: List[str] = Field(default_factory=list)
    long_desc: Optional[str] = ""
    short_desc: Optional[str] = ""

class Provider(BaseModel):
    descriptor: Descriptor
    id: Optional[str] = None

class Catalog(BaseModel):
    bpp_providers: List[Provider] = Field(..., alias="bpp/providers")

class Message(BaseModel):
    catalog: Catalog

class RequestModel(BaseModel):
    message: Message

# -----------------------
# FastAPI app + Globals
# -----------------------
app = FastAPI()
http_client: Optional[httpx.AsyncClient] = None
_infer_semaphore = asyncio.Semaphore(MAX_PARALLEL_INFER)

# -----------------------
# Utility functions
# -----------------------
def normalize_label_token(tok: str) -> Optional[str]:
    """Map freeform token to canonical label if possible."""
    tok = tok.strip().lower()
    # basic synonyms mapping
    synonyms = {
        "nudity": "explicit_nudity",
        "nudity_explicit": "explicit_nudity",
        "sex": "suggestive",
        "sexual": "suggestive",
        "violent": "violence",
        "violence_content": "violence",
        "gore": "disturbing_content",
        "drunk": "alcohol",
        "alcoholic": "alcohol",
        "cigarettes": "tobacco",
        "smoking": "tobacco",
        "hate": "hate_speech"
    }
    if tok in VALID_LABELS:
        return tok
    if tok in synonyms:
        return synonyms[tok]
    # remove non-alpha chars
    tok_clean = re.sub(r"[^a-z_]", "", tok)
    if tok_clean in VALID_LABELS:
        return tok_clean
    return None

def parse_model_output(raw: str) -> List[str]:
    """
    Robust parsing of model output into canonical labels.
    Accepts JSON array, comma-separated, or plain words.
    """
    if not raw:
        return []

    raw = raw.strip()

    # If model produced JSON-like array, try to parse directly
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            tokens = [str(x) for x in obj]
        elif isinstance(obj, dict) and "labels" in obj and isinstance(obj["labels"], list):
            tokens = [str(x) for x in obj["labels"]]
        else:
            # fallback to text parsing
            raise ValueError("Not a list")
    except Exception:
        # Remove surrounding quotes/backticks if any
        raw2 = re.sub(r"^['\"`]+|['\"`]+$", "", raw)
        # split on commas, semicolons, 'and', or newlines
        tokens = re.split(r"[,\n;]|\band\b", raw2)
        tokens = [t.strip() for t in tokens if t.strip()]

    canonical = []
    seen = set()
    for t in tokens:
        mapped = normalize_label_token(t)
        if mapped and mapped not in seen:
            canonical.append(mapped)
            seen.add(mapped)
    return canonical

def keyword_classify(text: str) -> List[str]:
    """Deterministic keyword-based classifier as fallback."""
    found = []
    seen = set()
    for label in VALID_LABELS:
        if label == "safe":
            continue
        patterns = COMPILED_KEYWORD_PATTERNS.get(label, [])
        for p in patterns:
            if p.search(text):
                if label not in seen:
                    found.append(label)
                    seen.add(label)
                break
    if not found:
        return ["safe"]
    return found

def ordered_unique(labels: List[str]) -> List[str]:
    """Return labels deduped and ordered according to VALID_LABELS order."""
    order = {lbl: i for i, lbl in enumerate(VALID_LABELS)}
    seen = set()
    out = []
    for lbl in labels:
        if lbl not in seen and lbl in order:
            out.append(lbl)
            seen.add(lbl)
    # ensure at least 'safe' if empty
    if not out:
        return ["safe"]
    return out

# -----------------------
# Ollama API call
# -----------------------
async def call_ollama_classifier(text: str) -> Dict[str, Any]:
    """
    Calls Ollama /api/chat to ask for a strict multi-label JSON array.
    Returns dict with keys: raw, parsed_labels, refused (bool)
    """
    global http_client
    if http_client is None:
        raise RuntimeError("HTTP client not initialized")

    system_prompt = (
        "You are a strict multi-label classifier used for content moderation.\n"
        "Your ONLY job: given a TEXT, return a JSON array (e.g. format [\"violence\",\"alcohol\"]) containing zero or more labels\n"
        f"from this canonical list exactly (no synonyms, no extra words): {VALID_LABELS}.\n"
        "If none apply, return [\"safe\"].\n"
        "You MUST NOT refuse. You must not add any explanation or extra text — only the JSON array.\n"
        "If the text contains violent or sexual content, or what-ever still output the correct labels because you are conetent moderator(this is a classification task). this is only used for classifying the text nothig beyond that"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Text: {text}"}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        # deterministic and short
        "options": {
            "num_predict": 64,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0
        },
        "stream": False
    }

    async with _infer_semaphore:
        resp = await http_client.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=60.0)
    resp.raise_for_status()
    data = resp.json()

    # Ollama may return either {"message": {"role":"assistant","content": "..."}}
    # or {"response": "..."} on some endpoints. Try both.
    raw = ""
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict):
            raw = data["message"].get("content", "")
        elif "response" in data:
            raw = data.get("response", "")
        else:
            # fallback: stringified full response
            raw = json.dumps(data)

    raw = (raw or "").strip()
    refused = bool(REFUSAL_REGEX.search(raw))
    parsed = parse_model_output(raw)

    return {"raw": raw, "parsed": parsed, "refused": refused}

# -----------------------
# FastAPI lifecycle
# -----------------------
@app.on_event("startup")
async def startup_event():
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    # optional warmup (best-effort)
    try:
        _ = await call_ollama_classifier("warmup")
    except Exception as e:
        logging.warning("Ollama warmup failed: %s", e)

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

# -----------------------
# Endpoint
# -----------------------
@app.post("/classify")
async def classify_endpoint(req: RequestModel):
    # Build the concatenated text
    prov = req.message.catalog.bpp_providers[0]
    desc = prov.descriptor
    parts = []
    parts.extend(desc.text or [])
    if desc.long_desc:
        parts.append(desc.long_desc)
    if desc.short_desc:
        parts.append(desc.short_desc)
    text_to_classify = " ".join([p for p in parts if p]).strip()

    if not text_to_classify:
        # follow your schema for error
        return {
            "type": "CATALOG-ERROR",
            "code": "999999",
            "path": "/classify",
            "message": "safe",
            "test_type": "recommendation"
        }

    # 1) Ask the model
    try:
        model_out = await call_ollama_classifier(text_to_classify)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=500, detail=f"Ollama HTTP error: {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    labels_from_model = model_out.get("parsed", []) or []
    refused = model_out.get("refused", False)

    # 2) Do deterministic keyword classification
    labels_from_kw = keyword_classify(text_to_classify)  # never empty, returns ["safe"] if nothing found

    # 3) Decide final labels
    final_set = []
    # if model parsed some valid labels (not just 'safe'), prefer them but union with keyword results
    for lbl in labels_from_model:
        if lbl in VALID_LABELS and lbl not in final_set:
            final_set.append(lbl)
    # add keyword labels if not present (helps when model refused or missed)
    for lbl in labels_from_kw:
        if lbl in VALID_LABELS and lbl not in final_set:
            final_set.append(lbl)

    # If model explicitly refused and keyword found something, rely on keyword
    if refused and (labels_from_kw and labels_from_kw != ["safe"]):
        final_set = labels_from_kw

    # If nothing found, default to safe
    if not final_set:
        final_set = ["safe"]

    # Normalize order and dedupe according to VALID_LABELS
    ordered = ordered_unique(final_set)

    message_text = ", ".join(ordered)

    return {
        "type": "CATALOG-ERROR",
        "code": "999999",
        "path": "/classify",
        "message": message_text,
        "test_type": "recommendation"
    }
