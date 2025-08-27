from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import requests

app = FastAPI()

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "llama3:8b"

VALID_LABELS = [
    "explicit_nudity", "suggestive", "violence", "disturbing_content",
    "rude_gestures", "alcohol", "drugs", "tobacco", "hate_speech", "safe"
]

class Descriptor(BaseModel):
    text: List[str]
    long_desc: str = ""
    short_desc: str = ""

class Provider(BaseModel):
    descriptor: Descriptor
    id: str

class Catalog(BaseModel):
    bpp_providers: List[Provider] = Field(alias="bpp/providers")

class Message(BaseModel):
    catalog: Catalog

class InputSchema(BaseModel):
    message: Message

def classify_text_with_ollama(text: str) -> List[str]:
    prompt = f"""
    you expert content moderator. examine the following content classify it into one or more categories from the list:
    {VALID_LABELS}

    Rules:
    - Pick ALL applicable categories.
    - If any category other than 'safe' applies, do NOT include 'safe'.
    - Only output a comma-separated list of labels.

    Content: {text}
    """

    response = requests.post(OLLAMA_API_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    if response.status_code == 200:
        raw_output = response.json().get("response", "").lower()
        labels = [lbl.strip() for lbl in raw_output.replace("\n", ",").split(",") if lbl.strip()]
        # Keep only valid labels
        labels = [lbl for lbl in labels if lbl in VALID_LABELS]

        # Enforce safe logic
        if any(lbl for lbl in labels if lbl != "safe"):
            labels = [lbl for lbl in labels if lbl != "safe"]
        if not labels:
            labels = ["safe"]

        return labels
    else:
        return ["safe"]

@app.post("/classify")
def classify(input_data: InputSchema):
    provider = input_data.message.catalog.bpp_providers[0]
    combined_text = " ".join(provider.descriptor.text) + " " + provider.descriptor.long_desc + " " + provider.descriptor.short_desc
    labels = classify_text_with_ollama(combined_text)

    return {
        "type": "CATALOG-ERROR" if labels != ["safe"] else "CATALOG",
        "code": "999999" if labels != ["safe"] else "111111",
        "path": "/classify",
        "message": ", ".join(labels),
        "test_type": "recommendation"
    }

