"""Lab 1 starter: versioned bilingual preprocessing for Bayan."""

import re
import unicodedata

PREPROC_VERSION = "1.2.0"

_TATWEEL = "ـ"

_PHONE_RE = re.compile(r"(?:\+?966|0)5\d{8}")
_NATIONAL_ID_RE = re.compile(r"\b[12]\d{9}\b")

_MULTISPACE_RE = re.compile(r"\s+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")

def normalize(text: str) -> str:
    """Return deterministic Bayan normalisation while preserving task signal."""
    # TODO(Lab 1): implement the course normalisation contract.

    # 1) Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2) Remove tatweel
    text = text.replace(_TATWEEL, "")

    # 3) Collapse repeated characters to maximum 2
    text = _REPEAT_RE.sub(r"\1\1", text)

    # 4) Normalize whitespace
    text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


def mask_pii(text: str) -> str:
    """Mask supported phone numbers and Saudi national-ID-shaped values."""
    # TODO(Lab 1): replace supported PII with <PHONE> / <NATIONAL_ID>.

    text = _PHONE_RE.sub("<PHONE>", text)
    text = _NATIONAL_ID_RE.sub("<NATIONAL_ID>", text)

    return text


def preprocess(text: str) -> str:
    """Apply the shared train/eval/serve preprocessing contract."""
    # TODO(Lab 1): compose masking and normalisation in the intended order.
    return normalize(mask_pii(text))
