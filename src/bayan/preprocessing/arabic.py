"""Lab 4: explicit Arabic normalisation profiles for model input."""
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ArabicProfile:
    name: str
    dediacritize: bool = False


# The supplied golden fixture defines bayan_ar_v1. The conservative profile
# is a local choice because the starter does not specify a second profile.
_PROFILES = {
    "bayan_ar_v1": str.maketrans({
        "\u0623": "\u0627", "\u0625": "\u0627", "\u0622": "\u0627",
        "\u0624": "\u0648", "\u0626": "\u064a",
        "\u0649": "\u064a", "\u0629": "\u0647",
    }),
    "bayan_ar_preserve_v1": {},
}
_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7-\u06e8\u06ea-\u06ed]")


def normalize_arabic(text: str, profile: ArabicProfile) -> str:
    """Return a model copy; retain the original text separately for display.

    Both profiles apply NFC, strip tatweel and collapse whitespace. Only
    bayan_ar_v1 folds letter variants. Diacritic removal is opt-in.
    Do not apply model-text character offsets to the original display text.
    """
    if profile.name not in _PROFILES:
        raise ValueError(f"Unknown Arabic profile: {profile.name!r}")
    model_text = unicodedata.normalize("NFC", text).replace("\u0640", "")
    if profile.dediacritize:
        model_text = _DIACRITICS.sub("", model_text)
    model_text = model_text.translate(_PROFILES[profile.name])
    return " ".join(model_text.split())


def segment(text: str) -> list[str]:
    # TODO(Lab 4): wire the chosen CAMeL Tools clitic segmentation scheme.
    raise NotImplementedError
