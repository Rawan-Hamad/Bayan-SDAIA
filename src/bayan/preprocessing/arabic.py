"""Lab 4: explicit Arabic normalisation profiles for model input."""
from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=1)
def _clitic_tokenizer():
    """Load once, on demand; baseline runs do not require CAMeL data."""
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tokenizers.morphological import MorphologicalTokenizer

    try:
        disambiguator = MLEDisambiguator.pretrained("calima-msa-r13")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Install CAMeL resources in the active environment: "
            "camel_data -i light"
        ) from exc
    return MorphologicalTokenizer(
        disambiguator, scheme="d3tok", split=True, diac=False,
    )


@lru_cache(maxsize=32768)
def _segment_word(word: str) -> tuple[str, ...]:
    # Preserve English identifiers, underscores, dates and punctuation.
    if not re.search("[\u0621-\u063a\u0641-\u064a]", word):
        return (word,)
    return tuple(piece for piece in _clitic_tokenizer().tokenize([word]) if piece)


def segment(text: str) -> list[str]:
    """D3 clitics with CAMeL '+' markers; no destructive letter folding.

    Apply the same wordwise MLE path in training, evaluation and inference.
    The analysis database is MSA-based; dialect coverage is not guaranteed.
    """
    return [piece for word in text.split() for piece in _segment_word(word)]

