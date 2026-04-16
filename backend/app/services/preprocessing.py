from __future__ import annotations

import html
import re
from difflib import get_close_matches
from functools import lru_cache

TANGLISH_MAP = {
    "romba": "very",
    "nalla": "good",
    "nala": "good",
    "mosam": "bad",
    "semma": "awesome",
    "super": "excellent",
    "waste": "poor",
    "iruku": "is",
    "irukku": "is",
    "vandhuchu": "arrived",
    "varudhu": "coming",
    "aagudhu": "becoming",
    "illa": "not",
    "illai": "not",
    "konjam": "little",
    "adhigam": "more",
    "kammi": "less",
    "mudiyadhu": "cannot",
    "mudiyala": "cannot",
    "panna": "do",
    "pannudhu": "does",
    "pannrom": "doing",
    "pidichiruku": "liked",
    "kudukudhu": "giving",
    "kadupu": "irritation",
    "kastam": "difficult",
    "mokka": "bad",
    "jaasthi": "too much",
    "vera level": "excellent",
    "pakka": "definitely",
    "azhagana": "beautiful",
    "mass": "cool",
    "theriyudhu": "feels like",
    "theriyadhu": "does not seem",
    "sollanum": "should say",
    "paakanum": "should see",
    "kadaisila": "finally",
    "muzhusa": "completely",
    "sandai": "issue",
    "tension illai": "no worries",
    "jolly": "happy",
    "upchanam": "disappointment",
}

REMOVE_WORDS = {
    "la",
    "ah",
    "ku",
    "than",
    "machan",
    "bro",
    "nanbaa",
    "anna",
    "ayyo",
    "adei",
}

ASPECT_KEYWORDS = {
    "battery": [
        "battery",
        "charge",
        "charging",
        "charger",
        "battery life",
        "battery backup",
        "backup",
        "drain",
        "power",
        "adapter",
    ],
    "performance": [
        "performance",
        "speed",
        "fast",
        "slow",
        "lag",
        "hang",
        "smooth",
        "processor",
        "ram",
        "gaming",
        "multitasking",
        "heating",
        "heat",
        "thermal",
        "software",
        "boot",
        "loading",
    ],
    "display": [
        "display",
        "screen",
        "brightness",
        "resolution",
        "touchscreen",
        "refresh rate",
        "color",
        "colour",
        "clarity",
        "panel",
        "fhd",
        "hd",
        "oled",
        "backlight",
    ],
    "price": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "affordable",
        "worth",
        "budget",
        "overpriced",
        "discount",
        "money",
        "value",
        "value for money",
    ],
    "quality": [
        "quality",
        "build quality",
        "durable",
        "premium",
        "material",
        "finish",
        "sturdy",
        "solid",
        "fragile",
        "body",
    ],
    "design": [
        "design",
        "look",
        "style",
        "weight",
        "size",
        "slim",
        "thin",
        "light",
        "portable",
        "appearance",
    ],
    "audio": [
        "sound",
        "audio",
        "speaker",
        "volume",
        "bass",
        "noise",
        "mic",
        "microphone",
        "voice",
    ],
    "keyboard": [
        "keyboard",
        "touchpad",
        "trackpad",
        "keys",
        "typing",
        "fingerprint",
        "keypad",
    ],
    "camera": [
        "camera",
        "webcam",
        "video",
        "meeting",
        "conference",
        "zoom",
    ],
    "storage": [
        "storage",
        "ssd",
        "hdd",
        "disk",
        "memory",
        "space",
    ],
    "connectivity": [
        "wifi",
        "bluetooth",
        "usb",
        "hdmi",
        "port",
        "network",
        "connectivity",
    ],
    "delivery": [
        "delivery",
        "shipping",
        "courier",
        "arrived",
        "delivered",
        "late",
        "delay",
        "box",
        "packaging",
        "damaged",
        "broken",
        "seal",
    ],
    "service": [
        "service",
        "support",
        "refund",
        "replacement",
        "warranty",
        "seller",
        "customer care",
    ],
    "generic": [
        "good",
        "bad",
        "awesome",
        "great",
        "nice",
        "best",
        "perfect",
        "happy",
        "love",
        "satisfied",
        "disappointed",
        "amazing",
        "genuine",
    ],
}

BRAND_LIST = [
    "Apple",
    "ASUS",
    "Acer",
    "CHUWI",
    "Dell",
    "HP",
    "Infinix",
    "Lenovo",
    "LG",
    "MSI",
    "Primebook",
    "Samsung",
    "Ultimus",
    "Walker",
    "ZEBRONICS",
    "realme",
]

DOMAIN_VOCAB = sorted(
    {
        token
        for words in ASPECT_KEYWORDS.values()
        for phrase in words
        for token in phrase.lower().split()
    }
    | {
        "laptop",
        "macbook",
        "windows",
        "macos",
        "flipkart",
        "amazon",
        "office",
        "graphics",
        "performance",
        "awesome",
        "excellent",
        "average",
        "premium",
        "developer",
    }
)
DOMAIN_VOCAB_SET = set(DOMAIN_VOCAB)
ASPECT_PATTERNS = {
    aspect: [
        re.compile(rf"\b{re.escape(keyword.lower())}\b")
        for keyword in keywords
    ]
    for aspect, keywords in ASPECT_KEYWORDS.items()
}
TANGLISH_TOKENS = set(TANGLISH_MAP) | REMOVE_WORDS


def clean_text(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = value.lower()
    value = re.sub(r"http\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+|#\w+", " ", value)
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    value = re.sub(r"[^a-z0-9\s.,!?/-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_tanglish(text: str) -> str:
    normalized_words: list[str] = []
    for raw_word in text.split():
        word = raw_word.strip(".,!?/-")
        if word in REMOVE_WORDS:
            continue
        replacement = TANGLISH_MAP.get(word, word)
        normalized_words.extend(replacement.split())
    return " ".join(normalized_words)


def remove_repeats(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


@lru_cache(maxsize=20000)
def correct_word(word: str) -> str:
    if len(word) <= 3 or not word.isalpha() or word in DOMAIN_VOCAB_SET:
        return word
    matches = get_close_matches(word, DOMAIN_VOCAB, n=1, cutoff=0.88)
    return matches[0] if matches else word


def correct_spelling(text: str) -> str:
    return " ".join(correct_word(token) for token in text.split())


def preprocess_review(text: str) -> str:
    cleaned = clean_text(text)
    normalized = normalize_tanglish(cleaned)
    no_repeats = remove_repeats(normalized)
    corrected = correct_spelling(no_repeats)
    return re.sub(r"\s+", " ", corrected).strip()


def extract_aspects(text: str) -> list[str]:
    normalized = str(text or "").lower()
    found = [
        aspect
        for aspect, patterns in ASPECT_PATTERNS.items()
        if any(pattern.search(normalized) for pattern in patterns)
    ]
    return found or ["generic"]


def matches_aspect(text: str, aspect: str) -> bool:
    return any(pattern.search(text) for pattern in ASPECT_PATTERNS.get(aspect, []))


def extract_brand(product_name: str | None) -> str:
    value = str(product_name or "")
    for brand in BRAND_LIST:
        if brand.lower() in value.lower():
            return brand
    return "Other"


def detect_language(text: str | None) -> str:
    normalized = clean_text(str(text or ""))
    if not normalized:
        return "unknown"
    tokens = normalized.split()
    tanglish_hits = sum(1 for token in tokens if token in TANGLISH_TOKENS)
    return "tanglish" if tanglish_hits >= 1 else "english"


def rating_to_sentiment(rating: int | float | str) -> str:
    try:
        score = int(float(rating))
    except (TypeError, ValueError):
        return "neutral"
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text or ""))
    return [part.strip() for part in parts if part.strip()]


def build_aspect_contexts(review_text: str, normalized_review: str, aspects: list[str]) -> dict[str, str]:
    raw_sentences = split_sentences(review_text)
    if not raw_sentences:
        raw_sentences = [str(review_text or "")]

    normalized_sentences = [preprocess_review(sentence) for sentence in raw_sentences]
    contexts: dict[str, str] = {}

    for aspect in aspects:
        if aspect == "generic":
            contexts[aspect] = review_text
            continue
        matched_sentences = [
            raw_sentence
            for raw_sentence, normalized_sentence in zip(raw_sentences, normalized_sentences)
            if matches_aspect(normalized_sentence, aspect)
        ]
        contexts[aspect] = " ".join(matched_sentences).strip() or review_text or normalized_review

    return contexts


def join_aspects(aspects: list[str]) -> str:
    return "|".join(aspects)


def normalize_aspect_string(value: list[str] | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return join_aspects([item for item in value if item]) or None
    aspects = split_aspects(value)
    return join_aspects(aspects) or None


def split_aspects(value: str | None) -> list[str]:
    if value is None:
        return []
    string_value = str(value).strip()
    if not string_value or string_value.lower() == "nan":
        return []
    delimiter = "|" if "|" in string_value else ","
    return [item.strip() for item in string_value.split(delimiter) if item.strip()]


def build_rag_text(*, brand: str, product_name: str, review: str, sentiment: str, aspects: list[str], rating: int) -> str:
    aspect_text = ", ".join(aspects) if aspects else "generic"
    return (
        f"Brand: {brand}. Product: {product_name}. Sentiment: {sentiment}. "
        f"Rating: {rating}/5. Aspects: {aspect_text}. Review: {review}"
    )
