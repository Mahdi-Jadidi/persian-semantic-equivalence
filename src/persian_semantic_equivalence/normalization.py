import re

ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه", "ؤ": "و", "إ": "ا", "أ": "ا"})


def normalize_persian(text: object) -> str:
    value = str(text).translate(ARABIC_TO_PERSIAN).replace("\u200f", "").replace("\u200e", "")
    value = re.sub(r"[\u064b-\u065f\u0670]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize_words(text: object) -> list[str]:
    return re.findall(r"[\w\u200c]+", normalize_persian(text), flags=re.UNICODE)
