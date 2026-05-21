import pickle
import re
from pathlib import Path
from urllib.parse import urlparse

from scipy.sparse import csr_matrix, hstack

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account",
    "update", "password", "banking", "paypal", "bank",
]

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".cf", ".ga", ".gq",
    ".xyz", ".top", ".click", ".work", ".gift", ".free",
}

IP_REGEX = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def extract_features(url: str) -> list:
    if "://" not in url:
        url_full = "http://" + url
    else:
        url_full = url

    try:
        parsed = urlparse(url_full)
        domain = parsed.netloc or url
    except ValueError:
        domain = url

    return [
        len(url),
        len(domain),
        url.count("."),
        url.count("-"),
        url.count("/"),
        1 if IP_REGEX.match(domain) else 0,
        1 if "@" in url else 0,
        1 if url.startswith("https") else 0,
        domain.count("."),
        int(any(w in url for w in SUSPICIOUS_WORDS)),
        1 if len(url) > 75 else 0,
    ]


def _explain(url: str, feats: list) -> list[str]:
    (_, _, dot_cnt, dash_cnt, _, is_ip,
     has_at, is_https, _, has_suspicious_word, is_long) = feats

    reasons: list[str] = []

    if is_long:
        reasons.append("Длина URL превышает норму")
    if is_ip:
        reasons.append("IP-адрес вместо доменного имени")
    if has_at:
        reasons.append("Символ @ в URL")
    if not is_https:
        reasons.append("Отсутствует HTTPS")
    if has_suspicious_word:
        found = [w for w in SUSPICIOUS_WORDS if w in url]
        if found:
            reasons.append(f"Подозрительные ключевые слова ({', '.join(found)})")
    if dot_cnt > 4:
        reasons.append(f"Много точек в URL ({dot_cnt})")
    if dash_cnt > 2:
        reasons.append(f"Много дефисов в URL ({dash_cnt})")

    url_lower = url.lower().split("?")[0].split("#")[0]
    for tld in SUSPICIOUS_TLDS:
        if url_lower.endswith(tld) or f"{tld}/" in url_lower:
            reasons.append(f"Подозрительный домен верхнего уровня ({tld})")
            break

    return reasons


class PhishingPredictor:
    def __init__(self, model_dir: str = ".") -> None:
        base = Path(model_dir)
        with open(base / "model.pkl", "rb") as f:
            self._model = pickle.load(f)
        with open(base / "scaler.pkl", "rb") as f:
            self._scaler = pickle.load(f)
        with open(base / "tfidf.pkl", "rb") as f:
            self._tfidf = pickle.load(f)

    def predict(self, url: str) -> dict:
        url = url.lower().strip()
        feats = extract_features(url)
        manual = self._scaler.transform([feats])
        tfidf_vec = self._tfidf.transform([url])
        x = hstack([csr_matrix(manual), tfidf_vec])
        proba = float(self._model.predict_proba(x)[0, 1])
        is_phishing = proba > 0.5
        return {
            "url": url,
            "is_phishing": is_phishing,
            "probability": proba,
            "label": "ФИШИНГ" if is_phishing else "ЛЕГИТИМНЫЙ",
            "suspicious_features": _explain(url, feats) if is_phishing else [],
        }

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict(u) for u in urls]
