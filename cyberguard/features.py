"""Lexical feature extraction. Looks only at URL text, never at the website."""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import parse_qsl, unquote, urlparse

import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking", "confirm",
    "password", "wallet", "bonus", "free", "support", "billing", "unlock", "suspend",
    "invoice", "recover",
]
BRANDS = [
    "paypal", "google", "microsoft", "apple", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "dhl", "fedex", "irs", "sbi", "hdfc", "icici", "linkedin",
]
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
    "rebrand.ly", "cutt.ly", "shorturl.at",
}
SUSPICIOUS_TLDS = {
    "xyz", "top", "click", "link", "info", "biz", "cc", "ru", "tk", "ml", "ga",
    "cf", "gq", "work", "zip", "mov", "support", "loan",
}
# Small offline list so we need neither the tldextract package nor internet access.
MULTI_PART_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.in", "co.jp", "com.br", "co.nz", "com.cn"}
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
HEX_IP_RE = re.compile(r"^0x[0-9a-f]+$", re.I)
VOWELS = set("aeiou")
RISKY_EXT = {".php", ".exe", ".zip", ".scr", ".apk", ".js", ".html", ".htm", ".jar"}

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "fragment_length",
    "dot_count", "subdomain_count", "hyphen_in_host", "digit_count", "digit_ratio",
    "host_digit_count", "letter_ratio", "uppercase_count", "special_char_count",
    "has_at_symbol", "has_ip_host", "has_port", "suspicious_word_count",
    "brand_in_subdomain", "brand_in_path", "is_shortener", "path_depth", "query_params",
    "host_entropy", "path_entropy", "double_slash_in_path", "percent_encoded_count",
    "is_punycode", "tld_length", "suspicious_tld", "longest_host_token",
    "host_vowel_ratio", "max_repeated_char_run", "file_extension_risky",
]


def scheme_free(url: str) -> str:
    """Drop the http(s):// prefix and a leading www. so every URL is treated alike.

    In popular public datasets the scheme and 'www.' often differ between the benign and
    malicious sources (an artifact of how the data was collected), so models learn that
    shortcut instead of real URL structure. Removing them avoids it.
    """
    u = str(url).strip()
    if "://" in u:
        u = u.split("://", 1)[1]
    return u[4:] if u.lower().startswith("www.") else u


def normalize(url: str) -> str:
    return "http://" + scheme_free(url)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def split_host(host: str) -> tuple[str, str, str]:
    """Return (subdomain, registered_domain_label, suffix)."""
    parts = host.split(".") if host else []
    if len(parts) >= 3 and ".".join(parts[-2:]) in MULTI_PART_SUFFIXES:
        return ".".join(parts[:-3]), parts[-3], ".".join(parts[-2:])
    if len(parts) >= 2:
        return ".".join(parts[:-2]), parts[-2], parts[-1]
    return "", host, ""


def _longest_run(text: str) -> int:
    best = run = 0
    prev = ""
    for ch in text:
        run = run + 1 if ch == prev else 1
        best, prev = max(best, run), ch
    return best


def extract_features(raw_url: str) -> dict:
    """Return a dict of numeric features for one URL (all zeros if unparseable)."""
    raw = str(raw_url).strip()
    try:
        url = normalize(raw)
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path, query, frag = parsed.path or "", parsed.query or "", parsed.fragment or ""
        lower = unquote(url).lower()
        try:
            has_port = int(parsed.port is not None)
        except ValueError:
            has_port = 1
        sub, domain, suffix = split_host(host)
        tld = suffix.split(".")[-1] if suffix else ""
        digits = sum(ch.isdigit() for ch in url)
        letters = sum(ch.isalpha() for ch in url)
        host_letters = [c for c in host if c.isalpha()]
        tokens = re.split(r"[.\-]", host) if host else []
        last_seg = path.rsplit("/", 1)[-1]
        ext = "." + last_seg.rsplit(".", 1)[-1].lower() if "." in last_seg else ""
        return {
            "url_length": len(url), "host_length": len(host), "path_length": len(path),
            "query_length": len(query), "fragment_length": len(frag),
            "dot_count": url.count("."),
            "subdomain_count": len(sub.split(".")) if sub else 0,
            "hyphen_in_host": host.count("-"), "digit_count": digits,
            "digit_ratio": digits / max(len(url), 1),
            "host_digit_count": sum(c.isdigit() for c in host),
            "letter_ratio": letters / max(len(url), 1),
            "uppercase_count": sum(c.isupper() for c in raw),
            "special_char_count": sum(url.count(c) for c in "@?&=%_~#$"),
            "has_at_symbol": int("@" in url),
            "has_ip_host": int(bool(IP_RE.match(host) or HEX_IP_RE.match(host))),
            "has_port": has_port,
            "suspicious_word_count": sum(w in lower for w in SUSPICIOUS_WORDS),
            "brand_in_subdomain": int(any(b in sub for b in BRANDS) and domain not in BRANDS),
            "brand_in_path": int(any(b in path.lower() for b in BRANDS)),
            "is_shortener": int(host.removeprefix("www.") in SHORTENERS),
            "path_depth": path.count("/"), "query_params": len(parse_qsl(query)),
            "host_entropy": shannon_entropy(host), "path_entropy": shannon_entropy(path),
            "double_slash_in_path": int("//" in path),
            "percent_encoded_count": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
            "is_punycode": int("xn--" in host),
            "tld_length": len(tld),
            "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
            "longest_host_token": max((len(t) for t in tokens), default=0),
            "host_vowel_ratio": (sum(c in VOWELS for c in host_letters) / len(host_letters)) if host_letters else 0.0,
            "max_repeated_char_run": _longest_run(url),
            "file_extension_risky": int(ext in RISKY_EXT),
        }
    except ValueError:
        return {name: 0 for name in FEATURE_NAMES}


def features_frame(urls) -> pd.DataFrame:
    """Vectorise many URLs into a DataFrame with a stable column order."""
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)


def risk_signals(raw_url: str) -> list[str]:
    """Human-readable red flags for one URL. Explanatory only, not the model's decision."""
    f = extract_features(raw_url)
    checks = [
        (f["has_ip_host"], "Host is a raw IP address instead of a domain name"),
        (f["has_at_symbol"], "Contains '@', which can hide the real destination"),
        (f["is_shortener"], "Uses a link shortener, so the final target is hidden"),
        (f["brand_in_subdomain"], "Brand name appears in the subdomain of an unrelated domain"),
        (f["suspicious_tld"], "Uses a top-level domain often seen in abuse"),
        (f["is_punycode"], "Contains punycode (possible look-alike characters)"),
        (f["suspicious_word_count"] >= 2, f"Contains {f['suspicious_word_count']} sensitive words (login, verify, ...)"),
        (f["hyphen_in_host"] >= 2, "Many hyphens in the host name"),
        (f["subdomain_count"] >= 3, "Unusually many subdomains"),
        (f["has_port"], "Uses an explicit non-default port"),
        (f["double_slash_in_path"], "Double slash inside the path"),
        (f["file_extension_risky"], "Points directly at an executable or script-like file"),
        (f["url_length"] > 100, "Very long URL"),
        (not str(raw_url).strip().lower().startswith("https://"), "Not using HTTPS"),
    ]
    return [msg for cond, msg in checks if cond]
