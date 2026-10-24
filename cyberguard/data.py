"""Dataset loading, label cleaning and leakage-aware splitting."""
from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import normalize, split_host

SEED = 42
LEGIT_LABELS = {"0", "benign", "legitimate", "legit", "good", "safe"}
URL_COLS = ["url", "urls", "link"]
LABEL_COLS = ["label", "type", "class", "status", "result"]
DATE_COLS = ["date", "timestamp", "time", "first_seen", "submission_time", "created", "created_at"]

DEMO_LEGIT = [
    "https://www.google.com", "https://www.wikipedia.org/wiki/Python_(programming_language)",
    "https://github.com/python/cpython", "https://www.python.org/downloads/",
    "https://stackoverflow.com/questions/tagged/python", "https://www.bbc.com/news",
    "https://docs.microsoft.com/en-us/learn/", "https://www.nasa.gov/missions",
    "https://www.youtube.com/watch?v=abc123", "https://www.amazon.in/gp/bestsellers",
    "https://www.linkedin.com/in/example", "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://www.khanacademy.org/math", "https://www.kernel.org", "https://pypi.org/project/pandas/",
    "https://www.mozilla.org/en-US/firefox/", "https://scikit-learn.org/stable/",
    "https://www.coursera.org/learn/machine-learning", "https://news.ycombinator.com",
    "https://www.reddit.com/r/learnpython", "https://www.ubuntu.com/download", "https://www.gnu.org/licenses/",
    "https://www.cloudflare.com/learning/", "https://www.w3schools.com/python/",
]
DEMO_MALICIOUS = [
    "http://secure-login.paypal-verify-account.example-xyz.com/signin?id=99283",
    "http://192.168.45.12/bank/login/update.php", "http://bit.ly/3xFr9aQ",
    "http://account-update.secure-banking-portal.info/verify/confirm", "http://free-bonus-wallet.top/claim?user=123&ref=88",
    "http://login.microsoft-account-verify.example-co.net/password/reset", "http://www.amaz0n-secure-signin.biz/ap/update",
    "http://203.0.113.77:8080/login.html", "http://verify-your-account-now.click/secure//confirm",
    "http://paypa1-support-update.example-site.org/login", "http://secure.bank-update-center.info/account/verify/login",
    "http://apple-id-locked.example-help.xyz/unlock?session=ab12cd34ef56", "http://netflix-billing-update.example-pay.top/signin",
    "http://user@fake-login-page.example.biz/verify", "http://confirm-password-reset.free-gift-card.example.ru/a/b/c/d",
    "http://my-secure-account.verify-login-update.example.cc/banking", "http://dhl-parcel-tracking.example-delivery.link/confirm?id=5543",
    "http://login-update-secure-bank-account.example-123.xyz", "http://198.51.100.9/wallet/verify/login", "http://tinyurl.com/y8abcd12",
    "http://webmail-quota-verify.example-mail.info/update/password", "http://secure-signin.bank-account-confirm.example-net.com/x",
    "http://irs-refund-claim.example-gov.top/verify/account", "http://whatsapp-gold-free-download.example.click/install",
]


def demo_dataset() -> pd.DataFrame:
    """Tiny hand-written set so the code runs out of the box. Scores on it mean nothing."""
    return pd.DataFrame({
        "url": DEMO_LEGIT + DEMO_MALICIOUS,
        "label": [0] * len(DEMO_LEGIT) + [1] * len(DEMO_MALICIOUS),
    })


def load_dataset(path: str, max_rows: int | None = 100_000) -> pd.DataFrame:
    """Load a CSV with a URL column and a label/type column. Returns columns url, label (0/1)."""
    raw = pd.read_csv(path)
    cols = {c.lower(): c for c in raw.columns}
    url_col = next((cols[c] for c in URL_COLS if c in cols), None)
    label_col = next((cols[c] for c in LABEL_COLS if c in cols), None)
    if url_col is None or label_col is None:
        raise ValueError(f"Need a url column and a label/type column. Found: {list(raw.columns)}")
    date_col = next((cols[c] for c in DATE_COLS if c in cols), None)
    keep = [url_col, label_col] + ([date_col] if date_col else [])
    df = raw[keep].rename(columns={url_col: "url", label_col: "raw_label", **({date_col: "date"} if date_col else {})})
    if date_col:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["url", "raw_label"])
    df = df.drop_duplicates(subset="url")
    df["label"] = df["raw_label"].astype(str).str.strip().str.lower().map(lambda v: 0 if v in LEGIT_LABELS else 1)
    if max_rows and len(df) > max_rows:
        df = df.sample(max_rows, random_state=SEED)
    return df[["url", "label"] + (["date"] if date_col else [])].reset_index(drop=True)


def registered_domain(url: str) -> str:
    try:
        host = (urlparse(normalize(url)).hostname or "").lower()
    except ValueError:  # malformed URLs (e.g. broken IPv6 brackets) exist in real data
        host = str(url).lower()
    _, domain, suffix = split_host(host)
    return f"{domain}.{suffix}" if suffix else host


def split_data(df: pd.DataFrame, test_size: float = 0.2, group_by_domain: bool = True, time_split: bool = False):
    """Train/test split. Grouping by domain keeps one site from landing on both sides (leakage).

    Falls back to a plain stratified split when grouping is impossible (e.g. tiny data).
    Returns (train_df, test_df).
    """
    if time_split:
        if "date" not in df or df["date"].notna().sum() < 0.9 * len(df):
            raise ValueError("--time-split needs a date/timestamp column with valid dates in your CSV")
        ordered = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        cut = int(len(ordered) * (1 - test_size))
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
        if train["label"].nunique() < 2 or test["label"].nunique() < 2:
            raise ValueError("Time split left only one class on one side; use a larger or more mixed dataset")
        return train.reset_index(drop=True), test.reset_index(drop=True)
    if group_by_domain and len(df) >= 50:
        groups = df["url"].map(registered_domain)
        if groups.nunique() > 10:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
            tr, te = next(gss.split(df, df["label"], groups))
            if df["label"].iloc[tr].nunique() == 2 and df["label"].iloc[te].nunique() == 2:
                return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)
