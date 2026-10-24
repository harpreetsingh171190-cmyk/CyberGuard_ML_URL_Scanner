import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cyberguard import FEATURE_NAMES, extract_features, features_frame
import pandas as pd
import pytest

from cyberguard.data import demo_dataset, load_dataset, split_data
from cyberguard.features import risk_signals, split_host
from cyberguard.modeling import explain, load_bundle, error_analysis, local_explanation, model_input, predict_urls, save_bundle, train_and_compare


def test_feature_count_and_order():
    assert list(extract_features("example.com")) == FEATURE_NAMES
    assert features_frame(["a.com", "b.org"]).shape == (2, len(FEATURE_NAMES))


def test_ip_and_at_symbol():
    f = extract_features("http://192.168.1.1/login")
    assert f["has_ip_host"] == 1
    assert extract_features("http://user@evil.example/x")["has_at_symbol"] == 1


def test_brand_impersonation():
    assert extract_features("http://paypal.secure-login.example.xyz")["brand_in_subdomain"] == 1
    assert extract_features("https://www.paypal.com/signin")["brand_in_subdomain"] == 0


def test_multi_part_suffix():
    assert split_host("login.bank.co.uk") == ("login", "bank", "co.uk")


def test_malformed_urls_do_not_crash():
    for bad in ["http://[::1", "", "   ", "http://a:b:c/", None]:
        assert len(extract_features(bad)) == len(FEATURE_NAMES)


def test_risk_signals():
    assert any("IP" in s for s in risk_signals("http://10.0.0.1/a"))


def test_label_cleaning_and_loading():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.csv")
        open(p, "w").write("URL,Type\nhttps://a.com,benign\nhttp://b.xyz/x,phishing\nhttps://a.com,benign\n")
        df = load_dataset(p)
        assert len(df) == 2 and sorted(df["label"]) == [0, 1]


def test_split_has_both_classes():
    tr, te = split_data(demo_dataset())
    assert tr["label"].nunique() == 2 and te["label"].nunique() == 2


def test_train_save_load_predict():
    tr, te = split_data(demo_dataset())
    results, fitted, probs, _ = train_and_compare(tr, te)
    with tempfile.TemporaryDirectory() as d:
        path = save_bundle(fitted["Random Forest"], "Random Forest", 0.5, {}, os.path.join(d, "m.joblib"), True)
        out = predict_urls(load_bundle(path), ["http://192.168.0.1/login.php", "https://www.python.org"])
    assert out.loc[0, "malicious_probability"] > out.loc[1, "malicious_probability"]


def test_local_explanation_points_to_real_features():
    tr, te = split_data(demo_dataset())
    _, fitted, _, _ = train_and_compare(tr, te)
    base = features_frame(tr["url"]).median().to_dict()
    bundle = {"model": fitted["Random Forest"], "threshold": 0.5, "baseline": base}
    top = local_explanation(bundle, "http://192.168.0.1/login.php", top=3)
    assert 0 < len(top) <= 3 and all(t["feature"] in FEATURE_NAMES for t in top)
    assert {"verdict", "signals", "top_features"} <= set(explain(bundle, "http://a.xyz/login"))


def test_ngram_hybrid_trains_and_predicts():
    from cyberguard.ngram import NgramHybrid
    df = demo_dataset()
    m = NgramHybrid().fit(model_input(df["url"], True), df["label"])
    p = m.predict_proba(model_input(["http://192.168.0.1/login.php", "https://www.python.org"], True))[:, 1]
    assert p[0] > p[1]


def test_time_split_orders_by_date_and_requires_dates():
    df = demo_dataset().sample(frac=1, random_state=1).reset_index(drop=True)  # mix classes over time
    df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D", tz="UTC")
    df = df.sample(frac=1, random_state=2)  # unsorted input
    tr, te = split_data(df, time_split=True)
    assert tr["date"].max() <= te["date"].min()
    with pytest.raises(ValueError):
        split_data(demo_dataset(), time_split=True)


def test_error_analysis_finds_mistakes():
    te = demo_dataset().iloc[[0, 30]].reset_index(drop=True)   # one legit, one malicious
    out = error_analysis(te, [0.9, 0.1], 0.5)                  # both wrong on purpose
    assert set(out["error_type"]) == {"false_positive", "false_negative"}


def test_clean_dataset_builder_is_balanced_and_host_only():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    from build_clean_dataset import build
    phish = pd.DataFrame({"url": [f"http://evil{i}.xyz/login?x={i}" for i in range(30)] + ["http://both.com/a"],
                          "date": pd.NaT, "label": 1})
    benign = pd.DataFrame({"url": [f"site{i}.com" for i in range(40)] + ["both.com"], "date": pd.NaT, "label": 0})
    out = build(phish, benign, per_class=20)
    assert out["label"].value_counts().nunique() <= 2 and "/" not in "".join(out["url"])
    assert out["url"].is_unique
