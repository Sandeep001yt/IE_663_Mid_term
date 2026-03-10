#!/usr/bin/env python3
"""
app.py  –  Streamlit UI for TIGBShareClassifier inference.

Run with:
    streamlit run app.py
"""

import json
import os
import tempfile

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Multimodal Sentiment Classifier",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: #0d0d0f;
        color: #e8e6e1;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #f5c842 0%, #f0855a 60%, #e84393 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #888;
        font-size: 0.95rem;
        font-family: 'DM Mono', monospace;
    }

    /* Card wrapper */
    .card {
        background: #17171a;
        border: 1px solid #2a2a30;
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
    }

    /* Result badge */
    .badge {
        display: inline-block;
        padding: 0.45rem 1.2rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-positive { background:#1a3a2a; color:#4ade80; border:1px solid #4ade80; }
    .badge-neutral  { background:#1e2a3a; color:#60a5fa; border:1px solid #60a5fa; }
    .badge-negative { background:#3a1a1a; color:#f87171; border:1px solid #f87171; }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 0.5rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
    }
    .prob-label { width: 68px; color: #aaa; }
    .prob-track {
        flex: 1;
        height: 8px;
        background: #2a2a30;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }
    .fill-positive { background: linear-gradient(90deg, #22c55e, #4ade80); }
    .fill-neutral  { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
    .fill-negative { background: linear-gradient(90deg, #ef4444, #f87171); }
    .prob-value { width: 46px; text-align:right; color:#e8e6e1; }

    /* Confidence ring (SVG) */
    .conf-wrap { text-align: center; margin: 0.4rem 0 1rem; }
    .conf-pct  { font-size: 2rem; font-weight: 800; color: #f5c842; }
    .conf-label{ font-size: 0.75rem; color:#777; font-family:'DM Mono',monospace; }

    /* Streamlit widget tweaks */
    label, .stTextArea label, .stFileUploader label {
        color: #ccc !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .stTextArea textarea {
        background: #1c1c21 !important;
        color: #e8e6e1 !important;
        border: 1px solid #2a2a30 !important;
        border-radius: 10px !important;
        font-family: 'DM Mono', monospace !important;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #f5c842, #f0855a);
        color: #0d0d0f;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.05em;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    div[data-testid="stSidebar"] { background: #101013; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Multimodal Classifier</h1>
        <p>text + image → sentiment prediction</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar – model settings ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    checkpoint_path = st.text_input(
        "Checkpoint path",
        value="twitter_GB_best_model_0.6872.pth",
        help="Path to your .pth model checkpoint",
    )
    config_path = st.text_input(
        "Config JSON path",
        value="./data/twitter.json",
        help="Path to the dataset config JSON",
    )
    merge_alpha = st.slider(
        "Merge alpha  (text ↔ image weight)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = image only · 1 = text only",
    )
    gpu_id = st.text_input("CUDA device", value="0")

    st.markdown("---")
    st.markdown(
        "<small style='color:#555;font-family:DM Mono,monospace'>"
        "Labels: negative · neutral · positive</small>",
        unsafe_allow_html=True,
    )

# ── Lazy imports (avoids crashing if deps missing at import time) ─────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model_cached(checkpoint: str, cfg_path: str, _gpu: str):
    """Load and cache the TIGBShareClassifier."""
    from data.template import config as base_config
    from model.TextImage import TIGBShareClassifier
    from utils.utils import deep_update_dict

    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu

    cfg = base_config
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = deep_update_dict(json.load(f), cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TIGBShareClassifier(config=cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    added_v = {int(k.split(".")[1]) for k in state_dict if k.startswith("additional_layers_v.")}
    added_a = {int(k.split(".")[1]) for k in state_dict if k.startswith("additional_layers_a.")}
    for _ in range(len(added_v)):
        model.add_layer(is_a=False)
    for _ in range(len(added_a)):
        model.add_layer(is_a=True)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, device


def preprocess(text: str, pil_image: Image.Image, device: str):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    enc = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    token_type_ids = enc["token_type_ids"].to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    image = transform(pil_image.convert("RGB")).unsqueeze(0).float().to(device)
    return input_ids, attention_mask, token_type_ids, image


@torch.no_grad()
def run_predict(text, pil_image, model, device, alpha):
    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
    ids, mask, ttype, img = preprocess(text, pil_image, device)
    o_t, o_i    = model(ids, mask, ttype, img)
    out_t, _, _ = model.classfier(o_t, is_a=True)
    out_i, _, _ = model.classfier(o_i, is_a=False)
    out         = alpha * out_t + (1 - alpha) * out_i
    probs       = F.softmax(out, dim=1).squeeze(0)
    pred_idx    = probs.argmax().item()
    return {
        "label": LABEL_MAP[pred_idx],
        "confidence": round(probs[pred_idx].item(), 4),
        "probabilities": {LABEL_MAP[i]: round(p, 4) for i, p in enumerate(probs.tolist())},
    }


# ── Main inputs ───────────────────────────────────────────────────────────────
col_img, col_txt = st.columns([1, 1.3], gap="medium")

with col_img:
    st.markdown(
        "<div class='card' style='min-height:220px'>",
        unsafe_allow_html=True,
    )
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )
    if uploaded_image:
        pil_img = Image.open(uploaded_image).convert("RGB")
        st.image(pil_img, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_txt:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    input_text = st.text_area(
        "Input text",
        height=180,
        placeholder="Paste a tweet, caption, or any text …",
    )
    st.markdown("</div>", unsafe_allow_html=True)

run_btn = st.button("▶  Run Prediction")

# ── Inference ─────────────────────────────────────────────────────────────────
if run_btn:
    if not input_text.strip():
        st.warning("Please enter some text.")
    elif uploaded_image is None:
        st.warning("Please upload an image.")
    elif not os.path.exists(checkpoint_path):
        st.error(f"Checkpoint not found: `{checkpoint_path}`")
    else:
        with st.spinner("Running inference …"):
            try:
                model, device = load_model_cached(checkpoint_path, config_path, gpu_id)
                result = run_predict(input_text, pil_img, model, device, merge_alpha)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        label = result["label"]
        conf  = result["confidence"]
        probs = result["probabilities"]

        # ── Result card ──────────────────────────────────────────────────
        badge_cls = f"badge-{label}"
        color_map = {"positive": "#4ade80", "neutral": "#60a5fa", "negative": "#f87171"}
        fill_cls  = {"positive": "fill-positive", "neutral": "fill-neutral", "negative": "fill-negative"}

        conf_pct = int(conf * 100)

        prob_bars = ""
        for lbl in ["positive", "neutral", "negative"]:
            p   = probs[lbl]
            pct = int(p * 100)
            prob_bars += f"""
            <div class="prob-row">
                <span class="prob-label">{lbl}</span>
                <div class="prob-track">
                    <div class="prob-fill {fill_cls[lbl]}" style="width:{pct}%"></div>
                </div>
                <span class="prob-value">{p:.3f}</span>
            </div>"""

        st.markdown(
            f"""
            <div class="card">
                <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem">
                    <span class="badge {badge_cls}">{label}</span>
                    <span style="color:#777;font-family:'DM Mono',monospace;font-size:0.8rem">
                        confidence&nbsp;<strong style="color:{color_map[label]}">{conf:.4f}</strong>
                    </span>
                </div>
                {prob_bars}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#333;font-size:0.72rem;"
    "font-family:DM Mono,monospace;margin-top:2rem'>"
    "TIGBShareClassifier · BERT + ViT multimodal fusion"
    "</div>",
    unsafe_allow_html=True,
)