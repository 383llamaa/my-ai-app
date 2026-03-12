"""
XOR Neural Network — Streamlit Web App
=======================================
Mobile-first UI, works on iPhone Safari.
Weights are trained once at startup via st.cache_resource
and reused for every prediction — no retraining on each click.

Deploy free: https://streamlit.io/cloud
"""

import streamlit as st
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XOR Neural Net",
    page_icon="🧠",
    layout="centered",          # clean single-column on mobile
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — dark theme, large touch targets, mobile-friendly typography
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0a0e17;
    color: #c8d8e8;
  }

  /* ── Main container ── */
  .block-container {
    max-width: 520px;
    padding: 2rem 1.2rem 4rem;
    margin: auto;
  }

  /* ── Section cards ── */
  .card {
    background: #0d1220;
    border: 1px solid #1a2840;
    border-radius: 12px;
    padding: 1.4rem 1.4rem 1rem;
    margin-bottom: 1.2rem;
  }

  /* ── Section heading ── */
  .section-label {
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    color: #3a5070;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
  }

  /* ── Result number ── */
  .result-box {
    background: #0d1220;
    border: 1px solid #1a2840;
    border-radius: 12px;
    padding: 1.6rem;
    text-align: center;
    margin-bottom: 1.2rem;
  }
  .result-digit {
    font-size: 5rem;
    font-weight: 700;
    line-height: 1;
  }
  .result-cyan  { color: #22d3ee; }
  .result-red   { color: #ef4444; }
  .result-dim   { color: #2a3a50; }

  /* ── Confidence bar track ── */
  .conf-track {
    height: 6px;
    background: #141e2c;
    border-radius: 99px;
    margin: 0.6rem 0 0.3rem;
    overflow: hidden;
  }
  .conf-fill-cyan { background: #22d3ee; height: 100%; border-radius: 99px; }
  .conf-fill-red  { background: #ef4444; height: 100%; border-radius: 99px; }

  /* ── Truth table rows ── */
  .tt-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.45rem 0.6rem;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 0.82rem;
  }
  .tt-correct { background: #0a2010; border: 1px solid #1a4020; }
  .tt-wrong   { background: #200a0a; border: 1px solid #4a1010; }
  .tt-cyan    { color: #22d3ee; }
  .tt-red     { color: #ef4444; }
  .tt-dim     { color: #3a5070; }

  /* ── Streamlit widget overrides ── */
  div[data-testid="stNumberInput"] input {
    background: #141e2c !important;
    color: #e8f0f8 !important;
    border: 1px solid #1a2840 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.1rem !important;
    padding: 0.6rem !important;
    text-align: center !important;
  }
  div[data-testid="stNumberInput"] input:focus {
    border-color: #22d3ee !important;
    box-shadow: 0 0 0 2px #22d3ee22 !important;
  }

  div[data-testid="stButton"] button {
    width: 100%;
    background: #22d3ee !important;
    color: #0a0e17 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    margin-top: 0.4rem !important;
    transition: background 0.15s !important;
  }
  div[data-testid="stButton"] button:hover {
    background: #0891b2 !important;
  }

  div[data-testid="stSuccess"] {
    background: #0a2010 !important;
    border: 1px solid #1a5030 !important;
    border-radius: 10px !important;
    color: #22d3ee !important;
    font-family: 'JetBrains Mono', monospace !important;
  }

  /* hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK  (self-contained — no extra imports needed)
# ──────────────────────────────────────────────────────────────────────────────

class _Sigmoid:
    def forward(self, z):
        self._s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return self._s
    def derivative(self):
        return self._s * (1.0 - self._s)

class _DenseLayer:
    def __init__(self, n_in, n_out):
        scale  = np.sqrt(1.0 / n_in)
        self.W = np.random.randn(n_out, n_in) * scale
        self.b = np.zeros(n_out)
        self.act = _Sigmoid()
        self._x = None

    def forward(self, x):
        self._x = x
        z = self.W @ x + self.b
        return self.act.forward(z)

    def backward(self, d_up):
        d = d_up * self.act.derivative()
        self.dW = np.outer(d, self._x)
        self.db = d
        return self.W.T @ d

    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class XORNetwork:
    """2 → 4 → 1 network trained on XOR."""

    def __init__(self):
        self.layers = [_DenseLayer(2, 4), _DenseLayer(4, 1)]

    def predict_raw(self, x1: float, x2: float) -> float:
        """Forward pass → sigmoid output in (0, 1)."""
        x = np.array([x1, x2], dtype=float)
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return float(out[0])

    def predict_class(self, x1: float, x2: float) -> int:
        return round(self.predict_raw(x1, x2))

    def _train(self, epochs=6000, lr=0.1):
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        for _ in range(epochs):
            idx = np.random.permutation(4)
            for i in idx:
                # forward
                out = np.array([x for x in [X[i]]], dtype=float)[0]
                a = out
                for layer in self.layers:
                    a = layer.forward(a)
                pred = float(a[0])
                # backward  (MSE gradient)
                grad = np.array([2.0 * (pred - y[i])])
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                for layer in self.layers:
                    layer.update(lr)

    def all_predictions(self):
        """Return predictions for all 4 XOR inputs."""
        cases = [(0,0,0),(0,1,1),(1,0,1),(1,1,0)]
        return [
            {
                "x1": x1, "x2": x2, "target": t,
                "raw": self.predict_raw(x1, x2),
                "cls": self.predict_class(x1, x2),
            }
            for x1, x2, t in cases
        ]


# ──────────────────────────────────────────────────────────────────────────────
# CACHED TRAINING  — runs exactly once per server session, never on each click
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training neural network…")
def load_model() -> XORNetwork:
    """
    @st.cache_resource caches the returned object for the lifetime of the
    Streamlit server process. On Streamlit Cloud this means the network is
    trained once when the first visitor loads the page, and every subsequent
    prediction (by any visitor) reuses the same in-memory weights.
    """
    np.random.seed(42)
    net = XORNetwork()
    net._train(epochs=6000, lr=0.1)
    return net


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────

net = load_model()      # ← cached; instant after first load

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 1.6rem 0 1rem;">
  <div style="font-size:2.4rem; margin-bottom:0.3rem;">🧠</div>
  <div style="font-size:1.3rem; font-weight:700; color:#f0f6ff; letter-spacing:-0.5px;">
    Neural Network Predictor
  </div>
  <div style="font-size:0.7rem; color:#3a5070; letter-spacing:0.1em; margin-top:4px;">
    XOR CLASSIFIER · 2 → 4 → 1 · SIGMOID · MSE
  </div>
</div>
""", unsafe_allow_html=True)

# ── Input card ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="section-label">Inputs</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    x1 = st.number_input(
        "Input 1",
        min_value=0.0, max_value=1.0,
        value=0.0, step=1.0,
        format="%.1f",
        help="Try 0 or 1 for exact XOR; decimals interpolate",
    )
with col2:
    x2 = st.number_input(
        "Input 2",
        min_value=0.0, max_value=1.0,
        value=0.0, step=1.0,
        format="%.1f",
        help="Try 0 or 1 for exact XOR; decimals interpolate",
    )

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("PREDICT  →", use_container_width=True)

# ── Result ────────────────────────────────────────────────────────────────────
raw  = net.predict_raw(x1, x2)
cls  = round(raw)
conf = raw if cls == 1 else 1.0 - raw       # confidence toward predicted class
pct  = conf * 100
bar_pct = int(conf * 100)
color_cls  = "result-cyan" if cls == 1 else "result-red"
color_name = "cyan" if cls == 1 else "red"

if predict_clicked or True:                  # always show live result
    st.markdown(f"""
    <div class="result-box">
      <div class="section-label">Output Class</div>
      <div class="result-digit {color_cls}">{cls}</div>
      <div class="conf-track">
        <div class="conf-fill-{color_name}" style="width:{bar_pct}%"></div>
      </div>
      <div style="font-size:0.75rem; color:#3a5070;">
        confidence &nbsp;<span style="color:#607090">{pct:.1f}%</span>
        &nbsp;·&nbsp; raw sigmoid &nbsp;<span style="color:#607090">{raw:.4f}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # st.success — as required
    label = "XOR output is  1  ✓" if cls == 1 else "XOR output is  0  ✓"
    st.success(f"**Prediction: {cls}** — {label}  (confidence {pct:.1f}%)")

# ── Truth table ───────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="section-label">Full XOR Truth Table</div>', unsafe_allow_html=True)

for row in net.all_predictions():
    ok      = row["cls"] == row["target"]
    css_row = "tt-correct" if ok else "tt-wrong"
    css_val = "tt-cyan" if ok else "tt-red"
    icon    = "✓" if ok else "✗"
    st.markdown(f"""
    <div class="tt-row {css_row}">
      <span class="tt-dim">[{row['x1']:.0f}, {row['x2']:.0f}]</span>
      <span class="tt-dim">target: {row['target']}</span>
      <span class="tt-dim">→</span>
      <span class="{css_val}">{row['raw']:.3f} &nbsp; (class {row['cls']})</span>
      <span class="{css_val}">{icon}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Architecture info ─────────────────────────────────────────────────────────
with st.expander("⬡  Model architecture & weights"):
    st.markdown("**Network:** 2 inputs → 4 hidden (sigmoid) → 1 output (sigmoid)")
    st.markdown("**Loss:** Mean Squared Error &nbsp;·&nbsp; **Optimiser:** SGD &nbsp;·&nbsp; **Epochs:** 6 000")
    st.markdown("**Layer 1 weights** (4 × 2):")
    st.dataframe(
        {f"w→h{i+1}": net.layers[0].W[i] for i in range(4)},
        use_container_width=True,
    )
    st.markdown("**Layer 2 weights** (1 × 4):")
    st.dataframe(
        {"w→out": net.layers[1].W[0]},
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 0.5rem;
            font-size:0.65rem; color:#1e2d40; letter-spacing:0.08em;">
  BUILT WITH NUMPY · DEPLOYED ON STREAMLIT CLOUD · NO PYTORCH / TENSORFLOW
</div>
""", unsafe_allow_html=True)
