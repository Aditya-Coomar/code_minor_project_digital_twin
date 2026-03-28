# ==========================================================
# Koopman Neural State Space Digital Twin — Analysis Dashboard
# Light theme, professional, all diagnostics included
# ==========================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Koopman Digital Twin — TEP Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (light, professional, clinical)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* metric card */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 4px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
}
.metric-card h4 {
    color: #64748b; margin: 0 0 0.25rem 0;
    font-size: 0.68rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.09em;
}
.metric-card .value {
    color: #0f172a; font-size: 1.35rem; font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .sub {
    color: #94a3b8; font-size: 0.72rem; margin-top: 0.15rem;
}

/* section label */
.sec-label {
    font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #94a3b8;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.35rem;
    margin: 1.4rem 0 0.9rem 0;
}

/* param row */
.param-row {
    display: flex; justify-content: space-between;
    padding: 0.35rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: 0.83rem;
}
.pk { color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 0.77rem; }
.pv { color: #0f172a; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

/* sensor card */
.sensor-card {
    background: #fff; border: 1px solid #e2e8f0;
    border-left: 4px solid #94a3b8; border-radius: 3px;
    padding: 0.55rem 0.85rem; margin-bottom: 0.45rem; overflow: hidden;
}
.sn  { font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.88rem; }
.sr2 { float: right; font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: #64748b; }
.r2bar-bg { background: #f1f5f9; border-radius: 2px; height: 5px; margin-top: 4px; overflow: hidden; }
.r2bar    { height: 100%; border-radius: 2px; }

/* figure panel */
.fig-panel {
    border: 1px solid #e2e8f0; border-radius: 3px;
    padding: 0.45rem; margin-bottom: 0.8rem; background: #fff;
}
.fig-cap {
    font-size: 0.68rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
    padding: 0.25rem 0.35rem 0.4rem; border-bottom: 1px solid #f1f5f9; margin-bottom: 0.35rem;
}

/* sidebar title */
.sb-title { font-size: 1.05rem; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
.sb-sub   { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }

/* badge */
.badge {
    display: inline-block; padding: 0.18rem 0.55rem;
    border-radius: 3px; font-size: 0.71rem; font-weight: 600; letter-spacing: 0.03em;
}
.badge-ok  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.badge-warn{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }

/* code block */
.code-block {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;
    padding: 1rem 1.2rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; color: #1e293b; white-space: pre-wrap; overflow-x: auto;
}

/* arch-block */
.arch-block {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
    padding: 1rem 1.2rem; margin-bottom:0.6rem;
}
.arch-block h5 {
    margin: 0 0 0.4rem 0; color: #3b82f6; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.arch-block p { margin: 0; font-size: 0.85rem; color: #475569; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CONSTANTS / PATHS
# ─────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(DIR, "diagnostics")
METRICS = os.path.join(DIAG, "metrics")
SCATTER = os.path.join(DIAG, "prediction_scatter")
TS = os.path.join(DIAG, "prediction_timeseries")
RESHIST = os.path.join(DIAG, "residual_histograms")
RESACORR = os.path.join(DIAG, "residual_autocorr")
ROLLOUT = os.path.join(DIAG, "rollout")
LATENT = os.path.join(DIAG, "latent_analysis")
MODEL_PT = os.path.join(DIR, "koopman_twin_best.pt")

SENSOR_NAMES = [f"xmeas_{i}" for i in range(1, 42)]
CONTROL_NAMES = [f"xmv_{i}" for i in range(1, 12)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL DEFINITIONS  (mirror of models/)
# ─────────────────────────────────────────────
class HistoryEncoder(nn.Module):
    def __init__(self, state_dim, control_dim, latent):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + control_dim, latent)
        self.gru = nn.GRU(
            latent,
            latent // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(latent),
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, latent),
        )

    def forward(self, x, u):
        inp = self.input_proj(torch.cat([x, u], dim=-1))
        out, _ = self.gru(inp)
        return self.out_proj(out[:, -1])


class KoopmanDynamics(nn.Module):
    RESIDUAL_SCALE = 0.3

    def __init__(self, latent, control_dim):
        super().__init__()
        self.A = nn.Linear(latent, latent, bias=False)
        self.B = nn.Linear(control_dim, latent, bias=False)
        self.linear_norm = nn.LayerNorm(latent)
        self.residual = nn.Sequential(
            nn.Linear(latent + control_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, latent),
        )
        self.residual_norm = nn.LayerNorm(latent)
        nn.init.orthogonal_(self.A.weight)

    def forward(self, z, u):
        lin = self.linear_norm(self.A(z) + self.B(u))
        res = self.residual_norm(self.residual(torch.cat([z, u], dim=-1)))
        return lin + self.RESIDUAL_SCALE * res


class ResidualDecoder(nn.Module):
    def __init__(self, latent, state_dim):
        super().__init__()
        self.norm = nn.LayerNorm(latent)
        self.delta_net = nn.Sequential(
            nn.Linear(latent + state_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, state_dim),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(self, z, x_ref):
        z = self.norm(z)
        return x_ref + self.delta_net(torch.cat([z, x_ref], dim=-1))


class KoopmanTwin(nn.Module):
    def __init__(self, state_dim, control_dim, latent=256):
        super().__init__()
        self.encoder = HistoryEncoder(state_dim, control_dim, latent)
        self.dynamics = KoopmanDynamics(latent, control_dim)
        self.decoder = ResidualDecoder(latent, state_dim)

    def rollout(self, x_hist, u_hist, u_future):
        z = self.encoder(x_hist, u_hist)
        x_prev = x_hist[:, -1]
        preds = []
        for t in range(u_future.shape[1]):
            u = u_future[:, t]
            z = self.dynamics(z, u)
            x_pred = self.decoder(z, x_prev)
            preds.append(x_pred)
            x_prev = x_pred
        return torch.stack(preds, dim=1)


# ─────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_overall_metrics():
    path = os.path.join(METRICS, "overall_metrics.txt")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    try:
                        out[k.strip()] = float(v.strip())
                    except:
                        out[k.strip()] = v.strip()
    return out


@st.cache_data
def load_sensor_metrics():
    path = os.path.join(METRICS, "per_sensor_metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


@st.cache_data
def load_dataset_shapes():
    """Return shapes of the saved numpy datasets without loading them fully."""
    shapes = {}
    for fname in ["X_hist.npy", "U_hist.npy", "U_future.npy", "Y_future.npy"]:
        fpath = os.path.join(DIR, fname)
        if os.path.exists(fpath):
            arr = np.load(fpath, mmap_mode="r")
            shapes[fname] = arr.shape
    return shapes


@st.cache_resource
def load_model():
    state_dim = 41
    control_dim = 11
    model = KoopmanTwin(state_dim, control_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    model.eval()
    return model


# ─────────────────────────────────────────────
# SMALL UI HELPERS
# ─────────────────────────────────────────────
def metric_card(title, value, sub=""):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
    <div class="metric-card">
        <h4>{title}</h4>
        <div class="value">{value}</div>
        {sub_html}
    </div>""",
        unsafe_allow_html=True,
    )


def section(label):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)


def param_row(key, val):
    st.markdown(
        f"""
    <div class="param-row">
        <span class="pk">{key}</span>
        <span class="pv">{val}</span>
    </div>""",
        unsafe_allow_html=True,
    )


def fig_panel(title, img_path, use_container_width=True):
    st.markdown(
        f'<div class="fig-panel"><div class="fig-cap">{title}</div></div>',
        unsafe_allow_html=True,
    )
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=use_container_width)
    else:
        st.caption(f"Image not found: {img_path}")


def r2_color(r2):
    if r2 >= 0.9:
        return "#16a34a"
    if r2 >= 0.7:
        return "#ca8a04"
    return "#dc2626"


def sensor_card(name, r2):
    clr = r2_color(r2)
    bar = max(0.0, r2) * 100
    st.markdown(
        f"""
    <div class="sensor-card" style="border-left-color:{clr};">
        <span class="sn" style="color:{clr};">{name}</span>
        <span class="sr2">R² = {r2:.4f}</span>
        <div class="r2bar-bg"><div class="r2bar" style="width:{bar:.1f}%;background:{clr};"></div></div>
    </div>""",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div class="sb-title">Koopman Digital Twin</div>'
        '<div class="sb-sub">TEP Process Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    section("Overall Performance")
    om = load_overall_metrics()
    if om:
        param_row(
            "Overall R²",
            f"{om.get('Overall R2', 'N/A'):.4f}"
            if isinstance(om.get("Overall R2"), float)
            else om.get("Overall R2", "N/A"),
        )
        param_row(
            "Overall RMSE",
            f"{om.get('Overall RMSE', 'N/A'):.4f}"
            if isinstance(om.get("Overall RMSE"), float)
            else om.get("Overall RMSE", "N/A"),
        )

    section("Architecture")
    param_row("Model", "KoopmanTwin")
    param_row("Encoder", "BiGRU (2-layer)")
    param_row("Latent dim", "256")
    param_row("State dim", "41  (xmeas_1–41)")
    param_row("Control dim", "11  (xmv_1–11)")
    param_row("Dynamics", "Linear A + Res.")
    param_row("Decoder", "Residual δ-net")

    section("Training Config")
    param_row("BATCH_SIZE", "64")
    param_row("EPOCHS", "200")
    param_row("LR", "1e-3")
    param_row("MAX_HORIZON", "20")
    param_row("WARMUP_EPOCHS", "20")
    param_row("RECON_WEIGHT", "0.5")
    param_row("ROLLOUT_WEIGHT", "2.0")
    param_row("LATENT_WEIGHT", "0.3")
    param_row("REG_WEIGHT", "1e-4")
    param_row("Optimizer", "AdamW")
    param_row("Scheduler", "OneCycleLR")
    param_row("Grad clip", "1.0")

    section("Dataset")
    param_row("Source", "TEP FaultFree Training")
    param_row("HISTORY", "30 steps")
    param_row("HORIZON", "20 steps")
    param_row("Smooth W", "5  (uniform_filter1d)")
    param_row("Scaling", "StandardScaler (X & U)")

    section("System")
    dev_str = (
        f"CUDA — {torch.cuda.get_device_name(0)}" if DEVICE.type == "cuda" else "CPU"
    )
    param_row("DEVICE", dev_str)
    model_exists = os.path.exists(MODEL_PT)
    badge_cls = "badge-ok" if model_exists else "badge-warn"
    badge_txt = "MODEL LOADED" if model_exists else "MODEL MISSING"
    st.markdown(
        f'<span class="badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True
    )

# ═══════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("## Koopman Neural State Space Digital Twin &mdash; Results Dashboard")
st.caption(
    "Tennessee Eastman Process (TEP) · Fault-Free Training Data · 41 measured variables · 11 manipulated variables"
)

# ═══════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════
(
    tab_overview,
    tab_dataset,
    tab_arch,
    tab_scatter,
    tab_ts,
    tab_res,
    tab_rollout,
    tab_latent,
    tab_predict,
) = st.tabs(
    [
        "📊 Overview",
        "🗄️ Dataset",
        "🧠 Architecture",
        "🔵 Scatter Plots",
        "📈 Time Series",
        "📉 Residuals",
        "🔄 Rollout",
        "🌐 Latent Space",
        "⚡ Sample Predictions",
    ]
)

# ══════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════
with tab_overview:
    om = load_overall_metrics()
    df = load_sensor_metrics()

    section("Global Metrics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Overall R²", f"{om.get('Overall R2', 0):.4f}", "Cross all 41 sensors"
        )
    with c2:
        metric_card("Overall RMSE", f"{om.get('Overall RMSE', 0):.4f}", "Scaled units")
    with c3:
        avg_r2 = df["r2"].mean() if not df.empty else 0
        metric_card("Mean Sensor R²", f"{avg_r2:.4f}", "Arithmetic mean")
    with c4:
        n_good = int((df["r2"] >= 0.9).sum()) if not df.empty else 0
        metric_card("R² ≥ 0.9", f"{n_good} / {len(df)}", "High-fidelity sensors")

    section("Per-Sensor R² Bar Chart")
    fig_panel("Sensor R² Ranking", os.path.join(METRICS, "sensor_r2_bar.png"))

    section("Sensor Metrics Table")
    if not df.empty:
        df_sorted = df.sort_values("r2", ascending=False).reset_index(drop=True)
        st.dataframe(
            df_sorted.style.format({"r2": "{:.4f}", "rmse": "{:.4f}"})
            .background_gradient(subset=["r2"], cmap="RdYlGn", vmin=0, vmax=1)
            .background_gradient(subset=["rmse"], cmap="RdYlGn_r", vmin=0, vmax=1),
            use_container_width=True,
            height=450,
        )
    else:
        st.info("per_sensor_metrics.csv not found.")

# ══════════════════════════════════════
# TAB 2 — DATASET
# ══════════════════════════════════════
with tab_dataset:
    section("Dataset Overview")
    shapes = load_dataset_shapes()
    fa = "X_hist.npy"
    if fa in shapes:
        n, h, s = shapes[fa]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Samples", f"{n:,}")
        with c2:
            metric_card("History Window", f"{h} steps")
        with c3:
            metric_card("State Dim (xmeas)", f"{s}")
        ctrl_dim = shapes["U_hist.npy"][-1] if "U_hist.npy" in shapes else 11
        with c4:
            metric_card("Control Dim (xmv)", f"{ctrl_dim}")
    else:
        st.info("Dataset .npy files not found in this folder.")

    section("Array Shapes")
    rows = []
    for fname, desc in [
        ("X_hist.npy", "State history window   [N, HISTORY=30, 41]"),
        ("U_hist.npy", "Control history window [N, HISTORY=30, 11]"),
        ("U_future.npy", "Future controls        [N, HORIZON=20, 11]"),
        ("Y_future.npy", "Future states (target) [N, HORIZON=20, 41]"),
    ]:
        shp = str(shapes.get(fname, "N/A"))
        rows.append({"File": fname, "Description": desc, "Shape": shp})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=180)

    section("State Variables (xmeas_1 – xmeas_41)")
    cols2 = st.columns(3)
    for idx, sn in enumerate(SENSOR_NAMES):
        with cols2[idx % 3]:
            st.markdown(f"<code>{sn}</code>", unsafe_allow_html=True)

    section("Control / Manipulated Variables (xmv_1 – xmv_11)")
    cols3 = st.columns(4)
    for idx, cn in enumerate(CONTROL_NAMES):
        with cols3[idx % 4]:
            st.markdown(f"<code>{cn}</code>", unsafe_allow_html=True)

    section("Preprocessing Pipeline")
    st.markdown(
        """
    <div class="code-block">1. Load  TEP_FaultFree_Training.RData
2. Detect simulation-run column (simulationRun)
3. Extract  xmeas_* → X_raw  (41 cols)   and   xmv_* → U_raw  (11 cols)
4. Smooth X_raw per-run with uniform_filter1d(window=5)  → removes sensor noise
5. StandardScaler.fit_transform(X_smooth) → X   (standardized states)
6. StandardScaler.fit_transform(U_raw)    → U   (standardized controls)
7. Build sliding windows per simulation run:
       X_hist   [ i-HISTORY : i ]    (30 × 41)
       U_hist   [ i-HISTORY : i ]    (30 × 11)
       U_future [ i          : i+HORIZON ] (20 × 11)
       Y_future [ i          : i+HORIZON ] (20 × 41)
8. Save as float32 .npy arrays</div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ══════════════════════════════════════
with tab_arch:
    section("Model Overview")
    st.markdown("""
    > **KoopmanTwin** lifts the nonlinear TEP process into a latent space where dynamics
    > are *approximately linear* (Koopman operator), then decodes predictions of the
    > next state via a *residual delta decoder*.
    """)

    st.markdown(
        """
    <div class="arch-block">
        <h5>1 · HistoryEncoder</h5>
        <p>Bidirectional GRU (2 layers, hidden = latent//2 each dir → concatenated = latent = 256).
        Input projected from (state_dim + control_dim = 52) → latent.
        Final hidden state passed through LayerNorm → Linear → GELU → Linear to produce latent vector <b>z</b>.</p>
    </div>
    <div class="arch-block">
        <h5>2 · KoopmanDynamics</h5>
        <p>Linear part: <b>z_next = LayerNorm(A·z + B·u)</b> where A ∈ ℝ<sup>256×256</sup> (orthogonal init) and B ∈ ℝ<sup>256×11</sup>.
        Nonlinear residual: 3-layer MLP (512 → 256 → 256) scaled by 0.3, added to the linear part.
        Also regularised by Frobenius norm of A during training.</p>
    </div>
    <div class="arch-block">
        <h5>3 · ResidualDecoder</h5>
        <p>Predicts state <em>change</em> (delta): <b>x_pred = x_ref + δ(z, x_ref)</b>.
        δ-network: LayerNorm(z) → 3-layer MLP (512 → 256 → state_dim).
        Last layer zero-initialized so warm-start = persistence baseline.</p>
    </div>
    <div class="arch-block">
        <h5>4 · Autoregressive Rollout</h5>
        <p>At each horizon step t: z ← Dynamics(z, u_t) then x_pred ← Decoder(z, x_prev).
        x_prev is updated to x_pred (fully autoregressive).
        Curriculum warmup: horizon fixed at 1 for first 20 epochs, then ramps by 1 every 6 epochs up to MAX_HORIZON=20.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    section("Loss Function")
    st.markdown(
        """
    <div class="code-block">Total = RECON_WEIGHT  × recon_loss       (0.50 × MSE of x̂_{-1})
       + ROLLOUT_WEIGHT × rollout_loss     (2.00 × autoregressive multi-step MSE)
       + LATENT_WEIGHT  × latent_loss      (0.30 × latent-consistency over 3 steps)
       + REG_WEIGHT     × ‖A‖_F           (1e-4 × Frobenius norm of Koopman A)

Optimizer  : AdamW  (lr=1e-3, weight_decay=1e-5)
Scheduler  : OneCycleLR (max_lr=1e-3, pct_start=0.05, anneal='cos')
Grad clip  : max_norm = 1.0
Saved when : R² on last rollout step improves</div>
    """,
        unsafe_allow_html=True,
    )

    section("Parameter Count")
    if os.path.exists(MODEL_PT):
        with st.spinner("Counting parameters…"):
            try:
                m = KoopmanTwin(41, 11)
                total = sum(p.numel() for p in m.parameters())
                trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Total Parameters", f"{total:,}")
                with c2:
                    metric_card("Trainable Parameters", f"{trainable:,}")
            except Exception as e:
                st.warning(f"Could not count: {e}")
    else:
        st.info("Model file not found.")

# ══════════════════════════════════════
# TAB 4 — SCATTER PLOTS (41 sensors)
# ══════════════════════════════════════
with tab_scatter:
    df_s = load_sensor_metrics()
    section("Predicted vs True — Scatter Plots (4000 samples per sensor)")

    sensor_filter = st.selectbox(
        "Filter sensors",
        ["All sensors", "R² ≥ 0.90", "R² ≥ 0.70", "R² < 0.70"],
        key="scatter_filter",
    )

    sensors_to_show = SENSOR_NAMES
    if not df_s.empty and sensor_filter != "All sensors":
        r2_map = dict(zip(df_s["sensor"], df_s["r2"]))
        if sensor_filter == "R² ≥ 0.90":
            sensors_to_show = [s for s in SENSOR_NAMES if r2_map.get(s, 0) >= 0.90]
        elif sensor_filter == "R² ≥ 0.70":
            sensors_to_show = [s for s in SENSOR_NAMES if r2_map.get(s, 0) >= 0.70]
        else:
            sensors_to_show = [s for s in SENSOR_NAMES if r2_map.get(s, 0) < 0.70]

    st.caption(f"Showing {len(sensors_to_show)} sensors")

    r2_map = dict(zip(df_s["sensor"], df_s["r2"])) if not df_s.empty else {}
    for i in range(0, len(sensors_to_show), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(sensors_to_show):
                sn = sensors_to_show[i + j]
                img = os.path.join(SCATTER, f"{sn}_scatter.png")
                r2v = r2_map.get(sn, float("nan"))
                with col:
                    sensor_card(sn, r2v)
                    if os.path.exists(img):
                        st.image(img, use_container_width=True)

# ══════════════════════════════════════
# TAB 5 — TIME SERIES (41 sensors)
# ══════════════════════════════════════
with tab_ts:
    df_s = load_sensor_metrics()
    section("Predicted vs True — Time Series (first 500 steps)")

    sensor_sel = st.multiselect(
        "Select sensors (leave empty = show all)",
        SENSOR_NAMES,
        default=[],
        key="ts_select",
    )
    sensors_show = sensor_sel if sensor_sel else SENSOR_NAMES

    r2_map = dict(zip(df_s["sensor"], df_s["r2"])) if not df_s.empty else {}
    for i in range(0, len(sensors_show), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(sensors_show):
                sn = sensors_show[i + j]
                img = os.path.join(TS, f"{sn}_timeseries.png")
                r2v = r2_map.get(sn, float("nan"))
                with col:
                    sensor_card(sn, r2v)
                    if os.path.exists(img):
                        st.image(img, use_container_width=True)

# ══════════════════════════════════════
# TAB 6 — RESIDUALS
# ══════════════════════════════════════
with tab_res:
    df_s = load_sensor_metrics()
    section("Residual Diagnostics — Histograms & Autocorrelation")

    res_sensor = st.selectbox("Select sensor", SENSOR_NAMES, key="res_sensor")
    r2_map = dict(zip(df_s["sensor"], df_s["r2"])) if not df_s.empty else {}

    c1, c2 = st.columns(2)
    with c1:
        section("Residual Histogram")
        hist_img = os.path.join(RESHIST, f"{res_sensor}_residual_hist.png")
        if os.path.exists(hist_img):
            st.image(hist_img, use_container_width=True)
        else:
            st.info("Not found.")

    with c2:
        section("Residual Autocorrelation (lags 1–49)")
        acorr_img = os.path.join(RESACORR, f"{res_sensor}_autocorr.png")
        if os.path.exists(acorr_img):
            st.image(acorr_img, use_container_width=True)
        else:
            st.info("Not found.")

    section("All Residual Histograms — Gallery")
    st.caption("Scroll through all 41 sensor residual histograms")
    for i in range(0, len(SENSOR_NAMES), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(SENSOR_NAMES):
                sn = SENSOR_NAMES[i + j]
                img = os.path.join(RESHIST, f"{sn}_residual_hist.png")
                with col:
                    st.caption(sn)
                    if os.path.exists(img):
                        st.image(img, use_container_width=True)

    section("All Residual Autocorrelations — Gallery")
    for i in range(0, len(SENSOR_NAMES), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if i + j < len(SENSOR_NAMES):
                sn = SENSOR_NAMES[i + j]
                img = os.path.join(RESACORR, f"{sn}_autocorr.png")
                with col:
                    st.caption(sn)
                    if os.path.exists(img):
                        st.image(img, use_container_width=True)

# ══════════════════════════════════════
# TAB 7 — ROLLOUT
# ══════════════════════════════════════
with tab_rollout:
    section("Multi-Step Rollout Stability")
    st.markdown("""
    > This curve shows how R² degrades as the rollout horizon increases from 2 to 20 steps.
    > A flat or slowly-decaying curve indicates stable long-horizon predictions.
    """)
    roll_img = os.path.join(ROLLOUT, "rollout_r2_curve.png")
    if os.path.exists(roll_img):
        st.image(roll_img, use_container_width=True)
    else:
        st.info("rollout_r2_curve.png not found.")

    section("Rollout Curriculum Schedule")
    st.markdown(
        """
    <div class="code-block">Epoch  1–20 : horizon = 1   (warmup — single-step only)
Epoch 21–26 : horizon = 2
Epoch 27–32 : horizon = 3
...  (+1 every 6 epochs)
Epoch 75+   : horizon = 20  (maximum)

Training objective: ROLLOUT_WEIGHT × MSE(preds[:horizon], y_future[:horizon])</div>
    """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════
# TAB 8 — LATENT SPACE
# ══════════════════════════════════════
with tab_latent:
    section("Latent Space PCA (2 components)")
    st.markdown("""
    > Each point is one data sample's latent vector **z** (dim 256) compressed to 2D via PCA.
    > Compact, well-separated clusters indicate an expressive and structured latent representation.
    """)
    lat_img = os.path.join(LATENT, "latent_pca.png")
    if os.path.exists(lat_img):
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(lat_img, use_container_width=True)
        with col_info:
            section("Encoder Details")
            param_row("Input dim", "state_dim + control_dim = 52")
            param_row("Proj dim", "256 (latent)")
            param_row("GRU layers", "2 bidirectional")
            param_row("GRU hidden", "128 per direction")
            param_row("Out dim", "256")
            param_row("OutputProj", "LN → Lin → GELU → Lin")
    else:
        st.info("latent_pca.png not found.")

# ══════════════════════════════════════
# TAB 9 — SAMPLE PREDICTIONS
# ══════════════════════════════════════
with tab_predict:
    section("Interactive Sample Predictions")
    st.markdown("""
    Load the trained Koopman model and run a live rollout prediction on a chosen dataset sample.
    You can select the sample index and rollout horizon, then compare predicted vs target trajectories.
    """)

    # Check for numpy files
    X_path = os.path.join(DIR, "X_hist.npy")
    U_path = os.path.join(DIR, "U_hist.npy")
    Uf_path = os.path.join(DIR, "U_future.npy")
    Y_path = os.path.join(DIR, "Y_future.npy")

    files_ok = all(
        os.path.exists(p) for p in [X_path, U_path, Uf_path, Y_path, MODEL_PT]
    )

    if not files_ok:
        missing = [
            p
            for p in [X_path, U_path, Uf_path, Y_path, MODEL_PT]
            if not os.path.exists(p)
        ]
        st.error(f"Missing required files:\n" + "\n".join(missing))
    else:
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                sample_idx = st.number_input(
                    "Sample index", min_value=0, max_value=225000, value=140, step=1
                )
            with c2:
                horizon = st.slider(
                    "Rollout horizon (steps)", min_value=1, max_value=20, value=20
                )
            with c3:
                sensor_choice = st.selectbox("Sensor to plot", SENSOR_NAMES, index=17)

            run_btn = st.form_submit_button(
                "▶ Run Prediction", type="primary", use_container_width=True
            )

        if run_btn:
            with st.spinner("Loading dataset and running rollout…"):
                try:
                    # Load slices (memory-mapped)
                    X_all = np.load(X_path, mmap_mode="r")
                    U_all = np.load(U_path, mmap_mode="r")
                    Uf_all = np.load(Uf_path, mmap_mode="r")
                    Y_all = np.load(Y_path, mmap_mode="r")

                    n_samples = X_all.shape[0]
                    idx = int(sample_idx) % n_samples

                    x_hist = torch.tensor(
                        X_all[idx : idx + 1].copy(), dtype=torch.float32
                    ).to(DEVICE)
                    u_hist = torch.tensor(
                        U_all[idx : idx + 1].copy(), dtype=torch.float32
                    ).to(DEVICE)
                    u_fut = torch.tensor(
                        Uf_all[idx : idx + 1, :horizon].copy(), dtype=torch.float32
                    ).to(DEVICE)
                    y_fut = Y_all[idx, :horizon]  # (horizon, 41)

                    model = load_model()
                    with torch.no_grad():
                        preds = model.rollout(x_hist, u_hist, u_fut)  # (1, horizon, 41)
                    preds_np = preds.cpu().numpy()[0]  # (horizon, 41)

                    sensor_idx = SENSOR_NAMES.index(sensor_choice)
                    pred_trace = preds_np[:, sensor_idx]
                    true_trace = y_fut[:, sensor_idx]
                    from sklearn.metrics import r2_score as _r2

                    r2v = _r2(true_trace, pred_trace)

                    # ── Metric cards ──
                    section("Prediction Results")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    rmse_v = float(np.sqrt(np.mean((pred_trace - true_trace) ** 2)))
                    mae_v = float(np.mean(np.abs(pred_trace - true_trace)))
                    with mc1:
                        metric_card("Sample Index", f"{idx:,}")
                    with mc2:
                        metric_card("Horizon", f"{horizon} steps")
                    with mc3:
                        metric_card(f"R² ({sensor_choice})", f"{r2v:.4f}")
                    with mc4:
                        metric_card("RMSE (scaled)", f"{rmse_v:.4f}")

                    # ── Plot ──
                    section(f"Predicted vs Target — {sensor_choice}")
                    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

                    tsteps = np.arange(horizon)

                    # Time series
                    axes[0].plot(
                        tsteps, true_trace, lw=2, label="Target", color="#1e40af"
                    )
                    axes[0].plot(
                        tsteps,
                        pred_trace,
                        lw=2,
                        label="Predicted",
                        color="#dc2626",
                        linestyle="--",
                    )
                    axes[0].fill_between(
                        tsteps, true_trace, pred_trace, alpha=0.12, color="#dc2626"
                    )
                    axes[0].set_xlabel("Rollout Step")
                    axes[0].set_ylabel("Scaled Value")
                    axes[0].set_title(f"{sensor_choice} — Rollout  (R²={r2v:.4f})")
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)

                    # Scatter
                    vmin = min(true_trace.min(), pred_trace.min())
                    vmax = max(true_trace.max(), pred_trace.max())
                    axes[1].scatter(
                        true_trace,
                        pred_trace,
                        s=60,
                        alpha=0.7,
                        color="#3b82f6",
                        edgecolors="white",
                        lw=0.5,
                    )
                    axes[1].plot([vmin, vmax], [vmin, vmax], "k--", lw=1, label="y=x")
                    axes[1].set_xlabel("True")
                    axes[1].set_ylabel("Predicted")
                    axes[1].set_title(f"{sensor_choice} — Scatter  (horizon={horizon})")
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    # ── All sensors table at this sample ──
                    section("Per-Sensor R² at This Sample")
                    sample_r2 = []
                    sample_rmse = []
                    for si in range(41):
                        r2i = float(_r2(y_fut[:, si], preds_np[:, si]))
                        rmi = float(
                            np.sqrt(np.mean((preds_np[:, si] - y_fut[:, si]) ** 2))
                        )
                        sample_r2.append(r2i)
                        sample_rmse.append(rmi)
                    res_df = pd.DataFrame(
                        {"Sensor": SENSOR_NAMES, "R²": sample_r2, "RMSE": sample_rmse}
                    )
                    res_df = res_df.sort_values("R²", ascending=False).reset_index(
                        drop=True
                    )
                    st.dataframe(
                        res_df.style.format(
                            {"R²": "{:.4f}", "RMSE": "{:.4f}"}
                        ).background_gradient(
                            subset=["R²"], cmap="RdYlGn", vmin=0, vmax=1
                        ),
                        use_container_width=True,
                        height=400,
                    )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    import traceback

                    st.code(traceback.format_exc())

        st.markdown("---")
        section("How to Run a Batch Prediction (Code Reference)")
        st.code(
            """
import numpy as np
import torch
import sys, os
sys.path.insert(0, r"d:/Minor Project/code/state_space_model")

from models.koopman_twin import KoopmanTwin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PT = "koopman_twin_best.pt"

model = KoopmanTwin(state_dim=41, control_dim=11)
model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
model.eval()
model.to(DEVICE)

# Load windows
X_hist  = np.load("X_hist.npy",  mmap_mode="r")   # (N, 30, 41)
U_hist  = np.load("U_hist.npy",  mmap_mode="r")   # (N, 30, 11)
U_future= np.load("U_future.npy",mmap_mode="r")   # (N, 20, 11)
Y_future= np.load("Y_future.npy",mmap_mode="r")   # (N, 20, 41)

# Pick a sample
idx     = 0
horizon = 10

x_hist  = torch.tensor(X_hist[idx:idx+1].copy(),       dtype=torch.float32).to(DEVICE)
u_hist  = torch.tensor(U_hist[idx:idx+1].copy(),       dtype=torch.float32).to(DEVICE)
u_future= torch.tensor(U_future[idx:idx+1, :horizon].copy(), dtype=torch.float32).to(DEVICE)

with torch.no_grad():
    preds = model.rollout(x_hist, u_hist, u_future)   # (1, horizon, 41)

preds_np = preds.cpu().numpy()[0]   # (horizon, 41)
target   = Y_future[idx, :horizon]  # (horizon, 41)

from sklearn.metrics import r2_score
print("R²:", r2_score(target, preds_np))
        """,
            language="python",
        )
