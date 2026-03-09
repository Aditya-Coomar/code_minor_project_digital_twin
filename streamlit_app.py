# ==========================================================
# Streamlit Model Analysis Dashboard
# Interactive version of analyze_model.py
# ==========================================================

import os
import glob
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="TEP Process Analysis Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# CUSTOM STYLING
# ==========================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Metric cards — clinical style with left accent */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #475569;
        border-radius: 2px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }
    .metric-card h4 {
        color: #64748b;
        margin: 0 0 0.2rem 0;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'Inter', sans-serif;
    }
    .metric-card .value {
        color: #1e293b;
        font-size: 1.25rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section headers — research paper style */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem 0;
        font-family: 'Inter', sans-serif;
    }

    /* Config parameter row */
    .config-param {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.85rem;
    }
    .config-param .param-key {
        color: #64748b;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
    }
    .config-param .param-val {
        color: #1e293b;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
    }

    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.03em;
    }
    .status-available {
        background: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    .status-missing {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }

    /* R2 bar */
    .r2-bar-container {
        background: #f1f5f9;
        border-radius: 1px;
        overflow: hidden;
        height: 6px;
        margin-top: 4px;
    }
    .r2-bar {
        height: 100%;
        border-radius: 1px;
    }

    /* Sensor card — lab report style */
    .sensor-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #94a3b8;
        border-radius: 2px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.4rem;
    }
    .sensor-card .sensor-name {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.88rem;
    }
    .sensor-card .sensor-r2 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #64748b;
        float: right;
    }

    /* Figure containers */
    .figure-panel {
        border: 1px solid #e2e8f0;
        border-radius: 2px;
        padding: 0.5rem;
        margin-bottom: 0.8rem;
        background: #ffffff;
    }
    .figure-panel .fig-caption {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        padding: 0.3rem 0.4rem 0.4rem;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 0.4rem;
        font-family: 'Inter', sans-serif;
    }
    .figure-panel img {
        height: 280px;
        object-fit: contain;
        width: 100%;
    }

    /* Sensor figure */
    .sensor-figure img {
        height: 240px;
        object-fit: contain;
        width: 100%;
    }

    /* Sidebar title */
    .sidebar-title {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }
    .sidebar-subtitle {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# CONSTANTS
# ==========================================================

DATA_PATH = "data/TEP_DATASET"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def parse_config(run_folder: str) -> dict:
    """Parse config.txt from a run folder into a dictionary."""
    config = {}
    config_path = os.path.join(run_folder, "config.txt")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    try:
                        config[key.strip()] = int(val.strip())
                    except ValueError:
                        try:
                            config[key.strip()] = float(val.strip())
                        except ValueError:
                            config[key.strip()] = val.strip()
    return config


def get_run_folders() -> list:
    """Get all run_* folders inside results/."""
    pattern = os.path.join(RESULTS_DIR, "run_*")
    folders = sorted(glob.glob(pattern))
    return folders


def has_persensor_results(run_folder: str) -> bool:
    """Check if per-sensor analysis results already exist."""
    csv_path = os.path.join(run_folder, "sensor_r2_values.csv")
    plots_dir = os.path.join(run_folder, "per_sensor_plots")
    return os.path.exists(csv_path) and os.path.isdir(plots_dir)


@st.cache_data
def load_validation_data():
    """Load the TEP FaultFree Testing RData file."""
    result = pyreadr.read_r(os.path.join(DATA_PATH, "TEP_FaultFree_Testing.RData"))
    df = list(result.values())[0]
    return df


# ==========================================================
# MODEL & DATASET DEFINITIONS
# ==========================================================


class RunAwareSequenceDataset(Dataset):
    def __init__(self, runs, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.indices = []

        unique_runs = np.unique(runs)
        for run in unique_runs:
            run_indices = np.where(runs == run)[0]
            for i in range(len(run_indices) - seq_len):
                self.indices.append(run_indices[i])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        x = self.data[start : start + self.seq_len]
        y = self.data[start + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


class LSTMTwin(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out)


def run_per_sensor_analysis(run_folder: str, config: dict, progress_bar=None):
    """Run full per-sensor analysis: load model, infer, generate plots + CSV."""
    seq_len = config.get("SEQ_LEN", 80)
    hidden_dim = config.get("HIDDEN_DIM", 256)
    num_layers = config.get("NUM_LAYERS", 2)
    batch_size = config.get("BATCH_SIZE", 32)

    # Load scaler
    scaler = joblib.load(os.path.join(run_folder, "scaler.pkl"))

    # Load validation data
    val_df = load_validation_data()
    sensor_names = val_df.columns[3:].tolist()
    val_runs = val_df["simulationRun"].values
    val_X = val_df.iloc[:, 3:].values
    val_X = scaler.transform(val_X)

    # Build dataset & loader
    val_dataset = RunAwareSequenceDataset(val_runs, val_X, seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    input_dim = val_X.shape[1]
    model = LSTMTwin(
        input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers
    ).to(DEVICE)
    model.load_state_dict(
        torch.load(os.path.join(run_folder, "lstm_model.pt"), map_location=DEVICE)
    )
    model.eval()

    # Inference
    preds_list = []
    actuals_list = []
    total_batches = len(val_loader)

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(val_loader):
            xb = xb.to(DEVICE)
            out = model(xb)
            preds_list.append(out.cpu().numpy())
            actuals_list.append(yb.numpy())
            if progress_bar is not None:
                progress_bar.progress(
                    (batch_idx + 1) / total_batches,
                    text=f"Running inference... Batch {batch_idx + 1}/{total_batches}",
                )

    val_preds = np.vstack(preds_list)
    val_actuals = np.vstack(actuals_list)

    # Generate per-sensor plots
    sensor_plot_dir = os.path.join(run_folder, "per_sensor_plots")
    os.makedirs(sensor_plot_dir, exist_ok=True)

    r2_values = []
    for i, sensor_name in enumerate(sensor_names):
        r2_val = r2_score(val_actuals[:, i], val_preds[:, i])
        r2_values.append(r2_val)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(val_actuals[:500, i], label="Actual", linewidth=1)
        ax.plot(val_preds[:500, i], label="Predicted", linewidth=1)
        ax.legend()
        ax.set_title(f"{sensor_name} | R² = {r2_val:.4f}")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Scaled Value")
        ax.grid(True, alpha=0.3)

        filename = f"{sensor_name}_R2_{r2_val:.3f}.png"
        fig.savefig(
            os.path.join(sensor_plot_dir, filename), dpi=100, bbox_inches="tight"
        )
        plt.close(fig)

    # Save R2 CSV
    sensor_r2_df = pd.DataFrame({"Sensor": sensor_names, "R2": r2_values})
    sensor_r2_df.sort_values("R2", ascending=False).to_csv(
        os.path.join(run_folder, "sensor_r2_values.csv"), index=False
    )

    return sensor_r2_df.sort_values("R2", ascending=False)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">TEP Process Analysis</div>'
        '<div class="sidebar-subtitle">LSTM Digital Twin Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    run_folders = get_run_folders()
    if not run_folders:
        st.error("No experiment runs found in `results/`")
        st.stop()

    # Show folder names only (not full paths)
    folder_names = [os.path.basename(f) for f in run_folders]
    selected_name = st.selectbox(
        "Experiment Run",
        folder_names,
        index=len(folder_names) - 1,
        help="Select a training experiment to analyze",
    )
    selected_folder = os.path.join(RESULTS_DIR, selected_name)

    st.markdown('<div class="section-label">Hyperparameters</div>', unsafe_allow_html=True)

    # Parse and display config
    config = parse_config(selected_folder)
    if config:
        for key, val in config.items():
            st.markdown(
                f'<div class="config-param">'
                f'<span class="param-key">{key}</span>'
                f'<span class="param-val">{val}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("No config.txt found.")

    st.markdown('<div class="section-label">System</div>', unsafe_allow_html=True)

    # Device info
    device_type = "CUDA (GPU)" if DEVICE.type == "cuda" else "CPU"
    st.markdown(
        f'<div class="config-param">'
        f'<span class="param-key">DEVICE</span>'
        f'<span class="param-val">{device_type}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("", unsafe_allow_html=True)  # spacer

    # Per-sensor result status
    if has_persensor_results(selected_folder):
        st.markdown(
            '<span class="status-badge status-available">ANALYSIS COMPLETE</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-badge status-missing">ANALYSIS PENDING</span>',
            unsafe_allow_html=True,
        )


# ==========================================================
# MAIN CONTENT
# ==========================================================

st.markdown(f"## Model Analysis &mdash; `{selected_name}`")

# ----------------------------------------------------------
# TAB LAYOUT
# ----------------------------------------------------------

tab_data, tab_training, tab_sensors = st.tabs(
    ["Dataset", "Training Results", "Per-Sensor Analysis"]
)

# ==========================================================
# TAB 1: DATA PREVIEW
# ==========================================================

with tab_data:
    st.markdown('<div class="section-label">Validation Dataset — TEP FaultFree Testing</div>', unsafe_allow_html=True)

    with st.spinner("Loading TEP dataset..."):
        val_df = load_validation_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <h4>Observations (n)</h4>
                <div class="value">{val_df.shape[0]:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <h4>Variables (p)</h4>
                <div class="value">{val_df.shape[1]}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <h4>Sensor Channels</h4>
                <div class="value">{len(val_df.columns[3:])}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-label">Data Sample (first 20 observations)</div>', unsafe_allow_html=True)
    st.dataframe(val_df.head(20), use_container_width=True, height=400)

    st.markdown('<div class="section-label">Descriptive Statistics</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(
        {
            "Variable": val_df.columns,
            "dtype": [str(val_df[c].dtype) for c in val_df.columns],
            "Non-Null Count": [val_df[c].notna().sum() for c in val_df.columns],
            "Mean (μ)": [
                f"{val_df[c].mean():.4f}"
                if np.issubdtype(val_df[c].dtype, np.number)
                else "—"
                for c in val_df.columns
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, height=300)

# ==========================================================
# TAB 2: TRAINING RESULTS
# ==========================================================

with tab_training:
    st.markdown('<div class="section-label">Evaluation Metrics</div>', unsafe_allow_html=True)

    # Metrics
    metrics_path = os.path.join(selected_folder, "metrics.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics_text = f.read()

        # Parse metrics into sections
        sections = metrics_text.strip().split("\n\n")
        cols = st.columns(len(sections))
        for idx, section in enumerate(sections):
            lines = section.strip().split("\n")
            with cols[idx]:
                header = lines[0] if lines else "Metrics"
                st.markdown(
                    f"""<div class="metric-card">
                        <h4>{header}</h4>""",
                    unsafe_allow_html=True,
                )
                for line in lines[1:]:
                    if ":" in line:
                        key, val = line.split(":", 1)
                        st.markdown(f"**{key.strip()}:** `{val.strip()}`")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No metrics.txt found for this run.")

    st.markdown("---")

    st.markdown('<div class="section-label">Diagnostic Plots</div>', unsafe_allow_html=True)

    # Training plots
    plot_files = [
        ("training_loss.png", "Fig. 1 — Training Loss Curve"),
        ("prediction_plot.png", "Fig. 2 — Predicted vs Observed"),
        ("r2_per_sensor.png", "Fig. 3 — R² Distribution by Sensor"),
        ("residual_distribution.png", "Fig. 4 — Residual Distribution"),
    ]

    available_plots = [
        (fname, title)
        for fname, title in plot_files
        if os.path.exists(os.path.join(selected_folder, fname))
    ]

    if available_plots:
        # Display in 2-column grid
        for i in range(0, len(available_plots), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(available_plots):
                    fname, title = available_plots[i + j]
                    with col:
                        st.markdown(
                            f'<div class="figure-panel">'
                            f'<div class="fig-caption">{title}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(
                            os.path.join(selected_folder, fname),
                            use_container_width=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No diagnostic plots available for this experiment.")

# ==========================================================
# TAB 3: PER-SENSOR ANALYSIS
# ==========================================================

with tab_sensors:
    st.markdown(
        '<div class="section-label">Per-Sensor Coefficient of Determination (R²)</div>',
        unsafe_allow_html=True,
    )

    results_exist = has_persensor_results(selected_folder)

    if not results_exist:
        st.warning(
            "Per-sensor analysis has not been computed for this experiment. "
            "Run the analysis below to generate predictions and R² scores."
        )

        # Pre-check required files
        scaler_ok = os.path.exists(os.path.join(selected_folder, "scaler.pkl"))
        model_ok = os.path.exists(os.path.join(selected_folder, "lstm_model.pt"))

        if not scaler_ok or not model_ok:
            missing = []
            if not scaler_ok:
                missing.append("scaler.pkl")
            if not model_ok:
                missing.append("lstm_model.pt")
            st.error(f"Missing required files: {', '.join(missing)}")
            st.stop()

        if st.button(
            "Run Per-Sensor Analysis", type="primary", use_container_width=True
        ):
            progress_bar = st.progress(0, text="Initializing...")
            with st.spinner("Running analysis..."):
                sensor_r2_df = run_per_sensor_analysis(
                    selected_folder, config, progress_bar
                )
            progress_bar.empty()
            st.success("Per-sensor analysis completed successfully.")
            st.rerun()
    else:
        # Load existing results
        csv_path = os.path.join(selected_folder, "sensor_r2_values.csv")
        sensor_r2_df = pd.read_csv(csv_path)

        # Apply R² >= 0.4 display filter (hardcoded)
        r2_threshold = 0.4
        filtered_df = sensor_r2_df[sensor_r2_df["R2"] >= r2_threshold].reset_index(
            drop=True
        )

        # Summary metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <h4>Sensors</h4>
                    <div class="value">{len(filtered_df)}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            avg_r2 = filtered_df["R2"].mean() if len(filtered_df) > 0 else 0
            st.markdown(
                f"""<div class="metric-card">
                    <h4>Avg R²</h4>
                    <div class="value">{avg_r2:.4f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # R² Table
        st.markdown('<div class="section-label">R² Scores by Sensor</div>', unsafe_allow_html=True)
        st.dataframe(
            filtered_df.style.format({"R2": "{:.6f}"}).background_gradient(
                subset=["R2"], cmap="RdYlGn", vmin=0, vmax=1
            ),
            use_container_width=True,
            height=350,
        )

        # Per-sensor plots
        st.markdown('<div class="section-label">Prediction vs Observed — Per Sensor</div>', unsafe_allow_html=True)

        plots_dir = os.path.join(selected_folder, "per_sensor_plots")
        plot_files_in_dir = os.listdir(plots_dir) if os.path.isdir(plots_dir) else []

        # Build a lookup: sensor_name -> plot filename
        sensor_to_plot = {}
        for pf in plot_files_in_dir:
            # Filenames look like: xmeas_1_R2_0.997.png
            # Extract sensor name (everything before _R2_)
            if "_R2_" in pf:
                sensor_name = pf.split("_R2_")[0]
                sensor_to_plot[sensor_name] = pf

        # Display plots in a 2-column grid for filtered sensors
        displayed_sensors = filtered_df["Sensor"].tolist()

        for i in range(0, len(displayed_sensors), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j < len(displayed_sensors):
                    sensor = displayed_sensors[i + j]
                    r2_val = filtered_df[filtered_df["Sensor"] == sensor]["R2"].values[
                        0
                    ]
                    with col:
                        # Color based on R2 value
                        if r2_val >= 0.9:
                            color = "#16a34a"
                        elif r2_val >= 0.7:
                            color = "#ca8a04"
                        else:
                            color = "#ea580c"

                        st.markdown(
                            f"""<div class="sensor-card" style="border-left-color: {color};">
                                <span class="sensor-name" style="color: {color};">
                                    {sensor}
                                </span>
                                <span class="sensor-r2">
                                    R² = {r2_val:.4f}
                                </span>
                                <div class="r2-bar-container">
                                    <div class="r2-bar" style="width: {max(0, r2_val) * 100:.1f}%; background: {color};"></div>
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        # Show plot if available
                        if sensor in sensor_to_plot:
                            plot_path = os.path.join(plots_dir, sensor_to_plot[sensor])
                            st.markdown(
                                '<div class="sensor-figure">',
                                unsafe_allow_html=True,
                            )
                            st.image(plot_path, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
