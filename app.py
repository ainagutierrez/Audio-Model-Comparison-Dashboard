
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Audio Model Comparison",
    layout="wide"
)

st.title("Audio Model Performance Dashboard")
st.markdown("Interactive comparison of deepfake audio detection models across datasets and transformations.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("audio_model_results.csv")
    transformation_order = ["none", "time_stretch", "pitch_shift", "eq", "reverb", "white_noise"]
    df["transformation"] = pd.Categorical(df["transformation"], categories=transformation_order, ordered=True)
    return df

df = load_data()

MODELS = sorted(df["model"].unique())
DATASETS = sorted(df["dataset"].unique())
TRANSFORMATIONS = list(df["transformation"].cat.categories)

# Sidebar filters
st.sidebar.header("Filters")

selected_models = st.sidebar.multiselect(
    "Models", MODELS, default=MODELS,
    help="Select which models to include"
)
selected_datasets = st.sidebar.multiselect(
    "Datasets", DATASETS, default=DATASETS,
    help="Select which datasets to include"
)
selected_transformations = st.sidebar.multiselect(
    "Transformations", TRANSFORMATIONS, default=TRANSFORMATIONS,
    help="Select which audio transformations to include"
)

filtered = df[
    df["model"].isin(selected_models) &
    df["dataset"].isin(selected_datasets) &
    df["transformation"].isin(selected_transformations)
]

if filtered.empty:
    st.warning("No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab5 = st.tabs([
    "Overview",
    "Heatmaps",
    "Robustness Curves",
    "Detail Explorer",
])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Models", len(selected_models))
    col2.metric("Datasets", len(selected_datasets))
    col3.metric("Transformations", len(selected_transformations))

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        model_avg = (
            filtered.groupby("model")["accuracy"].mean().reset_index()
            .sort_values("accuracy")
        )
        fig = px.bar(
            model_avg, x="accuracy", y="model", orientation="h",
            color="accuracy", color_continuous_scale="Viridis",
            labels={"accuracy": "Average Accuracy (%)", "model": "Model"},
            title="Average Performance per Model",
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        dataset_avg = (
            filtered.groupby("dataset")["accuracy"].mean().reset_index()
            .sort_values("accuracy")
        )
        fig = px.bar(
            dataset_avg, x="accuracy", y="dataset", orientation="h",
            color="accuracy", color_continuous_scale="RdYlGn",
            labels={"accuracy": "Average Accuracy (%)", "dataset": "Dataset"},
            title="Dataset Difficulty (Lower to Harder)",
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    transformation_avg = (
        filtered.groupby("transformation")["accuracy"].mean().reset_index()
    )
    fig = px.bar(
        transformation_avg, x="transformation", y="accuracy",
        color="accuracy", color_continuous_scale="Blues",
        labels={"accuracy": "Average Accuracy (%)", "transformation": "Transformation"},
        title="Impact of Audio Transformations",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: HEATMAPS ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("Performance Heatmap — Dataset × Model")

    transform_choice = st.selectbox(
        "Select transformation", TRANSFORMATIONS,
        key="heatmap_transform"
    )

    subset = filtered[filtered["transformation"] == transform_choice]
    pivot = subset.pivot_table(index="dataset", columns="model", values="accuracy")

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        color_continuous_scale="Viridis",
        labels={"color": "Accuracy (%)"},
        title=f"Heatmap — {transform_choice}",
        aspect="auto",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### All Transformations (grid view)")
    cols = st.columns(2)
    for i, t in enumerate(TRANSFORMATIONS):
        s = filtered[filtered["transformation"] == t]
        p = s.pivot_table(index="dataset", columns="model", values="accuracy")
        fig2 = px.imshow(
            p, text_auto=".1f", color_continuous_scale="Viridis",
            title=f"{t}", aspect="auto"
        )
        fig2.update_layout(height=320, margin=dict(t=40, b=10))
        cols[i % 2].plotly_chart(fig2, use_container_width=True)

# ── TAB 3: ROBUSTNESS CURVES ──────────────────────────────────────────────────
with tab3:
    st.subheader("Model Robustness to Audio Transformations")

    groupby_choice = st.radio(
        "Aggregate by", ["Overall", "Per Dataset"], horizontal=True,
        key="robustness_group"
    )

    if groupby_choice == "Overall":
        model_means = (
            filtered.groupby(["transformation", "model"])["accuracy"]
            .mean().reset_index()
        )
        fig = px.line(
            model_means, x="transformation", y="accuracy",
            color="model", markers=True,
            labels={"accuracy": "Accuracy (%)", "transformation": "Transformation"},
            title="Robustness across transformations (all datasets averaged)",
        )
        fig.update_layout(height=500, legend=dict(orientation="v"))
        st.plotly_chart(fig, use_container_width=True)
    else:
        dataset_sel = st.selectbox("Dataset", selected_datasets, key="robustness_dataset")
        model_means = (
            filtered[filtered["dataset"] == dataset_sel]
            .groupby(["transformation", "model"])["accuracy"].mean().reset_index()
        )
        fig = px.line(
            model_means, x="transformation", y="accuracy",
            color="model", markers=True,
            labels={"accuracy": "Accuracy (%)", "transformation": "Transformation"},
            title=f"Robustness — {dataset_sel}",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 5: DETAIL EXPLORER ────────────────────────────────────────────────────
with tab5:
    st.subheader("Row-level Explorer")

    c1, c2, c3 = st.columns(3)
    model_sel = c1.selectbox("Model", ["All"] + selected_models, key="detail_model")
    dataset_sel2 = c2.selectbox("Dataset", ["All"] + selected_datasets, key="detail_dataset")
    transform_sel2 = c3.selectbox("Transformation", ["All"] + TRANSFORMATIONS, key="detail_transform")

    detail = filtered.copy()
    if model_sel != "All":
        detail = detail[detail["model"] == model_sel]
    if dataset_sel2 != "All":
        detail = detail[detail["dataset"] == dataset_sel2]
    if transform_sel2 != "All":
        detail = detail[detail["transformation"] == transform_sel2]

    st.dataframe(
        detail.sort_values("accuracy", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    if not detail.empty:
        st.markdown(
            f"**{len(detail)} rows** | "
            f"Mean: **{detail['accuracy'].mean():.2f}%** | "
            f"Min: **{detail['accuracy'].min():.2f}%** | "
            f"Max: **{detail['accuracy'].max():.2f}%**"
        )
