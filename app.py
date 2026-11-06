"""Run with: streamlit run app.py"""

import matplotlib.pyplot as plt
import streamlit as st

from kronos_b3_lab.data import demo_candles, validate_candles
from kronos_b3_lab.evaluation import Config, evaluate
from kronos_b3_lab.models import make_models
from kronos_b3_lab.report import figure

st.set_page_config(page_title="Kronos / B3 Lab", page_icon="◈", layout="wide")
st.caption("VICTOR PIMENTA  /  OPEN RESEARCH  /  01")
st.title("Forecasts deserve a fair test.")
st.write("Kronos · TimesFM · Chronos. Same origins, honest baselines, inspectable results.")
st.warning(
    "Research only. No investment advice. Default data are synthetic, not historical B3 prices."
)
with st.sidebar:
    st.header("Experiment")
    st.caption("Offline baselines load instantly. Models download weights only after you run.")
    selected = st.multiselect(
        "Forecasters",
        ["seasonal-naive", "random-walk", "kronos", "timesfm", "chronos"],
        default=["seasonal-naive", "random-walk"],
    )
    context = st.slider("Context / sessions", 20, 120, 60, 5)
    horizon = st.slider("Horizon / sessions", 1, 20, 5)
    costs = st.number_input("Cost / bps per side", min_value=0.0, max_value=1000.0, value=10.0)
    asset_file = st.file_uploader("Asset candles (optional)", type="csv")
    ibov_file = st.file_uploader("IBOV candles (required with asset)", type="csv")
    run = st.button("Run experiment", type="primary", use_container_width=True)

if run or "result" not in st.session_state:
    import pandas as pd

    try:
        if bool(asset_file) != bool(ibov_file):
            raise ValueError(
                "Upload both CSVs together, or leave both empty for the synthetic demo."
            )
        if asset_file and ibov_file:
            asset = validate_candles(pd.read_csv(asset_file))
            ibov = validate_candles(pd.read_csv(ibov_file))
            label = "USER CSV · VERIFY SOURCE AND ADJUSTMENTS"
        else:
            asset, ibov = demo_candles()
            label = "SYNTHETIC FIXTURE · NOT MARKET PERFORMANCE"
        with st.spinner("Evaluating untouched future windows…"):
            config = Config(context, horizon, costs)
            models = make_models(selected, context, horizon, config.seed)
            result = evaluate(asset, ibov, models, config)
        st.session_state["result"] = result
        st.session_state["label"] = label
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        st.error(str(exc))
        st.stop()

result = st.session_state["result"]
left, middle, right = st.columns(3)
left.metric("Forecasters", len(result.metrics))
middle.metric("Untouched windows", result.predictions["origin"].nunique())
right.metric("Cost per side", f"{result.config['cost_bps']:g} bps")
st.caption(
    f"Last completed run · context {result.config['context']} · horizon {result.config['horizon']}"
)
chart = figure(result, st.session_state["label"])
st.pyplot(chart)
plt.close(chart)
st.subheader("Evidence, before conclusions")
st.dataframe(result.metrics, hide_index=True, use_container_width=True)
st.download_button(
    "Download forecasts · CSV",
    result.predictions.to_csv(index=False),
    "predictions.csv",
    "text/csv",
)
with st.expander("How this experiment works"):
    st.write(
        "Each window sees past candles and future dates only. A positive terminal forecast "
        "opens a long position at the next session open; otherwise capital stays in cash. Every "
        "position closes at the window end. Fees apply to both sides. IBOV is an index comparator, "
        "not an executable instrument. Window-end drawdown understates intrawindow risk. "
        "Changing parameters after inspecting results is test-set tuning, not independent evidence."
    )
