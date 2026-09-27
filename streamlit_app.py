"""
TraceLink AI — Real AI Mode
-----------------------------
Single-file Streamlit application. All local text-parsing has been
removed (no tokenizer, no line_dict, no keyword-weight/proximity
scoring, no regex-style loops). Structured extraction is now performed
entirely by the Anthropic API using forced tool-use, which guarantees
a schema-conformant JSON object back — no manual JSON-string parsing
or markdown-fence stripping required.

What this file still does locally, in plain Python, and why:
  * The API key is read once via os.environ.get("ANTHROPIC_API_KEY")
    and never hard-coded — set it in Streamlit Cloud's
    Settings -> Secrets panel as ANTHROPIC_API_KEY = "sk-ant-...".
  * The pass/fail *decision* (margin = lowest reading - MAT, and the
    ASME B31.3 fallback arithmetic when no MAT is stated) stays in
    deterministic Python. Anthropic's model extracts the numbers;
    it does not get asked to "decide" the compliance verdict itself,
    so the safety-critical comparison is auditable and reproducible.

Model note: the request specified `claude-3-5-sonnet-20241022`, which
has been retired on the Claude API. This uses the current comparable
model, `claude-sonnet-5` — change MODEL_NAME below if your account
should target a different one.
"""

import sys
import subprocess

# ============================================================================
# AUTOMATED RUNTIME INSTALLER — runs before any other import in this file.
# ----------------------------------------------------------------------------
# Belt-and-suspenders fix for a Streamlit Cloud container that boots from a
# stale/cached environment and skips requirements.txt: if a package can't be
# imported, install it with the *same* interpreter running this script
# (sys.executable — not a bare "pip", which can resolve to a different
# environment) and try the import again. Real fix to also apply on your end:
# confirm requirements.txt sits at the repo root next to this file, then use
# "Reboot app" (not just a rerun) in Streamlit Cloud so it rebuilds the
# environment from scratch. This installer just makes the app self-healing
# either way.
# ============================================================================


def _ensure_package(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        __import__(import_name)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pip_name])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Automatic install of '{pip_name}' failed (exit code {e.returncode}). "
                f"Add '{pip_name}' to requirements.txt and reboot the app on Streamlit Cloud."
            ) from e
        __import__(import_name)


_ensure_package("anthropic")
_ensure_package("streamlit")

import os
import json

import streamlit as st
import anthropic

st.set_page_config(
    page_title="TraceLink AI — Real AI Mode",
    layout="wide",
    page_icon="🛠️",
)

MODEL_NAME = "claude-sonnet-5"  # see note above — claude-3-5-sonnet-20241022 is retired

# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """You are a structured-data extraction engine for industrial \
engineering inspection reports (pressure vessels, process piping, structural \
steel, heat exchangers, storage tanks, and any other mechanical asset).

Read the raw inspection report text the user provides and call the \
`extract_compliance_data` tool with the data you find. Rules:
- Extract only what is actually stated in the text. Never invent, estimate, \
  or "helpfully" fill in a value that is not present — use null for anything \
  not stated.
- "design_limit_mat" is whatever the report calls its minimum/allowable/ \
  permitted thickness limit (may be labeled MAT, minimum allowable thickness, \
  minimum wall, etc.). If no such limit is explicitly stated, leave it null \
  even if you could calculate one yourself — calculation is handled outside \
  this tool.
- "design_variables" are the ASME B31.3 pipe-wall inputs (design pressure, \
  outside diameter, allowable stress, quality/joint factor, Y coefficient, \
  corrosion allowance) — populate only the ones actually present in the text.
- "ut_readings" should include every individual thickness/UT measurement \
  point in the report, keyed by whatever label the report uses for it \
  (e.g. "A1", "West HSS", "Gusset").
- "field_anomalies" should capture every inspector note, recommendation, \
  flagged indication, weld flaw, bolt condition issue, or similar unresolved \
  mechanical observation, in the report's own words.
- Do not comment on compliance, pass/fail, or safety — only extract data."""

EXTRACTION_TOOL = {
    "name": "extract_compliance_data",
    "description": "Record structured fields extracted from an industrial inspection report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "asset_category": {
                "type": ["string", "null"],
                "description": "The asset's classification/type as stated in the report.",
            },
            "metallurgy": {
                "type": ["string", "null"],
                "description": "Material/metallurgy specification as stated in the report.",
            },
            "nominal_thickness": {
                "type": ["object", "null"],
                "properties": {"value": {"type": "number"}, "unit": {"type": ["string", "null"]}},
                "required": ["value"],
            },
            "design_limit_mat": {
                "type": ["object", "null"],
                "description": "The explicitly stated minimum/allowable thickness limit, if any.",
                "properties": {"value": {"type": "number"}, "unit": {"type": ["string", "null"]}},
                "required": ["value"],
            },
            "design_variables": {
                "type": ["object", "null"],
                "description": "ASME B31.3 pipe-wall inputs, only if present in the text.",
                "properties": {
                    "design_pressure": {"type": ["number", "null"]},
                    "outside_diameter": {"type": ["number", "null"]},
                    "allowable_stress": {"type": ["number", "null"]},
                    "quality_factor": {"type": ["number", "null"]},
                    "y_coefficient": {"type": ["number", "null"]},
                    "corrosion_allowance": {"type": ["number", "null"]},
                },
            },
            "ut_readings": {
                "type": "object",
                "description": "Every measurement point, keyed by its label in the report.",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"value": {"type": "number"}, "unit": {"type": ["string", "null"]}},
                    "required": ["value"],
                },
            },
            "field_anomalies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Inspector notes, flaws, indications, or unresolved mechanical items.",
            },
        },
        "required": ["asset_category", "ut_readings", "field_anomalies"],
    },
}


def extract_with_claude(client, report_text):
    """Send the raw report straight to the model and force a tool call, so
    the SDK hands back an already-parsed Python dict — no JSON.loads /
    fence-stripping needed on our side."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_compliance_data"},
        messages=[{"role": "user", "content": report_text}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_compliance_data":
            return block.input
    raise RuntimeError("Model did not return a structured extraction — no tool_use block found.")


# ============================================================================
# DETERMINISTIC EVALUATION (stays in plain Python — not delegated to the model)
# ============================================================================

def evaluate(extracted):
    mat_field = extracted.get("design_limit_mat")
    mat = mat_field["value"] if mat_field else None
    calc_note = None

    if mat is None:
        dv = extracted.get("design_variables") or {}
        P, D, S, E, Y, CA = (
            dv.get("design_pressure"), dv.get("outside_diameter"), dv.get("allowable_stress"),
            dv.get("quality_factor"), dv.get("y_coefficient"), dv.get("corrosion_allowance"),
        )
        missing = [name for name, val in [
            ("Design Pressure", P), ("Outside Diameter", D), ("Allowable Stress", S),
            ("Quality/Joint Factor", E), ("Y Coefficient", Y), ("Corrosion Allowance", CA),
        ] if val is None]

        if missing:
            return {"status": "insufficient", "missing_vars": missing}

        t_design = (P * D) / (2 * (S * E + P * Y))
        mat = t_design + CA
        calc_note = (
            f"No explicit MAT/threshold was stated, so the ASME B31.3 straight-pipe formula was "
            f"applied to the model-extracted variables: t_design = (P×D)/(2×(S×E+P×Y)) = "
            f"{t_design:.4f}. Adding the corrosion allowance ({CA:.3f}) gives a calculated safety "
            f"ceiling of {mat:.4f}."
        )

    readings = extracted.get("ut_readings") or {}
    if not readings:
        return {"status": "no_measurements", "mat": mat, "calc_note": calc_note}

    lowest_label, lowest_entry = min(readings.items(), key=lambda kv: kv[1]["value"])
    lowest_val = lowest_entry["value"]
    margin = round(lowest_val - mat, 4)
    status = "blocked" if margin < 0 else "verified"

    return {
        "status": status, "mat": mat, "lowest_label": lowest_label, "lowest_val": lowest_val,
        "margin": margin, "calc_note": calc_note,
    }


# ============================================================================
# UI
# ============================================================================

st.markdown(
    """
    <style>
    .metric-box {background:#111827;border:1px solid #374151;border-radius:10px;
        padding:12px 16px;margin-bottom:8px;}
    .metric-label {color:#9CA3AF;font-size:0.75rem;text-transform:uppercase;letter-spacing:.04em;}
    .metric-value {color:#F9FAFB;font-size:1.25rem;font-weight:700;word-wrap:break-word;}
    .anomaly-box {background:#3F2D0B;border:1px solid #A16207;border-radius:10px;
        padding:12px 16px;margin-bottom:6px;color:#FDE68A;}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_box(col, label, value):
    col.markdown(
        f"<div class='metric-box'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div></div>",
        unsafe_allow_html=True,
    )


st.title("🛠️ TraceLink AI — Real AI Mode")
st.caption(f"Structured extraction via the Anthropic API ({MODEL_NAME}). No local parsing logic.")

client = get_client()
if client is None:
    st.error(
        "ANTHROPIC_API_KEY is not set. In Streamlit Cloud, open **Settings → Secrets** for this app "
        "and add:\n\n```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
    )
    st.stop()

uploaded = st.file_uploader("Drop an inspection report (.txt)", type=["txt"])

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    report_text = raw_bytes.decode("utf-8", errors="ignore")
    cache_key = f"{uploaded.name}:{len(raw_bytes)}:{hash(raw_bytes)}"

    if st.session_state.get("cache_key") != cache_key:
        with st.spinner(f"Sending report to {MODEL_NAME} for structured extraction..."):
            try:
                extracted = extract_with_claude(client, report_text)
                st.session_state["extracted"] = extracted
                st.session_state["cache_key"] = cache_key
                st.session_state["api_error"] = None
            except anthropic.APIError as e:
                st.session_state["api_error"] = f"Anthropic API error: {e}"
            except Exception as e:
                st.session_state["api_error"] = f"Extraction failed: {e}"

    if st.session_state.get("api_error"):
        st.error(st.session_state["api_error"])
        st.stop()

    extracted = st.session_state["extracted"]
    result = evaluate(extracted)

    st.subheader("📋 Data Extracted From the Inspection Report")
    c1, c2 = st.columns(2)
    metric_box(c1, "Asset Category", extracted.get("asset_category") or "Not stated")
    metric_box(c2, "Metallurgy", extracted.get("metallurgy") or "Not stated")

    if result["status"] == "insufficient":
        st.warning("**CONDITION UNVERIFIED — INSUFFICIENT BOUNDARY DATA INPUTS**")
        st.write("No MAT/threshold was stated, and these variables needed to calculate one "
                 "are also missing from the report:")
        for name in result["missing_vars"]:
            st.markdown(f"- ❌ **{name}**")
    else:
        m1, m2, m3 = st.columns(3)
        metric_box(m1, "Safety Ceiling (MAT)", f"{result['mat']:.4f}")
        if result["status"] != "no_measurements":
            metric_box(m2, "Lowest Reading", f"{result['lowest_label']}: {result['lowest_val']:.4f}")
            metric_box(m3, "Margin", f"{result['margin']:+.4f}")

        if result.get("calc_note"):
            st.info(result["calc_note"])

        if result["status"] == "blocked":
            st.error("🔴 **CRITICAL BOUNDARY DEFECT — COMPLIANCE STATUS: BLOCKED**")
            st.markdown(
                "- Lowest captured reading is below the safety ceiling.\n"
                "- Component is BLOCKED from continued service pending engineering disposition.\n"
                "- Route to Fitness-for-Service (FFS) / Authorized Inspector review."
            )
        elif result["status"] == "verified":
            st.success("🟢 **COMPLIANCE STATUS: VERIFIED SECURE**")
            st.markdown("- All captured readings are at or above the safety ceiling. Continue routine monitoring.")
        elif result["status"] == "no_measurements":
            st.warning("A safety ceiling was determined, but no UT/measurement readings were extracted from this report.")

    ut_readings = extracted.get("ut_readings") or {}
    if ut_readings:
        with st.expander(f"Full UT reading dictionary — {len(ut_readings)} point(s)"):
            for label, entry in sorted(ut_readings.items(), key=lambda kv: kv[1]["value"]):
                unit = f" {entry['unit']}" if entry.get("unit") else ""
                st.write(f"- {label}: {entry['value']}{unit}")

    anomalies = extracted.get("field_anomalies") or []
    if anomalies:
        st.markdown("### ⚠️ Unresolved Mechanical Anomaly Logs")
        for note in anomalies:
            st.markdown(f"<div class='anomaly-box'>{note}</div>", unsafe_allow_html=True)

    with st.expander("Raw structured response from the model"):
        st.json(extracted)

else:
    st.info("Upload a .txt inspection report above to run extraction.")
