"""
TraceLink AI — Real AI Mode (Multi-Component Schema)
-----------------------------------------------------
Single-file Streamlit application.

Structured extraction is performed entirely by the Anthropic API using
forced tool-use against a hierarchical, multi-component schema. A single
inspected asset (a structural frame, a heat exchanger, a piping run, ...)
can contain many independent sub-components (columns, base plates,
gussets, flanges, shells, ...), each with its own minimum allowable
thickness and its own list of UT thickness readings. This file evaluates
every sub-component separately rather than collapsing the asset into one
flat number.

What stays in plain, deterministic Python, and why:
  * The pass/fail decision for each component (margin = lowest UT reading
    for that component - that component's minimum allowable thickness,
    with an ASME B31.3 fallback calculation when a component is flagged
    as pressurized piping and has no stated limit) is computed in plain
    Python, not by the model. Anthropic's model only extracts the raw
    numbers from the report text; it never renders a compliance verdict
    itself, so the safety-critical arithmetic is auditable and
    reproducible independent of the model's phrasing on any given call.

Model note: "claude-3-5-sonnet-latest" was requested, but that model has
been retired on the Claude API. This uses a current model instead —
change MODEL_NAME below if your account should target a different one.
"""

import os
import json

import streamlit as st
import anthropic

st.set_page_config(
    page_title="TraceLink AI — Real AI Mode",
    layout="wide",
    page_icon="🛠️",
)

MODEL_NAME = "claude-sonnet-5"  # claude-3-5-sonnet-latest is retired; see note above

# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """You are a structured-data extraction engine for industrial \
engineering inspection reports covering multi-component assets: structural \
steel assemblies (columns, base plates, gussets, beams), pressure vessels, \
process piping, heat exchangers, storage tanks, conveyor frames, and similar \
mechanical assets.

Read the raw inspection report text the user provides and call the \
`extract_compliance_data` tool with the data you find. Rules:
- Extract only what is actually stated in the text. Never invent, estimate, \
  or "helpfully" fill in a value that is not present — use null for anything \
  not stated.
- Treat the asset as a collection of independent sub-components. Each \
  distinct structural or mechanical element mentioned (e.g. "North Column", \
  "Base Plates", "Gusset G1", "Beam Flange", "Shell Course 1") becomes its \
  own entry in components_matrix, with its own minimum_allowable_thickness \
  and its own ut_measurements list. Do not merge readings from different \
  components into one entry, and do not invent components that are not \
  named or clearly implied in the text.
- "minimum_allowable_thickness" is whatever the report calls that \
  component's minimum/allowable/permitted thickness limit (may be labeled \
  MAT, minimum allowable thickness, minimum wall, retirement thickness, \
  etc.). If no such limit is explicitly stated for that component, leave it \
  null even if you could calculate one yourself — calculation is handled \
  outside this tool.
- "is_pressurized_piping" should be true only if the asset itself is a \
  pressurized pipe/piping run and at least one component lacks a stated \
  limit, since that is what triggers the ASME B31.3 fallback math.
- "design_variables" are the ASME B31.3 pipe-wall inputs (design pressure, \
  outside diameter, allowable stress, quality/joint factor, Y coefficient, \
  corrosion allowance) — populate only the ones actually present in the \
  text. These apply at the asset level, not per component.
- "ut_measurements" for each component should include every individual \
  thickness/UT reading taken on that specific component, with whatever \
  location label the report uses (e.g. "C1", "F3", "Point A").
- "field_anomalies" should capture every inspector note, recommendation, \
  flagged indication, weld flaw, un-torqued bolt, or geometry/alignment \
  issue, in the report's own words, regardless of which component it \
  relates to.
- Do not comment on compliance, pass/fail, or safety — only extract data."""

EXTRACTION_TOOL = {
    "name": "extract_compliance_data",
    "description": "Record structured, multi-component fields extracted from an industrial inspection report.",
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
            "is_pressurized_piping": {
                "type": "boolean",
                "description": "True if the asset is a pressurized pipe/piping run lacking a stated limit, triggering the B31.3 fallback.",
            },
            "design_variables": {
                "type": ["object", "null"],
                "description": "ASME B31.3 pipe-wall inputs, only if present in the text. Applies at the asset level.",
                "properties": {
                    "design_pressure": {"type": ["number", "null"]},
                    "outside_diameter": {"type": ["number", "null"]},
                    "allowable_stress": {"type": ["number", "null"]},
                    "quality_factor": {"type": ["number", "null"]},
                    "y_coefficient": {"type": ["number", "null"]},
                    "corrosion_allowance": {"type": ["number", "null"]},
                },
            },
            "components_matrix": {
                "type": "array",
                "description": "One entry per independent sub-component of the asset.",
                "items": {
                    "type": "object",
                    "properties": {
                        "component_name": {"type": "string"},
                        "minimum_allowable_thickness": {
                            "type": ["object", "null"],
                            "properties": {
                                "value": {"type": ["number", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                        },
                        "nominal_thickness": {
                            "type": ["object", "null"],
                            "properties": {
                                "value": {"type": ["number", "null"]},
                                "unit": {"type": ["string", "null"]},
                            },
                        },
                        "ut_measurements": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "location_label": {"type": "string"},
                                    "value": {"type": "number"},
                                    "unit": {"type": ["string", "null"]},
                                },
                                "required": ["location_label", "value"],
                            },
                        },
                    },
                    "required": ["component_name", "ut_measurements"],
                },
            },
            "field_anomalies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Inspector notes, flaws, indications, or unresolved mechanical items.",
            },
        },
        "required": ["asset_category", "components_matrix", "field_anomalies"],
    },
}


def extract_with_claude(client, report_text):
    """Send the report to Claude and return the structured extraction as a Python dict."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "tool", "name": "extract_compliance_data"},
        messages=[{"role": "user", "content": report_text}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_compliance_data":
            return dict(block.input)

    raise RuntimeError(
        "Model did not return a structured extraction — no tool_use block found."
    )

# ============================================================================
# DETERMINISTIC EVALUATION (stays in plain Python — not delegated to the model)
# ============================================================================

B31_3_VARS = [
    ("design_pressure", "Design Pressure"),
    ("outside_diameter", "Outside Diameter"),
    ("allowable_stress", "Allowable Stress"),
    ("quality_factor", "Quality/Joint Factor"),
    ("y_coefficient", "Y Coefficient"),
    ("corrosion_allowance", "Corrosion Allowance"),
]


def calc_b31_3_mat(design_variables):
    """Return (mat, missing_var_names, calc_note). mat is None if inputs are incomplete."""
    dv = design_variables or {}
    missing = [label for key, label in B31_3_VARS if dv.get(key) is None]
    if missing:
        return None, missing, None

    P = dv["design_pressure"]
    D = dv["outside_diameter"]
    S = dv["allowable_stress"]
    E = dv["quality_factor"]
    Y = dv["y_coefficient"]
    CA = dv["corrosion_allowance"]

    t_design = (P * D) / (2 * (S * E + P * Y))
    mat = t_design + CA
    calc_note = (
        f"No explicit minimum allowable thickness was stated for this component, so the "
        f"ASME B31.3 straight-pipe formula was applied to the model-extracted design "
        f"variables: t_design = (P×D)/(2×(S×E+P×Y)) = {t_design:.4f}. Adding the corrosion "
        f"allowance ({CA:.3f}) gives a calculated safety ceiling of {mat:.4f}."
    )
    return mat, [], calc_note


def evaluate_component(component, is_pressurized_piping, design_variables):
    name = component.get("component_name") or "Unnamed Component"
    ut_measurements = component.get("ut_measurements") or []
    mat_field = component.get("minimum_allowable_thickness")
    mat = mat_field.get("value") if mat_field else None
    mat_unit = mat_field.get("unit") if mat_field else None
    calc_note = None

    if not ut_measurements:
        return {"name": name, "status": "no_measurements", "mat": mat, "mat_unit": mat_unit}

    if mat is None:
        if is_pressurized_piping:
            mat, missing, calc_note = calc_b31_3_mat(design_variables)
            if mat is None:
                return {
                    "name": name, "status": "insufficient",
                    "missing_vars": missing, "ut_measurements": ut_measurements,
                }
        else:
            return {
                "name": name, "status": "insufficient",
                "missing_vars": ["Minimum Allowable Thickness (not stated for this component)"],
                "ut_measurements": ut_measurements,
            }

    lowest = min(ut_measurements, key=lambda r: r["value"])
    margin = round(lowest["value"] - mat, 4)
    status = "blocked" if margin < 0 else "verified"

    return {
        "name": name, "status": status, "mat": mat, "mat_unit": mat_unit,
        "lowest": lowest, "margin": margin, "calc_note": calc_note,
    }


def evaluate(extracted):
    components = extracted.get("components_matrix") or []
    is_piping = bool(extracted.get("is_pressurized_piping"))
    design_variables = extracted.get("design_variables")

    results = [evaluate_component(c, is_piping, design_variables) for c in components]
    blocked = [r for r in results if r["status"] == "blocked"]
    unresolved = [r for r in results if r["status"] in ("insufficient", "no_measurements")]

    if not results:
        global_status = "no_components"
    elif blocked:
        global_status = "BLOCKED"
    elif unresolved or extracted.get("field_anomalies"):
        global_status = "CONDITION NOT FULLY VERIFIED - ENGINEERING REVIEW REQUIRED"
    else:
        global_status = "VERIFIED SECURE"

    return {"results": results, "blocked": blocked, "unresolved": unresolved, "global_status": global_status}


def remediation_steps(component_name):
    """Bulleted next steps, tailored a little by the kind of component involved."""
    n = component_name.lower()
    if "column" in n:
        return [
            "Route to a structural engineer for a reduced-section load rating before any load is reapplied.",
            "Verify against adjacent column readings to rule out a localized corrosion cell.",
            "Consider a doubler plate or full section replacement per the engineer's disposition.",
        ]
    if "base plate" in n or "baseplate" in n:
        return [
            "Check anchor bolt torque and grout condition — thinning at a base plate often co-occurs with bolt/grout issues.",
            "Confirm bearing area is still adequate for the design axial/moment load at the reduced thickness.",
            "Schedule replacement or plate doubling per Authorized Inspector / structural engineer review.",
        ]
    if "gusset" in n:
        return [
            "Evaluate remaining gusset section against the connection's design shear/moment capacity.",
            "Inspect welds at the gusset-to-member interface for cracking driven by the reduced stiffness.",
            "Do not defer — gussets are frequently the limiting element in a braced connection.",
        ]
    if "flange" in n:
        return [
            "Re-torque per the flange's bolt pattern and verify gasket seating before returning to service.",
            "Check for leakage/weeping at the reduced-thickness zone under normal operating pressure.",
            "Route to piping engineering for a B31.3 reassessment of this joint.",
        ]
    if "shell" in n or "head" in n:
        return [
            "Perform a local Fitness-for-Service (FFS) assessment (e.g. API 579 Level 1/2) before continued operation.",
            "Grid the surrounding area to bound the extent of the thin region.",
            "Consider a pressure de-rate as an interim measure pending repair.",
        ]
    return [
        "Route this component to Fitness-for-Service (FFS) / Authorized Inspector review.",
        "Bound the extent of the thin area with supplemental UT grid readings.",
        "Do not return the component to unrestricted service until disposition is issued.",
    ]

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
    .component-card {border-radius:12px;padding:16px 18px;margin-bottom:14px;border:1px solid #374151;}
    .component-card.blocked {background:#450A0A;border-color:#DC2626;}
    .component-card.verified {background:#052E1B;border-color:#16A34A;}
    .component-card.unresolved {background:#1F2937;border-color:#6B7280;}
    .component-title {font-size:1.05rem;font-weight:700;color:#F9FAFB;margin-bottom:8px;}
    .status-pill {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.72rem;
        font-weight:700;letter-spacing:.03em;text-transform:uppercase;margin-left:8px;}
    .status-pill.blocked {background:#DC2626;color:#FEF2F2;}
    .status-pill.verified {background:#16A34A;color:#F0FDF4;}
    .status-pill.unresolved {background:#6B7280;color:#F9FAFB;}
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


def render_component_card(result):
    status = result["status"]
    css_class = "blocked" if status == "blocked" else ("verified" if status == "verified" else "unresolved")
    pill_label = {"blocked": "Blocked", "verified": "Verified", "insufficient": "Unresolved", "no_measurements": "No Data"}[status]

    st.markdown(
        f"<div class='component-card {css_class}'>"
        f"<div class='component-title'>{result['name']}"
        f"<span class='status-pill {css_class}'>{pill_label}</span></div>",
        unsafe_allow_html=True,
    )

    if status in ("blocked", "verified"):
        unit_suffix = f" {result['mat_unit']}" if result.get("mat_unit") else ""
        lowest = result["lowest"]
        lowest_unit = f" {lowest['unit']}" if lowest.get("unit") else ""
        c1, c2, c3 = st.columns(3)
        metric_box(c1, "Discovered Minimum Limit", f"{result['mat']:.4f}{unit_suffix}")
        metric_box(c2, "Lowest Floor Reading", f"{lowest['location_label']}: {lowest['value']:.4f}{lowest_unit}")
        metric_box(c3, "True Margin", f"{result['margin']:+.4f}")
        if result.get("calc_note"):
            st.info(result["calc_note"])

        if status == "blocked":
            st.markdown("**Engineering remediation steps:**")
            for step in remediation_steps(result["name"]):
                st.markdown(f"- {step}")

    elif status == "insufficient":
        st.write("Cannot compute a margin for this component — missing:")
        for name in result["missing_vars"]:
            st.markdown(f"- ❌ **{name}**")
        readings = result.get("ut_measurements") or []
        if readings:
            labels = ", ".join(f"{r['location_label']}={r['value']}" for r in readings)
            st.caption(f"Readings on file for this component: {labels}")

    else:  # no_measurements
        st.write("A limit is on file for this component, but no UT readings were extracted for it.")

    st.markdown("</div>", unsafe_allow_html=True)


st.title("🛠️ TraceLink AI — Real AI Mode")
st.caption(f"Structured, multi-component extraction via the Anthropic API ({MODEL_NAME}).")

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
    outcome = evaluate(extracted)

    st.subheader("📋 Data Extracted From the Inspection Report")
    c1, c2, c3 = st.columns(3)
    metric_box(c1, "Asset Category", extracted.get("asset_category") or "Not stated")
    metric_box(c2, "Metallurgy", extracted.get("metallurgy") or "Not stated")
    metric_box(c3, "Pressurized Piping", "Yes" if extracted.get("is_pressurized_piping") else "No")

    st.markdown("---")

    gs = outcome["global_status"]
    if gs == "no_components":
        st.warning("No sub-components were extracted from this report.")
    elif gs == "BLOCKED":
        names = ", ".join(r["name"] for r in outcome["blocked"])
        st.error(f"🔴 **CRITICAL BOUNDARY DEFECT — COMPLIANCE STATUS: BLOCKED**\n\nBreaching component(s): {names}")
    elif gs.startswith("CONDITION NOT FULLY VERIFIED"):
        st.warning(f"🟡 **COMPLIANCE STATUS: {gs}**")
    else:
        st.success("🟢 **COMPLIANCE STATUS: VERIFIED SECURE**")

    st.subheader("🧩 Component-Level Results")
    for result in outcome["results"]:
        render_component_card(result)

    anomalies = extracted.get("field_anomalies") or []
    if anomalies:
        st.markdown("### ⚠️ Unresolved Mechanical Anomaly Logs")
        for note in anomalies:
            st.markdown(f"<div class='anomaly-box'>{note}</div>", unsafe_allow_html=True)

    with st.expander("Raw structured response from the model"):
        st.json(extracted)

    with st.expander("Reference standard definitions"):
        st.markdown(
            "- **ASME B31.3** — Process Piping code; defines the pressure-design wall-thickness "
            "formula used as the fallback calculation when a piping component has no stated limit.\n"
            "- **API 579-1/ASME FFS-1** — Fitness-for-Service; used to assess whether a "
            "locally thinned component can remain in service and under what conditions.\n"
            "- **API 653** — In-service inspection, repair, and reconstruction of atmospheric "
            "storage tanks.\n"
            "- **Minimum Allowable Thickness (MAT)** — The lowest thickness at which a "
            "component still meets its design-basis strength/pressure requirement; readings "
            "below this value indicate a breach requiring engineering disposition."
        )

else:
    st.info("Upload a .txt inspection report above to run extraction.")
