"""
fixit — Real AI Mode (Deep Multi-Component Schema)
-----------------------------------------------------
Single-file Streamlit application.

Structured extraction is performed by the Anthropic API using forced
tool-use against a nested, multi-component schema: each independent
sub-component of an inspected asset (a column, a base plate, a gusset,
a flange, a process tube, ...) gets its own object with its own stated
minimum thickness, its own list of current UT readings, its own list of
prior-inspection UT readings (for trend monitoring), plus an
auto-detected engineering framework/code jurisdiction and a
model-reported extraction confidence score. A single asset-level
pressure event (observed peak vs. design pressure) is captured when the
source text states one.

What stays in plain, deterministic Python, and why:
  * The pass/fail decision for each component (margin = lowest current
    UT reading for that component - that component's stated minimum,
    with an ASME B31.3 fallback used ONLY when the asset is process
    piping and a component has no stated minimum — never for structural
    steel/non-pressurized assets) is computed in plain Python, never by
    the model, so the safety-critical comparison is reproducible.
  * The historical degradation delta (previous lowest reading vs.
    current lowest reading) and the pressure-excursion variance
    (observed vs. design pressure) are both computed as plain
    arithmetic in Python, never asserted by the model.
  * A lightweight internal validator pass runs over the model's JSON
    output (not over the raw text with regex) as a sanity net before
    anything is rendered. This is best-effort, not a guarantee of
    perfect extraction — no automated pass can promise 100% accuracy
    against arbitrary messy field text, so findings are surfaced as
    review flags, not treated as ground truth.
  * The engineering disposition is never auto-prescribed as a single
    fixed outcome. A confirmed exceedance (a mathematical fact — margin
    below zero) is always rendered separately from the recommended next
    step, which is labeled as "Awaiting Authorized Engineering/Inspector
    Review" rather than a final repair order. Unconfirmed findings
    (e.g. a "possible" weld indication) are always kept as open,
    unconfirmed items requiring secondary NDT validation, never
    upgraded to a confirmed defect.

Model note: MODEL_NAME stays "claude-opus-5-5" per your explicit
instruction to keep it and disregard the request to switch to
"claude-3-5-sonnet-latest". Say so if you ever want that changed.

Title note: the "fixit" wordmark below is a <span>, not an <h1> — this
avoids Streamlit's automatic anchor-link injection on native heading
elements (which was overlapping the glass title) while keeping the
same pure-CSS glass fill + hover shimmer. No <canvas>, no JS, no
particle/dot tracking of any kind is used for the logo itself. The one
remaining bit of JS in this file is the cursor-following glow on the
result cards lower down, which is unrelated to the logo; it's wrapped
in try/except and degrades silently if unavailable.
"""

import sys
import subprocess

try:
    import anthropic
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "anthropic>=0.34.0"])
    import anthropic

import os
import re
import json

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="fixit",
    layout="wide",
    page_icon="🛠️",
)

# ============================================================================
# GLOBAL STYLE SHEET + CENTER-TOP GLASS TITLE (pure CSS, no canvas/JS)
# ============================================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 15% 20%, rgba(54,15,37,0.55), transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(31,20,53,0.6), transparent 50%),
            radial-gradient(circle at 50% 90%, rgba(31,20,53,0.35), transparent 55%),
            #0A0E1A;
        background-attachment: fixed;
    }

    .glass-title-container {
        margin: 0 auto;
        text-align: center;
        display: block;
        width: 100%;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .glass-title {
        position: relative;
        display: inline-block;
        font-size: 5rem;
        font-weight: 800;
        font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
        color: rgba(255, 255, 255, 0.12);
        -webkit-text-stroke: 1px rgba(255, 255, 255, 0.28);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        text-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.3);
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        margin: 0;
        transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1), text-shadow 0.5s cubic-bezier(0.16, 1, 0.3, 1);
    }
    /* Sheen-sweep layer: a second copy of the letters, clipped to a gradient
       that slides from off-screen-left to off-screen-right on hover. No
       canvas, no per-pixel anything — just a clipped background on text. */
    .glass-title::before {
        content: attr(data-text);
        position: absolute;
        left: 0; top: 0; width: 100%; height: 100%;
        background: linear-gradient(120deg, transparent 35%, rgba(255,255,255,0.9) 50%, transparent 65%);
        background-size: 250% 100%;
        background-position: -250% 0;
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        -webkit-text-stroke: 1px transparent;
        pointer-events: none;
        transition: background-position 1.1s ease;
    }
    .glass-title-container:hover .glass-title {
        transform: scale(1.03);
        text-shadow: 0 0 35px rgba(251, 191, 36, 0.35);
    }
    .glass-title-container:hover .glass-title::before {
        background-position: 250% 0;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px) saturate(180%);
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        background-repeat: no-repeat;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 242, 254, 0.45);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37), 0 0 26px rgba(0, 242, 254, 0.18);
    }
    .metric-label { color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { color: #F9FAFB; font-size: 1.2rem; font-weight: 700; word-wrap: break-word; }
    .accent-cyan { color: #00F2FE; }
    .accent-gold { color: #FBBF24; }

    .component-card.blocked {
        background: rgba(69, 10, 10, 0.45) !important;
        border: 1px solid #EF4444 !important;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2) !important;
    }
    .component-card.verified { border-color: rgba(22,163,74,0.5); }
    .component-card.unresolved { border-color: rgba(107,114,128,0.5); }
    /* Aero-glow status light: pulses on hover, matching pass/fail disposition. */
    .component-card.verified:hover {
        animation: breathe-emerald-hover 1.4s ease-in-out infinite;
        border-color: rgba(16,185,129,0.6);
    }
    .component-card.blocked:hover {
        animation: breathe-crimson-hover 1.2s ease-in-out infinite;
    }
    @keyframes breathe-emerald-hover {
        0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.25); }
        50% { box-shadow: 0 0 40px rgba(16,185,129,0.45); }
    }
    @keyframes breathe-crimson-hover {
        0%, 100% { box-shadow: 0 0 20px rgba(239,68,68,0.3); }
        50% { box-shadow: 0 0 45px rgba(239,68,68,0.55); }
    }

    .status-pill {
        display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.7rem;
        font-weight:700; letter-spacing:.03em; text-transform:uppercase; margin-left:8px;
    }
    .status-pill.blocked { background:#DC2626; color:#FEF2F2; }
    .status-pill.verified { background:#16A34A; color:#F0FDF4; }
    .status-pill.unresolved { background:#6B7280; color:#F9FAFB; }
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(25px) saturate(180%) !important;
    }
    .raw-terminal {
        background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
        padding: 14px; color: #9CA3AF; font-family: monospace; font-size: 0.8rem;
        max-height: 640px; overflow-y: auto; white-space: pre-wrap;
        backdrop-filter: blur(25px) saturate(180%);
    }

    /* Pressure-excursion operational alert: crimson-amber, sits at the very
       top of the matrix panel so it can never be buried in footnotes. */
    .pressure-alert-banner {
        background: linear-gradient(120deg, rgba(127,29,29,0.55), rgba(120,53,15,0.55));
        border: 1px solid #F59E0B;
        border-radius: 16px;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.25);
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .pressure-alert-banner .headline {
        color: #FEF3C7; font-size: 1.0rem; font-weight: 800; letter-spacing: .02em;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .pressure-alert-banner .body-text { color: #FCE7C3; font-size: 0.92rem; }

    /* Degradation-anomaly card: gold warning, for multi-inspection trend loss. */
    .degradation-card {
        background: rgba(120, 95, 15, 0.28);
        border: 1px solid #FBBF24;
        border-radius: 16px;
        box-shadow: 0 0 24px rgba(251, 191, 36, 0.2);
        padding: 14px 18px;
        margin: 10px 0 14px 0;
    }
    .degradation-card .headline {
        color: #FDE68A; font-weight: 800; font-size: 0.85rem; letter-spacing: .02em;
        text-transform: uppercase; margin-bottom: 4px;
    }
    .degradation-card .body-text { color: #FEF3C7; font-size: 0.9rem; }

    /* Global status ledger: replaces a bare "BLOCKED" pill with a readable
       diagnostic sentence naming the exact contributing risk entries. */
    .status-ledger-banner {
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 16px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .status-ledger-banner.blocked {
        background: rgba(69, 10, 10, 0.5);
        border: 1px solid #EF4444;
        color: #FEE2E2;
        box-shadow: 0 0 26px rgba(239, 68, 68, 0.2);
    }
    .status-ledger-banner.unverified {
        background: rgba(69, 50, 10, 0.4);
        border: 1px solid #9CA3AF;
        color: #F3F4F6;
    }
    .status-ledger-banner.secure {
        background: rgba(6, 55, 30, 0.4);
        border: 1px solid #16A34A;
        color: #DCFCE7;
    }
    .status-ledger-banner b { color: inherit; }

    /* Confirmed-exceedance / disposition separation */
    .fact-block {
        border-left: 3px solid #EF4444;
        padding: 6px 0 6px 12px;
        margin: 10px 0;
    }
    .fact-block .fact-label {
        color: #F87171; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em;
    }
    .disposition-block {
        border-left: 3px solid #9CA3AF;
        padding: 6px 0 6px 12px;
        margin: 10px 0;
    }
    .disposition-block .disposition-label {
        color: #D1D5DB; font-size: 0.7rem; font-weight: 800; text-transform: uppercase; letter-spacing: .04em;
    }
    </style>

    <div class="glass-title-container">
        <span class="glass-title" data-text="fixit">fixit</span>
    </div>
    """,
    unsafe_allow_html=True,
)

MODEL_NAME = "claude-opus-5-5"  # kept per explicit instruction — do not switch to a retired alias

# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


SYSTEM_PROMPT = """You are a structured-data extraction engine for industrial \
engineering inspection field logs covering multi-component assets: structural \
steel assemblies (columns, base plates, gussets, beams, braces), pressure \
vessels, process piping (including tube bundles), heat exchangers, storage \
tanks, and conveyor frames.

Read the raw inspection text the user provides and call the \
`extract_compliance_data` tool with the data you find. Rules:
- Extract only what is actually stated in the text. Never invent, estimate, \
  or "helpfully" fill in a value that is not present — use null for anything \
  not stated, and list it in missing_engineering_variables_ledger if it is \
  needed for a downstream calculation.
- Treat the asset as a collection of independent sub-components. Each \
  distinct structural or mechanical element named or clearly implied in the \
  text (e.g. "North Column", "Base Plate", "Gusset G1", "Beam Flange", \
  "Process Tubes") gets its own entry in components_matrix, with its own \
  explicit_minimum_required_mat and its own ut_thickness_measurements list. \
  Do not merge readings from different components into one entry, and do \
  not invent components that are not named or clearly implied. If a \
  component is named but the text gives it neither a stated minimum nor any \
  reading, still create its entry with both left empty/null — do not omit it.
- "historical_ut_thickness_measurements" holds readings the text attributes \
  to a prior inspection (a previous date, a "last inspection", a baseline) \
  for that same component, in the same {location_label, value, unit} shape \
  as ut_thickness_measurements. Leave the list empty if the text gives no \
  prior-inspection reading for that component. Never treat a current-cycle \
  reading as historical, and never treat a historical reading as current.
- "engineering_framework" is the applicable code/jurisdiction if the text \
  states or clearly implies one (e.g. "ASME Sec VIII", "ASME B31.3", \
  "AWS D1.1", "API 653"); null if not determinable.
- "extraction_confidence_score" is your own 0.0-1.0 estimate of how legible \
  and unambiguous the source text was for this extraction — not a measure \
  of the asset's physical condition.
- "explicit_minimum_required_mat" is whatever the report calls that \
  component's minimum/allowable/permitted thickness limit. If it is not \
  explicitly stated for that component, leave it null even if you could \
  calculate one yourself — calculation is handled outside this tool.
- "piping_design_variables" (design pressure, outside diameter, allowable \
  stress, quality/joint factor, Y coefficient, corrosion allowance) apply at \
  the asset level and are only relevant for process piping. Extract each \
  only if explicitly present. Be careful not to confuse the allowable stress \
  value (typically a large number, e.g. in the thousands, in psi or MPa) \
  with an adjacent small decimal thickness reading — they are different \
  quantities even when they appear near each other in the text.
- "pressure_event" captures a single logged operating-pressure excursion at \
  the asset level: the observed/peak pressure reading and the stated design \
  pressure threshold, only if the text explicitly gives both. Leave both \
  null if either is not explicitly stated. Do not confuse this with the \
  B31.3 design_pressure used for wall-thickness calculation — record both \
  fields independently even if they share the same numeric value.
- Preserve stated uncertainty rather than resolving it. A "possible" or \
  "suspected" weld indication must be recorded as an unconfirmed finding \
  (is_confirmed_failure: false) describing it as needing NDT validation — \
  never upgraded to a confirmed defect. Unverified/untorqued fasteners must \
  likewise be recorded as unconfirmed and described as pending verification, \
  not as a confirmed connection failure. Only set is_confirmed_failure: true \
  when the text itself states the item failed, is rejected, or is out of \
  tolerance.
- "field_anomalies" should capture every inspector note, flagged indication, \
  weld observation, fastener condition note, or geometry/alignment issue, in \
  the report's own words in the "notes" field.
- Do not comment on overall compliance, pass/fail, or safety — only extract \
  data as stated."""

EXTRACTION_TOOL = {
    "name": "extract_compliance_data",
    "description": "Record structured, multi-component fields extracted from an industrial inspection field log.",
    "input_schema": {
        "type": "object",
        "properties": {
            "asset_category": {"type": ["string", "null"]},
            "metallurgy_specification": {
                "type": ["string", "null"],
                "description": "Material/metallurgy specification, mapped to sub-parts where the text distinguishes them.",
            },
            "engineering_framework": {
                "type": ["string", "null"],
                "description": "Auto-discovered applicable code/jurisdiction, e.g. ASME Sec VIII, ASME B31.3, AWS D1.1, API 653.",
            },
            "extraction_confidence_score": {
                "type": ["number", "null"],
                "description": "0.0-1.0 self-estimate of extraction legibility/confidence.",
            },
            "piping_design_variables": {
                "type": ["object", "null"],
                "description": "ASME B31.3 pipe-wall inputs, only if the asset is process piping and these are present in the text.",
                "properties": {
                    "design_pressure": {"type": ["number", "null"]},
                    "outside_diameter": {"type": ["number", "null"]},
                    "allowable_stress": {"type": ["number", "null"]},
                    "quality_factor": {"type": ["number", "null"]},
                    "y_coefficient": {"type": ["number", "null"]},
                    "corrosion_allowance": {"type": ["number", "null"]},
                },
            },
            "pressure_event": {
                "type": ["object", "null"],
                "description": "A single logged operating-pressure excursion at the asset level, only if both values are explicitly stated.",
                "properties": {
                    "observed_pressure": {"type": ["number", "null"]},
                    "design_pressure_threshold": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                },
            },
            "components_matrix": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "component_name": {"type": "string"},
                        "nominal_thickness": {
                            "type": ["object", "null"],
                            "properties": {"value": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]}},
                        },
                        "explicit_minimum_required_mat": {
                            "type": ["object", "null"],
                            "properties": {"value": {"type": ["number", "null"]}, "unit": {"type": ["string", "null"]}},
                        },
                        "ut_thickness_measurements": {
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
                        "historical_ut_thickness_measurements": {
                            "type": "array",
                            "description": "Prior-inspection readings for this same component, if the text gives any.",
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
                    "required": ["component_name", "ut_thickness_measurements"],
                },
            },
            "missing_engineering_variables_ledger": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required parameters missing from the source text, with why they matter.",
            },
            "field_anomalies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "finding": {"type": "string"},
                        "is_confirmed_failure": {"type": "boolean"},
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": ["finding", "is_confirmed_failure"],
                },
            },
        },
        "required": ["asset_category", "components_matrix", "field_anomalies"],
    },
}


def extract_with_claude(client, report_text):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=3500,
        system=SYSTEM_PROMPT,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": report_text}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_compliance_data":
            return dict(block.input)
    raise RuntimeError("Model did not return a structured extraction — no tool_use block found.")

# ============================================================================
# DETERMINISTIC EVALUATION
# ============================================================================

B31_3_VARS = [
    ("design_pressure", "Design Pressure"),
    ("outside_diameter", "Outside Diameter"),
    ("allowable_stress", "Allowable Stress"),
    ("quality_factor", "Quality/Joint Factor"),
    ("y_coefficient", "Y Coefficient"),
    ("corrosion_allowance", "Corrosion Allowance"),
]


def calc_b31_3_mat(piping_vars):
    dv = piping_vars or {}
    missing = [label for key, label in B31_3_VARS if dv.get(key) is None]
    if missing:
        return None, missing, None
    P, D, S, E, Y, CA = (dv["design_pressure"], dv["outside_diameter"], dv["allowable_stress"],
                          dv["quality_factor"], dv["y_coefficient"], dv["corrosion_allowance"])
    t_design = (P * D) / (2 * (S * E + P * Y))
    mat = t_design + CA
    note = (f"No explicit minimum was stated for this component, so the ASME B31.3 straight-pipe "
            f"formula was applied: t_design = (P×D)/(2×(S×E+P×Y)) = {t_design:.4f}. Adding the "
            f"corrosion allowance ({CA:.3f}) gives a calculated safety ceiling of {mat:.4f}.")
    return mat, [], note


def compute_degradation(component):
    """Plain-Python trend check: compares the lowest historical reading to
    the lowest current reading for this component. Returns None if there is
    nothing to compare, or a dict describing the delta (only ever flagged as
    an anomaly when thickness has genuinely decreased)."""
    current = component.get("ut_thickness_measurements") or []
    historical = component.get("historical_ut_thickness_measurements") or []
    if not current or not historical:
        return None

    lowest_current = min(current, key=lambda r: r["value"])
    lowest_historical = min(historical, key=lambda r: r["value"])
    delta = round(lowest_historical["value"] - lowest_current["value"], 4)

    return {
        "component_name": component.get("component_name") or "Unnamed Component",
        "previous_value": lowest_historical["value"],
        "current_value": lowest_current["value"],
        "unit": lowest_current.get("unit") or lowest_historical.get("unit") or "",
        "delta": delta,
        "is_loss": delta > 0,  # positive delta = thickness went down over time
    }


def evaluate_component(component, is_piping, piping_vars):
    """is_piping gates the B31.3 fallback so it never fires for structural steel
    or other non-pressurized assets."""
    name = component.get("component_name") or "Unnamed Component"
    ut_list = component.get("ut_thickness_measurements") or []
    mat_field = component.get("explicit_minimum_required_mat")
    mat = mat_field.get("value") if mat_field else None
    mat_unit = mat_field.get("unit") if mat_field else None
    degradation = compute_degradation(component)

    if not ut_list:
        # Distinguish "we have a stated minimum but nothing to test it against"
        # from "we have neither criteria nor measurements at all" — the two
        # are engineering-critical to tell apart in the UI.
        status = "no_measurements" if mat is not None else "no_criteria_no_measurements"
        return {"name": name, "status": status, "mat": mat, "mat_unit": mat_unit,
                 "degradation": degradation, "raw": component}

    calc_note = None
    if mat is None:
        if is_piping:
            mat, missing, calc_note = calc_b31_3_mat(piping_vars)
            if mat is None:
                return {"name": name, "status": "insufficient", "missing_vars": missing,
                         "ut_measurements": ut_list, "degradation": degradation, "raw": component}
        else:
            return {"name": name, "status": "insufficient",
                     "missing_vars": ["Explicit Minimum Required MAT (not stated for this component)"],
                     "ut_measurements": ut_list, "degradation": degradation, "raw": component}

    lowest = min(ut_list, key=lambda r: r["value"])
    margin = round(lowest["value"] - mat, 4)
    status = "blocked" if margin < 0 else "verified"
    return {"name": name, "status": status, "mat": mat, "mat_unit": mat_unit, "lowest": lowest,
             "margin": margin, "calc_note": calc_note, "degradation": degradation, "raw": component}


def evaluate_pressure_event(pressure_event):
    """Plain-Python variance calc for a logged overpressure excursion.
    Returns None unless both values are explicitly present and the observed
    reading actually exceeds the design threshold."""
    if not pressure_event:
        return None
    observed = pressure_event.get("observed_pressure")
    design = pressure_event.get("design_pressure_threshold")
    if observed is None or design is None or design == 0:
        return None
    if observed <= design:
        return None
    variance_abs = round(observed - design, 4)
    variance_pct = round((variance_abs / design) * 100, 1)
    unit = pressure_event.get("unit") or "psi"
    return {
        "observed": observed, "design": design, "unit": unit,
        "variance_abs": variance_abs, "variance_pct": variance_pct,
    }


def evaluate(extracted):
    components = extracted.get("components_matrix") or []
    asset_category = (extracted.get("asset_category") or "").lower()
    is_piping = "pip" in asset_category  # covers "piping" / "pipeline" / "process pipe"
    piping_vars = extracted.get("piping_design_variables")

    results = [evaluate_component(c, is_piping, piping_vars) for c in components]
    blocked = [r for r in results if r["status"] == "blocked"]
    unresolved = [r for r in results if r["status"] in ("insufficient", "no_measurements", "no_criteria_no_measurements")]
    degradations = [r["degradation"] for r in results if r.get("degradation") and r["degradation"]["is_loss"]]
    confirmed_failures = [a for a in (extracted.get("field_anomalies") or []) if a.get("is_confirmed_failure")]
    unconfirmed_findings = [a for a in (extracted.get("field_anomalies") or []) if not a.get("is_confirmed_failure")]
    ledger = extracted.get("missing_engineering_variables_ledger") or []
    pressure_excursion = evaluate_pressure_event(extracted.get("pressure_event"))

    if not results:
        global_status = "CONDITION UNVERIFIED"
    elif blocked or confirmed_failures:
        global_status = "BLOCKED"
    elif unresolved or ledger:
        global_status = "CONDITION UNVERIFIED"
    else:
        global_status = "VERIFIED SECURE"

    ledger_message = build_status_ledger_message(
        global_status, blocked, unresolved, pressure_excursion, unconfirmed_findings
    )

    return {"results": results, "blocked": blocked, "unresolved": unresolved,
             "confirmed_failures": confirmed_failures, "unconfirmed_findings": unconfirmed_findings,
             "degradations": degradations, "pressure_excursion": pressure_excursion,
             "global_status": global_status, "ledger_message": ledger_message}


def build_status_ledger_message(global_status, blocked, unresolved, pressure_excursion, unconfirmed_findings):
    """Builds a context-specific diagnostic sentence naming the exact
    parameters driving the restriction, instead of a bare status word."""
    if global_status == "VERIFIED SECURE":
        return "VERIFIED SECURE: All mapped components carry a stated criteria set and a current reading at or above minimum. No confirmed exceedances on file."

    risk_entries = []
    lead = None

    if blocked:
        worst = min(blocked, key=lambda r: r["margin"])
        lead = f"{worst['name']} thickness drops below the stated minimum threshold"
        for r in blocked:
            unit = f" {r['mat_unit']}" if r.get("mat_unit") else ""
            risk_entries.append(f"{r['name']} margin {r['margin']:+.4f}{unit} below minimum")

    if pressure_excursion:
        risk_entries.append(
            f"{pressure_excursion['observed']:g} {pressure_excursion['unit']} overpressure excursion "
            f"(+{pressure_excursion['variance_abs']:g} {pressure_excursion['unit']} / "
            f"+{pressure_excursion['variance_pct']:g}%)"
        )

    for r in unresolved:
        if r["status"] == "no_criteria_no_measurements":
            risk_entries.append(f"unverified {r['name'].lower()} — no criteria or readings on file")
        elif r["status"] == "no_measurements":
            risk_entries.append(f"unverified {r['name'].lower()} — criteria on file but untested")
        elif r["status"] == "insufficient":
            risk_entries.append(f"{r['name'].lower()} margin uncalculable — missing engineering variables")

    for a in unconfirmed_findings:
        risk_entries.append(f"unconfirmed finding pending NDT validation ({a.get('finding', 'unspecified')})")

    if not risk_entries:
        return f"{global_status}: No mapped components resolved to a confirmed status; awaiting further data."

    if lead is None:
        lead = f"{len(risk_entries)} unresolved risk item(s) on file"

    entries_text = "; ".join(risk_entries)
    return f"{global_status}: {lead}. Unresolved risk entries include: {entries_text}."


def remediation_steps(component_name):
    n = component_name.lower()
    if "column" in n:
        return ["Route to a structural engineer for a reduced-section load rating before any load is reapplied.",
                 "Cross-check against adjacent column readings to rule out a localized corrosion cell.",
                 "Consider a doubler plate or full section replacement per the engineer's disposition."]
    if "plate" in n:
        return ["Check anchor bolt torque and grout condition — thinning here often co-occurs with bolt/grout issues.",
                 "Confirm bearing area is still adequate for the design load at the reduced thickness.",
                 "Schedule replacement or plate doubling per Authorized Inspector review."]
    if "gusset" in n:
        return ["Evaluate remaining section against the connection's design shear/moment capacity.",
                 "Inspect welds at the gusset-to-member interface for cracking driven by reduced stiffness.",
                 "Do not defer — gussets are frequently the limiting element in a braced connection."]
    if "flange" in n:
        return ["Re-torque per the flange's bolt pattern and verify gasket seating before returning to service.",
                 "Check for leakage/weeping at the reduced-thickness zone under normal operating pressure.",
                 "Route to piping engineering for a B31.3 reassessment of this joint."]
    if "tube" in n:
        return ["Do not assume acceptability from thickness alone — schedule a wall-thickness examination (UT/eddy current) for this bundle.",
                 "Confirm the design acceptance criteria for this tube bundle with piping/mechanical engineering before disposition.",
                 "Flag for Authorized Inspector review pending both criteria and readings being established."]
    if "shell" in n or "head" in n:
        return ["Perform a local Fitness-for-Service assessment (e.g. API 579 Level 1/2) before continued operation.",
                 "Grid the surrounding area to bound the extent of the thin region.",
                 "Consider a pressure de-rate as an interim measure pending repair."]
    return ["Route this component to Fitness-for-Service / Authorized Inspector review.",
             "Bound the extent of the thin area with supplemental UT grid readings.",
             "Do not return the component to unrestricted service until disposition is issued."]

# ============================================================================
# INTERNAL VALIDATOR PASS (sanity net over the model's JSON — best-effort, not a guarantee)
# ============================================================================

def audit_extraction(report_text, extracted):
    notices = []
    components = extracted.get("components_matrix") or []

    numeric_tokens = re.findall(r"\b\d+\.\d+\b", report_text)
    total_readings = sum(len(c.get("ut_thickness_measurements") or []) for c in components)
    if numeric_tokens and total_readings < max(1, int(len(numeric_tokens) * 0.5)):
        notices.append(
            f"The source text contains {len(numeric_tokens)} decimal values, but only "
            f"{total_readings} were mapped into components — some readings may not have been assigned."
        )

    keywords = ["column", "plate", "gusset", "flange", "shell", "head", "nozzle", "beam", "brace", "support", "tube"]
    text_lower = report_text.lower()
    matrix_text = " ".join(c.get("component_name", "").lower() for c in components)
    for kw in keywords:
        if kw in text_lower and kw not in matrix_text:
            notices.append(
                f"The source text mentions '{kw}' but no extracted component name contains it — "
                f"confirm nothing was dropped."
            )

    for c in components:
        for r in c.get("ut_thickness_measurements") or []:
            if not r.get("unit"):
                notices.append(
                    f"Reading '{r.get('location_label')}' on '{c.get('component_name')}' has no unit "
                    f"recorded — confirm it wasn't lost during extraction."
                )

    return notices

# ============================================================================
# CURSOR-FOLLOWING GLOW ON RESULT CARDS (best-effort; unrelated to the title)
# ============================================================================

components.html(
    """
    <script>
    (function () {
        try {
            const doc = window.parent.document;
            doc.addEventListener('mousemove', function (e) {
                const cards = doc.querySelectorAll('.glass-card');
                cards.forEach(function (card) {
                    const rect = card.getBoundingClientRect();
                    const x = e.clientX - rect.left;
                    const y = e.clientY - rect.top;
                    if (x >= -60 && x <= rect.width + 60 && y >= -60 && y <= rect.height + 60) {
                        card.style.backgroundImage =
                            'radial-gradient(circle at ' + x + 'px ' + y + 'px, rgba(0,242,254,0.14), transparent 55%)';
                    } else {
                        card.style.backgroundImage = 'none';
                    }
                });
            });
        } catch (err) {
            // Same-origin parent-document access isn't guaranteed across all
            // Streamlit deployments/versions — fail silently.
        }
    })();
    </script>
    """,
    height=0,
)


def metric_html(label, value, accent=None):
    cls = f"metric-value {accent}" if accent else "metric-value"
    return f"<div class='glass-card'><div class='metric-label'>{label}</div><div class='{cls}'>{value}</div></div>"


def render_pressure_alert_banner(pressure_excursion):
    st.markdown(
        f"<div class='pressure-alert-banner'>"
        f"<div class='headline'>⚠ Pressure Excursion Detected</div>"
        f"<div class='body-text'>Observed peak of {pressure_excursion['observed']:g} "
        f"{pressure_excursion['unit']} against a stated design pressure threshold of "
        f"{pressure_excursion['design']:g} {pressure_excursion['unit']}. Threshold variance: "
        f"+{pressure_excursion['variance_abs']:g} {pressure_excursion['unit']} / "
        f"+{pressure_excursion['variance_pct']:g}% overload. Flagged for immediate operational review — "
        f"do not defer to a footnote.</div></div>",
        unsafe_allow_html=True,
    )


def render_degradation_card(deg):
    st.markdown(
        f"<div class='degradation-card'>"
        f"<div class='headline'>Degradation Anomaly Captured</div>"
        f"<div class='body-text'>{deg['component_name']}: prior lowest reading "
        f"{deg['previous_value']:.4f} {deg['unit']} now measures {deg['current_value']:.4f} {deg['unit']} — "
        f"an observed reduction of -{deg['delta']:.4f} {deg['unit']}. Flagged for immediate corrosion-rate "
        f"and remaining-life evaluation loops.</div></div>",
        unsafe_allow_html=True,
    )


def render_status_ledger_banner(outcome):
    gs = outcome["global_status"]
    css_class = "blocked" if gs == "BLOCKED" else ("secure" if gs == "VERIFIED SECURE" else "unverified")
    st.markdown(
        f"<div class='status-ledger-banner {css_class}'><b>{outcome['ledger_message']}</b></div>",
        unsafe_allow_html=True,
    )


def render_component_card(result):
    status = result["status"]
    css_class = "blocked" if status == "blocked" else ("verified" if status == "verified" else "unresolved")
    pill_label = {
        "blocked": "Blocked",
        "verified": "Verified",
        "insufficient": "Unresolved",
        "no_measurements": "No Data",
        "no_criteria_no_measurements": "Unverified",
    }[status]

    st.markdown(
        f"<div class='glass-card component-card {css_class}'>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#F9FAFB;'>{result['name']}"
        f"<span class='status-pill {css_class}'>{pill_label}</span></div>",
        unsafe_allow_html=True,
    )

    # Multi-inspection historical trend — surfaced for every component that has
    # a comparable prior reading, regardless of current pass/fail status.
    deg = result.get("degradation")
    if deg and deg["is_loss"]:
        render_degradation_card(deg)

    if status in ("blocked", "verified"):
        unit_suffix = f" {result['mat_unit']}" if result.get("mat_unit") else ""
        lowest = result["lowest"]
        lowest_unit = f" {lowest['unit']}" if lowest.get("unit") else ""
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_html("Measured Minimum UT", f"{lowest['location_label']}: {lowest['value']:.4f}{lowest_unit}"), unsafe_allow_html=True)
        c2.markdown(metric_html("Required Minimum MAT", f"{result['mat']:.4f}{unit_suffix}"), unsafe_allow_html=True)
        margin_accent = "accent-gold" if status == "blocked" else "accent-cyan"
        c3.markdown(metric_html("Computed True Margin", f"{result['margin']:+.4f}", margin_accent), unsafe_allow_html=True)
        if result.get("calc_note"):
            st.info(result["calc_note"])

        if status == "blocked":
            # Confirmed fact (the math) kept visually and structurally separate
            # from the recommended pathway (which is not a final disposition).
            st.markdown(
                f"<div class='fact-block'><div class='fact-label'>Confirmed Exceedance — Mathematical Fact</div>"
                f"Measured minimum ({lowest['value']:.4f}{lowest_unit}) is below the required minimum MAT "
                f"({result['mat']:.4f}{unit_suffix}) by {abs(result['margin']):.4f}.</div>",
                unsafe_allow_html=True,
            )
            steps_html = "".join(f"<li>{s}</li>" for s in remediation_steps(result["name"]))
            st.markdown(
                f"<div class='disposition-block'><div class='disposition-label'>Recommended Engineering Pathway — "
                f"Awaiting Authorized Engineering/Inspector Review</div><ul>{steps_html}</ul>"
                f"<div style='color:#9CA3AF;font-size:0.8rem;'>This is a suggested exploration path, not a final "
                f"disposition — the confirmed exceedance above stands independent of whichever pathway is ultimately "
                f"authorized.</div></div>",
                unsafe_allow_html=True,
            )
    elif status == "insufficient":
        st.write("Cannot compute a margin for this component — missing:")
        for m in result["missing_vars"]:
            st.markdown(f"- ❌ **{m}**")
    elif status == "no_measurements":
        st.write("A minimum is on file for this component, but no UT readings were extracted for it.")
    else:  # no_criteria_no_measurements
        st.warning(
            "No design acceptance criteria or wall-thickness examination metrics provided; "
            "structural condition is unverified."
        )

    with st.expander("View Raw Source Extraction Line"):
        st.json(result["raw"])

    st.markdown("</div>", unsafe_allow_html=True)


client = get_client()
if client is None:
    st.error(
        "ANTHROPIC_API_KEY is not set. In Streamlit Cloud, open **Settings → Secrets** for this app "
        "and add:\n\n```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
    )
    st.stop()

uploaded = st.file_uploader("Drop an inspection field log (.txt)", type=["txt"])

if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    report_text = raw_bytes.decode("utf-8", errors="ignore")
    cache_key = f"{uploaded.name}:{len(raw_bytes)}:{hash(raw_bytes)}"

    if st.session_state.get("cache_key") != cache_key:
        with st.spinner(f"Checking this amazing file"):
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
    audit_notices = audit_extraction(report_text, extracted)

    score = extracted.get("extraction_confidence_score")
    if score is None:
        confidence_label = "Unknown"
    elif score >= 0.85:
        confidence_label = "High"
    elif score >= 0.6:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Pressure-excursion alert always sits at the very top of the matrix panel.
    if outcome["pressure_excursion"]:
        render_pressure_alert_banner(outcome["pressure_excursion"])

    st.markdown("#### Global Engineering Status Ledger")
    render_status_ledger_banner(outcome)

    st.markdown("#### Dual Status Banners")
    m1, m2 = st.columns(2)
    m1.markdown(metric_html("Extraction Confidence (text legibility)", confidence_label, "accent-cyan"), unsafe_allow_html=True)
    gs = outcome["global_status"]
    gs_accent = "accent-gold" if gs == "BLOCKED" else None
    m2.markdown(metric_html("Global Engineering Safety Status", gs, gs_accent), unsafe_allow_html=True)

    if audit_notices:
        st.markdown("#### 🟡 System Extraction Audit Notice")
        for notice in audit_notices:
            st.warning(notice)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("📡 Ingested Raw Inspection File Stream")
        st.markdown(f"<div class='raw-terminal'>{report_text}</div>", unsafe_allow_html=True)

    with right:
        st.subheader("🧬 Live Structural Verification Matrix")
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_html("Asset Category", extracted.get("asset_category") or "Not stated"), unsafe_allow_html=True)
        c2.markdown(metric_html("Metallurgy", extracted.get("metallurgy_specification") or "Not stated"), unsafe_allow_html=True)
        c3.markdown(metric_html("Engineering Framework", extracted.get("engineering_framework") or "Not determined"), unsafe_allow_html=True)

        st.markdown("#### Calculation Trail Ledger")
        st.caption("Component Name → Measured Minimum UT → Required Minimum MAT → Computed True Margin → Status → Recommended Engineering Pathway")
        for result in outcome["results"]:
            render_component_card(result)

        ledger = extracted.get("missing_engineering_variables_ledger") or []
        if ledger:
            st.markdown("#### 📒 Data Deficit Ledger")
            for item in ledger:
                st.markdown(f"- {item}")

        anomalies = extracted.get("field_anomalies") or []
        if anomalies:
            st.markdown("#### ⚠️ Field Anomalies")
            for a in anomalies:
                tag = "Confirmed Failure" if a.get("is_confirmed_failure") else "Unresolved — Requires Validation"
                st.markdown(
                    f"<div class='glass-card'><b>{a.get('finding')}</b> "
                    f"<span class='status-pill {'blocked' if a.get('is_confirmed_failure') else 'unresolved'}'>{tag}</span>"
                    f"<div style='color:#D1D5DB;margin-top:6px;'>{a.get('notes') or ''}</div></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Raw structured response from the model"):
        st.json(extracted)

    with st.expander("Reference standard definitions"):
        st.markdown(
            "- **ASME B31.3** — Process Piping code; source of the pressure-design wall-thickness "
            "fallback formula used when a piping component has no stated limit.\n"
            "- **ASME Section VIII** — Rules for construction of pressure vessels.\n"
            "- **AWS D1.1** — Structural Welding Code (Steel).\n"
            "- **API 653** — In-service inspection, repair, and reconstruction of atmospheric storage tanks.\n"
            "- **API 579-1/ASME FFS-1** — Fitness-for-Service; used to assess whether a locally "
            "thinned component can remain in service and under what conditions.\n"
            "- **Minimum Allowable Thickness (MAT)** — The lowest thickness at which a component "
            "still meets its design-basis strength/pressure requirement."
        )

else:
    st.info("Upload a .txt inspection field log above to run extraction.")
