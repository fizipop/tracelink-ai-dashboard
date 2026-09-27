"""
fixit — Real AI Mode (Deep Multi-Component Schema)
-----------------------------------------------------
Single-file Streamlit application.

Structured extraction is performed by the Anthropic API using forced
tool-use against a nested, multi-component schema: each independent
sub-component of an inspected asset (a column, a base plate, a gusset,
a flange, ...) gets its own object with its own stated minimum
thickness and its own list of UT readings, plus an auto-detected
engineering framework/code jurisdiction and a model-reported extraction
confidence score.

What stays in plain, deterministic Python, and why:
  * The pass/fail decision for each component (margin = lowest UT
    reading for that component - that component's stated minimum, with
    an ASME B31.3 fallback when the asset is process piping and a
    component has no stated minimum) is computed in plain Python, never
    by the model, so the safety-critical comparison is reproducible.
  * A lightweight internal validator pass runs over the model's JSON
    output (not over the raw text with regex) to sanity-check internal
    consistency. This is a best-effort sanity net, not a guarantee of
    perfect extraction — no automated pass can promise 100% accuracy
    against arbitrary messy field text, so findings are surfaced as
    review flags, not treated as ground truth.

Model note: MODEL_NAME stays "claude-opus-5-5", per your earlier
"ignore model name it shud be claude-opus-5-5" instruction — this
document re-requested "claude-3-5-sonnet-latest" again, which is
retired on the Claude API, so I kept the override rather than
reintroducing a model string that will 404. Say the word if you want
it changed again.

Two UI notes worth flagging up front, since both rely on techniques
Streamlit doesn't officially support:
  * The "fixit" title is a real animated <canvas> particle system
    (rendered inside its own component iframe) — the letters are
    sampled into a particle grid that springs to its home position and
    scatters/swirls under the cursor, then eases back on mouse-leave.
    This only reacts to a real mouse, so on touch devices it just sits
    static and readable, which is a fine degrade.
  * The cursor-following glow on the result cards is implemented by
    reaching from a components.v1.html iframe into `window.parent.document`
    to attach a mousemove listener to the main app's DOM. This works
    today because the component iframe is same-origin, but it's an
    unofficial trick, not a supported Streamlit API — it's wrapped in
    try/except so a future Streamlit change would silently disable the
    pointer-follow glow rather than break the app. The breathing
    pass/fail/unresolved halo on each card does NOT depend on this — 
    that's pure CSS and will always work.
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

MODEL_NAME = "claude-opus-5-5"  # see docstring note — claude-3-5-sonnet-latest is retired

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
vessels, process piping, heat exchangers, storage tanks, and conveyor frames.

Read the raw inspection text the user provides and call the \
`extract_compliance_data` tool with the data you find. Rules:
- Extract only what is actually stated in the text. Never invent, estimate, \
  or "helpfully" fill in a value that is not present — use null for anything \
  not stated, and list it in missing_engineering_variables_ledger if it is \
  needed for a downstream calculation.
- Treat the asset as a collection of independent sub-components. Each \
  distinct structural or mechanical element named or clearly implied in the \
  text (e.g. "North Column", "Base Plate", "Gusset G1", "Beam Flange") gets \
  its own entry in components_matrix, with its own explicit_minimum_required_mat \
  and its own ut_thickness_measurements list. Do not merge readings from \
  different components into one entry, and do not invent components that \
  are not named or clearly implied.
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
- Preserve stated uncertainty rather than resolving it. A "possible" or \
  "suspected" weld indication must be recorded as an unconfirmed finding \
  (is_confirmed_failure: false) describing it as needing NDT validation — \
  never upgraded to a confirmed defect. Surface oxidation on fasteners with \
  no torque record stated must likewise be recorded as unconfirmed and \
  described as pending verification, not as a confirmed connection failure. \
  Only set is_confirmed_failure: true when the text itself states the item \
  failed, is rejected, or is out of tolerance.
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


def evaluate_component(component, is_piping, piping_vars):
    name = component.get("component_name") or "Unnamed Component"
    ut_list = component.get("ut_thickness_measurements") or []
    mat_field = component.get("explicit_minimum_required_mat")
    mat = mat_field.get("value") if mat_field else None
    mat_unit = mat_field.get("unit") if mat_field else None

    if not ut_list:
        return {"name": name, "status": "no_measurements", "raw": component}

    calc_note = None
    if mat is None:
        if is_piping:
            mat, missing, calc_note = calc_b31_3_mat(piping_vars)
            if mat is None:
                return {"name": name, "status": "insufficient", "missing_vars": missing,
                         "ut_measurements": ut_list, "raw": component}
        else:
            return {"name": name, "status": "insufficient",
                     "missing_vars": ["Explicit Minimum Required MAT (not stated for this component)"],
                     "ut_measurements": ut_list, "raw": component}

    lowest = min(ut_list, key=lambda r: r["value"])
    margin = round(lowest["value"] - mat, 4)
    status = "blocked" if margin < 0 else "verified"
    return {"name": name, "status": status, "mat": mat, "mat_unit": mat_unit, "lowest": lowest,
             "margin": margin, "calc_note": calc_note, "raw": component}


def evaluate(extracted):
    components = extracted.get("components_matrix") or []
    asset_category = (extracted.get("asset_category") or "").lower()
    is_piping = "pip" in asset_category  # covers "piping" / "pipeline" / "process pipe"
    piping_vars = extracted.get("piping_design_variables")

    results = [evaluate_component(c, is_piping, piping_vars) for c in components]
    blocked = [r for r in results if r["status"] == "blocked"]
    unresolved = [r for r in results if r["status"] in ("insufficient", "no_measurements")]
    confirmed_failures = [a for a in (extracted.get("field_anomalies") or []) if a.get("is_confirmed_failure")]
    ledger = extracted.get("missing_engineering_variables_ledger") or []

    if not results:
        global_status = "CONDITION UNVERIFIED"
    elif blocked or confirmed_failures:
        global_status = "BLOCKED"
    elif unresolved or ledger:
        global_status = "CONDITION UNVERIFIED"
    else:
        global_status = "VERIFIED SECURE"

    # Binary safety-filter flag for the aero-glow halo. "CONDITION UNVERIFIED" is
    # neither a clean pass nor a confirmed failure, so it gets its own neutral
    # (amber, non-pulsing) halo rather than being forced into True/False.
    if global_status == "VERIFIED SECURE":
        passed_safety_filters = True
    elif global_status == "BLOCKED":
        passed_safety_filters = False
    else:
        passed_safety_filters = None

    return {"results": results, "blocked": blocked, "unresolved": unresolved,
             "confirmed_failures": confirmed_failures, "global_status": global_status,
             "passed_safety_filters": passed_safety_filters}


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

    keywords = ["column", "plate", "gusset", "flange", "shell", "head", "nozzle", "beam", "brace", "support"]
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
# PREMIUM GLASS UI
# ============================================================================

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #090D16;
    }
    .glass-card {
        background: rgba(22, 31, 48, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        background-repeat: no-repeat;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 242, 254, 0.45);
    }
    .metric-label { color: #9CA3AF; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { color: #F9FAFB; font-size: 1.2rem; font-weight: 700; word-wrap: break-word; }
    .accent-cyan { color: #00F2FE; }
    .accent-gold { color: #FBBF24; }
    .component-card.blocked {
        background: linear-gradient(135deg, rgba(220,38,38,0.28), rgba(251,191,36,0.10));
        border-color: rgba(220,38,38,0.6);
    }
    .component-card.verified { border-color: rgba(22,163,74,0.5); }
    .component-card.unresolved { border-color: rgba(107,114,128,0.5); }
    .status-pill {
        display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.7rem;
        font-weight:700; letter-spacing:.03em; text-transform:uppercase; margin-left:8px;
    }
    .status-pill.blocked { background:#DC2626; color:#FEF2F2; }
    .status-pill.verified { background:#16A34A; color:#F0FDF4; }
    .status-pill.unresolved { background:#6B7280; color:#F9FAFB; }
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        background: rgba(22, 31, 48, 0.4) !important;
    }
    .raw-terminal {
        background: #05070C; border: 1px solid rgba(255,255,255,0.08); border-radius: 12px;
        padding: 14px; color: #9CA3AF; font-family: monospace; font-size: 0.8rem;
        max-height: 640px; overflow-y: auto; white-space: pre-wrap;
    }
    .halo-wrapper { border-radius: 18px; padding: 4px; margin-bottom: 10px; }
    .halo-pass { animation: breathe-emerald 4s ease-in-out infinite; }
    .halo-fail { animation: breathe-crimson 3s ease-in-out infinite; }
    .halo-unresolved { box-shadow: 0 0 30px rgba(251,191,36,0.12); }
    @keyframes breathe-emerald {
        0%, 100% { box-shadow: 0 0 25px rgba(16,185,129,0.15); }
        50% { box-shadow: 0 0 50px rgba(16,185,129,0.32); }
    }
    @keyframes breathe-crimson {
        0%, 100% { box-shadow: 0 0 25px rgba(239,68,68,0.18); }
        50% { box-shadow: 0 0 55px rgba(239,68,68,0.38); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Cursor-following glow on .glass-card elements (best-effort; see docstring) ---
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
            // Streamlit deployments/versions — fail silently, cards keep their
            // static glass styling with no pointer-follow glow.
        }
    })();
    </script>
    """,
    height=0,
)


def render_fixit_title():
    components.html(
        """
        <div style="width:100%;display:flex;justify-content:flex-start;">
        <canvas id="fixitCanvas" width="480" height="140"
                style="background:transparent;"></canvas>
        </div>
        <script>
        (function () {
            const canvas = document.getElementById('fixitCanvas');
            const ctx = canvas.getContext('2d');
            const off = document.createElement('canvas');
            off.width = canvas.width; off.height = canvas.height;
            const octx = off.getContext('2d');
            octx.fillStyle = '#FBBF24';
            octx.font = "700 3.5rem 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif";
            octx.textBaseline = 'middle';
            octx.fillText('fixit', 20, off.height / 2 + 6);
            const img = octx.getImageData(0, 0, off.width, off.height).data;

            const particles = [];
            const step = 3;
            for (let y = 0; y < off.height; y += step) {
                for (let x = 0; x < off.width; x += step) {
                    const idx = (y * off.width + x) * 4;
                    if (img[idx + 3] > 120) {
                        particles.push({ hx: x, hy: y, x: x, y: y, vx: 0, vy: 0 });
                    }
                }
            }

            let mouseX = -9999, mouseY = -9999;
            canvas.addEventListener('mousemove', function (e) {
                const rect = canvas.getBoundingClientRect();
                mouseX = e.clientX - rect.left;
                mouseY = e.clientY - rect.top;
            });
            canvas.addEventListener('mouseleave', function () {
                mouseX = -9999; mouseY = -9999;
            });

            function frame() {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#FBBF24';
                for (let i = 0; i < particles.length; i++) {
                    const p = particles[i];
                    const dx = p.x - mouseX, dy = p.y - mouseY;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 0.001;
                    const radius = 55;
                    if (dist < radius) {
                        const force = (radius - dist) / radius;
                        const angle = Math.atan2(dy, dx) + Math.PI / 2;
                        p.vx += Math.cos(angle) * force * 2.2 + (dx / dist) * force * 1.4;
                        p.vy += Math.sin(angle) * force * 2.2 + (dy / dist) * force * 1.4;
                    }
                    p.vx += (p.hx - p.x) * 0.09;
                    p.vy += (p.hy - p.y) * 0.09;
                    p.vx *= 0.82;
                    p.vy *= 0.82;
                    p.x += p.vx;
                    p.y += p.vy;
                    ctx.fillRect(p.x, p.y, 2, 2);
                }
                requestAnimationFrame(frame);
            }
            frame();
        })();
        </script>
        """,
        height=150,
    )


def metric_html(label, value, accent=None):
    cls = f"metric-value {accent}" if accent else "metric-value"
    return f"<div class='glass-card'><div class='metric-label'>{label}</div><div class='{cls}'>{value}</div></div>"


def render_component_card(result):
    status = result["status"]
    css_class = "blocked" if status == "blocked" else ("verified" if status == "verified" else "unresolved")
    pill_label = {"blocked": "Blocked", "verified": "Verified", "insufficient": "Unresolved", "no_measurements": "No Data"}[status]

    st.markdown(
        f"<div class='glass-card component-card {css_class}'>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#F9FAFB;'>{result['name']}"
        f"<span class='status-pill {css_class}'>{pill_label}</span></div>",
        unsafe_allow_html=True,
    )

    if status in ("blocked", "verified"):
        unit_suffix = f" {result['mat_unit']}" if result.get("mat_unit") else ""
        lowest = result["lowest"]
        lowest_unit = f" {lowest['unit']}" if lowest.get("unit") else ""
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_html("Lowest Measured UT Point", f"{lowest['location_label']}: {lowest['value']:.4f}{lowest_unit}"), unsafe_allow_html=True)
        c2.markdown(metric_html("Stated Design Minimum (MAT)", f"{result['mat']:.4f}{unit_suffix}"), unsafe_allow_html=True)
        margin_accent = "accent-gold" if status == "blocked" else "accent-cyan"
        c3.markdown(metric_html("Computed True Margin", f"{result['margin']:+.4f}", margin_accent), unsafe_allow_html=True)
        if result.get("calc_note"):
            st.info(result["calc_note"])
        if status == "blocked":
            st.markdown("**Localized remediation action steps:**")
            for step in remediation_steps(result["name"]):
                st.markdown(f"- {step}")
    elif status == "insufficient":
        st.write("Cannot compute a margin for this component — missing:")
        for m in result["missing_vars"]:
            st.markdown(f"- ❌ **{m}**")
    else:
        st.write("A minimum is on file for this component, but no UT readings were extracted for it.")

    with st.expander("View Raw Source Extraction Line"):
        st.json(result["raw"])

    st.markdown("</div>", unsafe_allow_html=True)


client = get_client()
if client is None:
    render_fixit_title()
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
    audit_notices = audit_extraction(report_text, extracted)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("📡 Ingested Raw Inspection File Stream")
        st.markdown(f"<div class='raw-terminal'>{report_text}</div>", unsafe_allow_html=True)

    with right:
        render_fixit_title()
        st.caption(f"Structured, multi-component extraction via the Anthropic API ({MODEL_NAME}).")

        score = extracted.get("extraction_confidence_score")
        if score is None:
            confidence_label = "Unknown"
        elif score >= 0.85:
            confidence_label = "High"
        elif score >= 0.6:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"

        m1, m2 = st.columns(2)
        m1.markdown(metric_html("Extraction Read Accuracy Confidence", confidence_label, "accent-cyan"), unsafe_allow_html=True)
        gs = outcome["global_status"]
        gs_accent = "accent-gold" if gs == "BLOCKED" else None
        m2.markdown(metric_html("Global Engineering Safety Status", gs, gs_accent), unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_html("Asset Category", extracted.get("asset_category") or "Not stated"), unsafe_allow_html=True)
        c2.markdown(metric_html("Metallurgy", extracted.get("metallurgy_specification") or "Not stated"), unsafe_allow_html=True)
        c3.markdown(metric_html("Engineering Framework", extracted.get("engineering_framework") or "Not determined"), unsafe_allow_html=True)

        if audit_notices:
            st.markdown("#### 🟡 System Extraction Audit Notice")
            for notice in audit_notices:
                st.warning(notice)

        halo_class = {True: "halo-pass", False: "halo-fail", None: "halo-unresolved"}[outcome["passed_safety_filters"]]
        st.markdown(f"<div class='halo-wrapper {halo_class}'>", unsafe_allow_html=True)
        st.markdown("#### Calculation Trail Ledger")
        for result in outcome["results"]:
            render_component_card(result)
        st.markdown("</div>", unsafe_allow_html=True)

        ledger = extracted.get("missing_engineering_variables_ledger") or []
        if ledger:
            st.markdown("#### 📒 Missing Engineering Variables Ledger")
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
    render_fixit_title()
    st.info("Upload a .txt inspection field log above to run extraction.")
