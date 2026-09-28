"""
fixit — Real AI Mode (Dual-Pass Hybrid Extraction)
-----------------------------------------------------
Single-file Streamlit application.

ARCHITECTURE CHANGE FROM THE PRIOR VERSION: the model is no longer asked
to return a deeply nested JSON array (components -> nested MAT object ->
nested UT-reading array). On very dense reports, a deeply nested
tool-use schema asks the model to hold open several levels of JSON
structure simultaneously while also transcribing many numeric values,
and that combination is where token-budget pressure and truncated tool
calls are most likely to show up. Splitting the job in two — a single
flat text field for transcription, parsed by plain Python afterward —
removes one layer of that structural bookkeeping and gives one string
field a much larger, singular token allowance instead of splitting the
budget across many nested objects. That is a real mitigation, not a
guarantee: no format change makes an LLM's output immune to truncation
on an arbitrarily long input, so this file still surfaces parse
problems as visible audit notices rather than silently pretending
nothing was dropped.

Dual-pass framework:
  PASS 1 (model, via extract_with_claude): the model reads the raw
    report and returns asset-level fields (asset_category, metallurgy,
    engineering_framework, jurisdiction, confidence_metrics [a
    multi-axis HIGH/MEDIUM/LOW self-assessment, not a single ambiguous
    float], piping_design_variables, pressure_events [an array, so
    narrative-only excursions never vanish], missing_engineering_
    variables_ledger, field_anomalies [each tagged with one of a strict
    5-tier risk_category: FAIL_BELOW_CRITERION / CONFLICT /
    MISSING_INFORMATION / INFORMATIONAL / REQUIRES_VERIFICATION]) as
    before, PLUS a single string field, "flat_manifest_block", holding
    every component's name, stated minimum, unit, current UT readings,
    any prior-inspection UT readings, and optional decision-traceability
    metadata (source document, authority tier, and a one-line reason for
    each selected value) as plain marked-up text (see
    FLAT_MANIFEST_FORMAT_SPEC below for the exact grammar).
  PASS 2 (plain Python, native_parameter_matrix_parser): splits that
    string on its own explicit boundary markers and reconstructs the
    same per-component structure the rest of this file already expects
    (component_name / explicit_minimum_required_mat / ut_thickness_
    measurements / historical_ut_thickness_measurements). This pass
    runs entirely in the container with ordinary string methods and
    regex — no second model call, no dependency on the model getting
    JSON nesting right.

What stays in plain, deterministic Python, and why (unchanged from
before):
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
  * A lightweight internal validator pass runs over the *parsed* data
    (not over the raw text with regex, and not trusted from the model
    as ground truth) as a sanity net before anything is rendered. This
    is best-effort, not a guarantee of perfect extraction — findings
    are surfaced as review flags, not treated as fact.
  * The engineering disposition is never auto-prescribed as a single
    fixed outcome. A confirmed exceedance (a mathematical fact — margin
    below zero) is always rendered separately from the recommended next
    step, which is labeled as "Awaiting Authorized Engineering/Inspector
    Review" rather than a final repair order. Unconfirmed findings
    (e.g. a "possible" weld indication, or an untorqued fastener) are
    always kept as open, unconfirmed items pending secondary
    verification, never upgraded to a confirmed defect.

REFACTOR (units / component types / precedence / master records):
  * Every reading keeps the unit printed on its own row; a document-wide unit
    is only a flagged last resort (see _parse_reading_list).
  * audit_and_reconcile() runs after parsing and before any margin is
    computed: dual-unit normalization, tube unit sanity check (> 0.500 in
    => mm misidentified), signed-addendum supersession, repair-pad linkage,
    unit-aware historical reconciliation (<= 1% after conversion), and MAT
    binding by component_type (tube MAT is never the shell MAT).
  * build_master_records() emits ONE consolidated record per physical
    component (S1..S6, T1..T4) with its full lineage and verified calculation.

Model note: MODEL_NAME stays "claude-opus-5-5" per your standing
instruction.

Title note: the "fixit" wordmark is a <span> inside its own isolated
container div, not an <h1> — this avoids Streamlit's automatic
anchor-link injection on native heading elements. A global CSS rule
also hides Streamlit's anchor-chain icon on any other native headers
used elsewhere in the page (st.subheader, markdown "####" blocks), so
no 🔗 icon appears anywhere in the app. No <canvas>, no JS, no
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

    /* Remove Streamlit's default top padding and any residual native-header
       anchor-chain icons (🔗) anywhere else in the page. */
    .block-container { padding-top: 2rem; }
    [data-testid="stHeaderActionElements"] { display: none !important; }
    h1 a, h2 a, h3 a, h4 a, h5 a, h6 a { display: none !important; }

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

    /* Calculation-trail table used inside each component card */
    .calc-trail-table { width: 100%; border-collapse: collapse; margin: 8px 0 4px 0; }
    .calc-trail-table td { padding: 4px 8px; font-size: 0.85rem; color: #E5E7EB; vertical-align: top; }
    .calc-trail-table td.label { color: #9CA3AF; white-space: nowrap; width: 1%; }

    /* 5-tier risk taxonomy ledger — replaces the old flat "unresolved" bucket */
    .risk-tier-card { border-radius: 14px; padding: 10px 16px; margin-bottom: 8px; font-size: 0.88rem; }
    .risk-tier-card .tier-label { font-weight: 800; font-size: 0.68rem; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 3px; }
    .risk-tier-card.tier-fail { background: rgba(69,10,10,0.4); border: 1px solid #EF4444; color: #FECACA; }
    .risk-tier-card.tier-fail .tier-label { color: #F87171; }
    .risk-tier-card.tier-conflict { background: rgba(120,53,15,0.35); border: 1px solid #F97316; color: #FED7AA; }
    .risk-tier-card.tier-conflict .tier-label { color: #FB923C; }
    .risk-tier-card.tier-missing { background: rgba(30,58,138,0.3); border: 1px solid #60A5FA; color: #DBEAFE; }
    .risk-tier-card.tier-missing .tier-label { color: #93C5FD; }
    .risk-tier-card.tier-verify { background: rgba(120,95,15,0.28); border: 1px solid #FBBF24; color: #FEF3C7; }
    .risk-tier-card.tier-verify .tier-label { color: #FDE68A; }
    .risk-tier-card.tier-info { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.15); color: #D1D5DB; }
    .risk-tier-card.tier-info .tier-label { color: #9CA3AF; }

    /* Decoupled math-vs-workflow status line inside a component card */
    .decoupled-status-line { font-size: 0.8rem; color: #9CA3AF; margin: 4px 0 8px 0; }
    .decoupled-status-line b { color: #E5E7EB; }
    </style>

    <div><div class="glass-title-container">
        <span class="glass-title" data-text="fixit">fixit</span>
    </div></div>
    """,
    unsafe_allow_html=True,
)

MODEL_NAME = "claude-opus-5-5"  # kept per explicit standing instruction

# ============================================================================
# ANTHROPIC CLIENT
# ============================================================================

def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


# ----------------------------------------------------------------------------
# FLAT MANIFEST FORMAT SPEC — the exact grammar the model must emit inside
# the single "flat_manifest_block" string field. Kept as a constant (rather
# than only inline in the prompt) so the parser docstring below and the
# system prompt can both reference the identical spec text.
# ----------------------------------------------------------------------------
FLAT_MANIFEST_FORMAT_SPEC = """\
Component: <component name, exactly as named or clearly implied in the text>
Component_Type: <MANDATORY. One of: shell, tube, channel, flange, head, nozzle, support, plate, other. The
  physical sub-component class. Tube bundle / tube wall / heating coil / T1..Tn = "tube"; shell course /
  shell cylinder / S1..Sn = "shell". Tubes and shells have DIFFERENT minimum allowable thicknesses.>
MAT: <numeric stated minimum FOR THIS COMPONENT'S OWN CLASS, only if explicitly stated — omit this whole line if none is stated. Never copy the shell MAT onto a tube.>
Unit: <unit of the MAT value ONLY (in or mm). It is NOT a default for the readings — every reading carries its own unit (see UT lines).>
MAT_Authority_Tier: <one of the six source-precedence tiers below — omit if MAT is omitted>
MAT_Source: <the specific document/section the minimum came from, e.g. "Design / Operating Information" — omit if unknown>
MAT_General_Criterion: <numeric value of a general/default criterion this minimum overrides, if the text gives both a general and a special-case number — omit if not applicable>
MAT_Reason: <one short sentence on why this minimum (vs. any other candidate number in the text) was selected — omit if there was only one candidate number>
Attached_Reinforcement: pad_thickness=<number> <unit>; pressure_retaining=false   (omit unless a repair pad / reinforcement plate / doubler is attached to THIS component; the base-shell readings still go on the UT/Reading_Set lines, the pad thickness goes ONLY here)
Reading_Set: <document_type> | <document_date, ISO YYYY-MM-DD if possible, else as written> | <signed_addendum: yes/no> | <source document name> | <label>=<value> <unit>, <label>=<value> <unit>, ...
  (one line PER DOCUMENT that gives current-cycle readings for this component. If a signed addendum / re-run and an
   earlier final report BOTH give readings for the same component, emit two Reading_Set lines — never pick or merge
   them yourself; Python applies supersession. Example:
   Reading_Set: Final Inspection Report | 2026-09-28 | no | Final Inspection Report - Sept 28 | S3=9.45 mm
   Reading_Set: Signed Addendum | 2026-09-30 | yes | Signed Addendum - Sept 30 | S3-A=9.61 mm, S3-B=9.58 mm)
UT: <label>=<value> <unit>, ...   (legacy single-source form of Reading_Set — use it only when exactly one document gives readings; omit if none. Every reading MUST carry its own unit.)
UT_Authority_Tier: <one of the six source-precedence tiers below, for the current UT readings — omit if UT is omitted>
UT_Source: <the specific document/section the current readings came from — omit if unknown>
UT_Source_Date: <date of that reading, if stated — omit if unknown>
UT_All_Values_In_Region: <every candidate value the text gave for this same location/region, comma-separated, if more than one was mentioned — omit if there was only one>
UT_Reason: <one short sentence on why the selected lowest reading (vs. other candidates in the region) was chosen — omit if there was only one candidate>
Historical_Set: <date, YYYY-MM-DD or year> | <source document name / kind, e.g. "Signed 2024 report" or "2024 spreadsheet"> | <label>=<value> <unit>, ...
  (one line PER source of prior-inspection readings. If the same historical reading appears in two sources in
   different units — e.g. 0.402 in in a spreadsheet and 10.20 mm in a signed report — emit BOTH lines with each
   source's own printed unit; never convert or de-duplicate yourself. Python reconciles them.)
Historical_UT: <label>=<value> <unit>, ...   (legacy single-source form of Historical_Set — omit if no prior-inspection readings exist)
Component_End

Repeat one such block per component, separated by a blank line. Use a
real prior location label from the text for each <label> if the text
names sampling points (e.g. North=0.598); if the text gives readings
with no location name at all, label them sequentially R1=, R2=, etc.
Never invent a reading or a minimum that is not in the text. Every
_Authority_Tier / _Source / _General_Criterion / _Reason line is optional
provenance metadata — include it only when the text actually supports it;
never fabricate a reason or a source document name.

LABELING RULES THAT DRIVE DOWNSTREAM SAFETY LOGIC (label text is parsed
by plain Python afterward — get these exact, since a mislabeled reading
changes which calculation it feeds):
  * ONE BLOCK PER PHYSICAL COMPONENT — NEVER SPLIT ACROSS TABLES. If the
    same named component (e.g. "S3", "Shell Course 2") appears in more
    than one table or section of the report — for example a current-cycle
    UT table AND a separate prior-year/historical UT table — it still
    gets exactly ONE "Component:" block. Put the current readings on the
    UT: line and the prior-year readings on the Historical_UT: line of
    THAT SAME block. Never emit two "Component: S3" blocks just because
    the name appears in two different tables.
  * PAD / REINFORCEMENT READINGS ARE NEVER MIXED INTO THE BASE-SHELL UT:
    LINE. If a reading is explicitly taken on a repair pad, reinforcement
    plate, doubler plate, or similar overlay rather than the original
    parent-metal shell/wall, label that reading's location with a word
    that says so (e.g. "S3_Pad=0.415", "Pad=0.415", "ReinforcementPlate
    =0.415") so it is recognizable as a pad reading. A component with
    both a base-shell reading and a pad reading over the same location
    reports both — the base-shell reading on the UT: line under its own
    plain location label (e.g. "S3=0.368") and the pad reading under a
    label containing "Pad"/"Reinforcement"/"Plate"/"Doubler". Never let a
    pad reading raise, lower, or replace the base-shell reading for that
    location.
  * PRE-PREP VS. POST-PREP READINGS. If the text records a reading taken
    before surface preparation (grinding/cleaning/blasting) and a second
    reading taken after surface preparation at the same location, label
    each with a location that says which is which (e.g. "S7_PrePrep
    =9.48", "S7_PostPrep=9.55" — "Pre-Prep"/"Post-Prep",
    "As-Found"/"As-Left", or "Before-Prep"/"After-Prep" are all
    acceptable, but always include one of those words). Never report only
    the post-prep number and silently drop the pre-prep number, and never
    merge the two into a single averaged or "best" value — both must
    appear as separate entries on the UT: line.

UNIT / TYPE / PAD RULES THAT DRIVE DOWNSTREAM SAFETY LOGIC:
  * EVERY reading carries its own unit right after the number ("9.45 mm",
    "0.072 in"). Never copy the document-wide unit onto a row that shows a
    different one, and never convert values yourself.
  * Component_Type is mandatory on every block; tubes and shells never share
    a MAT.
  * A repair pad's thickness goes only on Attached_Reinforcement, never on the
    base-shell reading lines.

SOURCE PRECEDENCE TIERS (highest authority first — use these exact strings
for any _Authority_Tier line, and never let a lower tier silently overwrite
a higher one; see the CONFLICT rule below instead):
  1. Signed Inspection / UT Official Record
  2. Controlled Inspection Worksheet
  3. Supplemental UT Measurement
  4. Maintenance Note
  5. Handwritten / Unsigned Field Note
  6. General Narrative / Qualitative Summary"""


SYSTEM_PROMPT = f"""You are a structured-data extraction engine for industrial \
engineering inspection field logs covering multi-component assets: structural \
steel assemblies (columns, base plates, gussets, beams, braces), pressure \
vessels, process piping (including tube bundles / heating coils), heat \
exchangers, storage tanks, and conveyor frames.

Read the raw inspection text the user provides and call the \
`extract_compliance_data` tool. Rules:

- Extract only what is actually stated in the text. Never invent, estimate, \
  or "helpfully" fill in a value that is not present — use null (or an \
  omitted line, per the manifest format below) for anything not stated, and \
  list it in missing_engineering_variables_ledger if it is needed for a \
  downstream calculation.

- "flat_manifest_block" is a single string holding every component's data as \
  plain marked-up text — NOT JSON, NOT a nested array. Follow this exact \
  grammar for every component block:

{FLAT_MANIFEST_FORMAT_SPEC}

- Treat the asset as a collection of independent sub-components. Each \
  distinct structural or mechanical element named or clearly implied in the \
  text (e.g. "North Column", "Base Plate", "Gusset G1", "Beam Flange", \
  "Process Tubes") gets its own "Component:" block. Do not merge readings \
  from different components into one block, and do not invent components \
  that are not named or clearly implied. If a component is named but the \
  text gives it neither a stated minimum nor any reading, still emit its \
  block with both the MAT and UT lines omitted — do not omit the component \
  entirely.
- A "Historical_UT:" line holds readings the text attributes to a prior \
  inspection (a previous date, a "last inspection", a baseline) for that \
  same component. Omit the line if the text gives no prior-inspection \
  reading for that component. Never treat a current-cycle reading as \
  historical, and never treat a historical reading as current.
- SOURCE PRECEDENCE: when the text gives more than one candidate value for \
  the same measurement or the same criterion (e.g. a signed UT sheet and a \
  handwritten field note disagree), select the value from the highest-tier \
  source per the precedence list above, record that choice via \
  MAT_Authority_Tier/UT_Authority_Tier plus a one-line MAT_Reason/UT_Reason, \
  and log every other candidate value in MAT_General_Criterion or \
  UT_All_Values_In_Region as applicable. A lower-tier source must NEVER \
  silently overwrite a higher-tier one. If authority truly cannot be \
  determined (both sources carry equal apparent standing, or the text gives \
  no way to rank them), do not silently pick one — instead add an entry to \
  field_anomalies with risk_category "CONFLICT", naming both values and \
  both sources in the "notes" field, and still select the higher of the two \
  values as a conservative placeholder in the manifest so downstream \
  calculations do not silently under-report risk.
- UNIT RULE (highest-priority rule for every measurement): every reading in \
  a UT:/Reading_Set:/Historical_Set:/Historical_UT: line, and every MAT or pad \
  thickness, MUST carry its own unit written directly after the number \
  (e.g. "9.45 mm", "0.072 in"). Take the unit from the nearest explicit \
  marker for THAT row/cell — the cell text itself, then that column header, \
  then that table's caption — and only afterwards from a document-wide \
  default. An explicit row/column unit marker ALWAYS overrides the \
  document-wide default: if a header says "all thicknesses in inches" but a \
  row or column is labeled mm, that row is mm. Record the value exactly as \
  printed with its printed unit; never convert units yourself. If no unit is \
  recoverable for a reading, write the number with no unit — never guess one.
- COMPONENT TYPING: every Component block MUST include Component_Type \
  (shell, tube, channel, flange, head, nozzle, support, plate, other). A \
  heat exchanger has separate shell and tube components with different \
  minimum allowable thicknesses; tag each correctly.
- MAT BINDING: "global_minimum_allowable_thickness" is a SHELL-CLASS default \
  and must never be used for tubes. Put every class-specific minimum in \
  "component_mat_table" (one entry per component_type, e.g. tube = 0.070 in, \
  shell = 0.375 in, each with value and unit). A component's own MAT: line is \
  used only when the text states a minimum for that specific component.
- DOCUMENT METADATA & SUPERSESSION: list every source document in \
  "document_registry" (name, type such as Final Inspection Report / Signed \
  Addendum / Re-run, date, and whether it is a signed addendum). When a signed \
  addendum or re-run gives new readings for a component that an earlier report \
  also covers, emit BOTH as separate Reading_Set lines with their own \
  document metadata. Do not choose between them and do not delete the older \
  one — supersession is applied afterward in Python.
- HISTORICAL RECORDS: emit one Historical_Set line per source (e.g. a signed \
  2024 report and a 2024 spreadsheet), each with its own printed unit, even \
  when they appear to be the same physical reading in different units. Do not \
  convert or de-duplicate; Python reconciles them by physical equivalence.
- REPAIR PADS: if a reinforcement/repair pad is attached to a component, put \
  the base pressure-retaining shell readings on the reading lines and the pad \
  thickness ONLY on the Attached_Reinforcement line (pressure_retaining=false). \
  Never place a pad thickness among the base-shell readings.
- "global_minimum_allowable_thickness" is an asset-level (not per-component) \
  minimum thickness the text states applies to the whole vessel/asset by \
  default — e.g. a header or design-basis line reading "Minimum Allowable \
  Shell Thickness (MAT): 0.375 in". Extract it only when the text states it \
  as a document-wide default, with both value and unit; null if no such \
  document-wide default is stated. This is separate from any per-component \
  MAT: line in the flat manifest — a component's own explicitly stated \
  minimum (its own MAT: line) always takes precedence over this document-wide \
  default; this field exists purely so a component the text never gives its \
  own explicit minimum can still be evaluated against the stated document-wide \
  default rather than falling through to MISSING_INFORMATION. Never invent \
  this value if the text gives no document-wide default at all.
- "engineering_framework" is the applicable governing code if the text \
  states or clearly implies one (e.g. "ASME Sec VIII", "ASME B31.3", \
  "AWS D1.1", "API 653"); null if not determinable. "jurisdiction" is the \
  applicable regulatory jurisdiction/authority if stated (e.g. a state \
  boiler/pressure-vessel jurisdiction, a client/site standard); null if not \
  determinable. Both are asset-level, not per-component.
- "confidence_metrics" is your own self-assessment, not a measure of the \
  asset's physical condition: text_extraction_confidence (HIGH/MEDIUM/LOW) \
  rates how legible and unambiguous the source text itself was; \
  calculation_confidence (HIGH/MEDIUM/LOW) rates how confident you are that \
  the values you selected are the ones a careful human would have picked \
  (lower this whenever you had to apply the source-precedence rule above); \
  source_conflict_level (NONE/LOW/MEDIUM/HIGH) rates how much the source \
  documents disagreed with each other across the whole report.
- "piping_design_variables" (design pressure, outside diameter, allowable \
  stress, quality/joint factor, Y coefficient, corrosion allowance) apply at \
  the asset level and are only relevant for process piping. Extract each \
  only if explicitly present. Be careful not to confuse the allowable stress \
  value (typically a large number, e.g. in the thousands, in psi or MPa) \
  with an adjacent small decimal thickness reading — they are different \
  quantities even when they appear near each other in the text.
- "pressure_events" is an array — capture EVERY distinct logged \
  operating-pressure excursion, startup spike, or upset mentioned anywhere \
  in the text, including in narrative operator logs or notes, not only ones \
  that appear in a data table. Each entry needs a unique event_id you assign \
  (e.g. "PE-01"), the peak/observed pressure, the stated design pressure \
  threshold, timestamp_or_context (whatever time/context marker the text \
  gives, e.g. "09:42 startup upset"), duration_minutes (how long the \
  excursion was sustained, in minutes, only if the text states or clearly \
  implies one — e.g. a start/end time pair, or an explicit "for N min"; \
  leave null if no duration is stated or determinable — never guess or \
  infer a duration from context alone), is_duplicate_or_continuation (true \
  if this entry is the same physical event as an earlier one just \
  mentioned again elsewhere in the text, e.g. re-stated in a summary \
  paragraph), and related_event_ids (the event_id(s) it duplicates or \
  continues, if any). Do NOT compute variance yourself, and do NOT decide \
  yourself whether an event "qualifies" as an excursion — the pressure-AND \
  -duration qualification gate is applied afterward in plain Python from \
  the values you provide. Only include an event if both the observed and \
  design pressure are explicitly stated.
- Preserve stated uncertainty rather than resolving it. A "possible" or \
  "suspected" weld indication must be recorded in field_anomalies with \
  risk_category "REQUIRES_VERIFICATION" and notes describing it as needing \
  NDT validation — never upgraded to a confirmed defect. A corroded, loose, \
  or unverified fastener must likewise be risk_category \
  "REQUIRES_VERIFICATION", with notes describing it as pending torque \
  verification — never treated as a confirmed connection failure.
- "field_anomalies" uses a strict 5-tier, mutually-exclusive risk_category \
  for every entry — pick exactly one per finding:
    - FAIL_BELOW_CRITERION: a numerical measurement is stated in the text \
      itself (in prose, not just in the manifest) as strictly below an \
      explicitly provided requirement.
    - CONFLICT: two data sources directly disagree and authority cannot be \
      automatically resolved (see SOURCE PRECEDENCE above).
    - MISSING_INFORMATION: data required for a critical calculation or \
      evaluation was not supplied anywhere in the text.
    - INFORMATIONAL: historical repairs/replacements, superseded readings \
      (e.g. an old reading explicitly replaced by a newer one), duplicate \
      unit conversions, or routine qualitative observations with no safety \
      implication (e.g. "looks fine").
    - REQUIRES_VERIFICATION: a physical/visual anomaly (unconfirmed weld \
      indication, surface corrosion, alignment offset) needing human or NDT \
      inspection before it can be called confirmed or dismissed.
  Capture every inspector note, flagged indication, weld observation, \
  fastener condition note, superseded/replaced reading, or geometry/ \
  alignment issue, in the report's own words in the "notes" field.
- Do not comment on overall compliance, pass/fail, or safety, and do not \
  recommend a specific code-level action (e.g. "perform an API 579 FFS \
  assessment") anywhere in free text — only extract data as stated. All \
  pass/fail comparison, degradation-delta, pressure-variance math, and any \
  code-specific recommendation gating are performed afterward in plain \
  Python, not by you."""

EXTRACTION_TOOL = {
    "name": "extract_compliance_data",
    "description": (
        "Record structured fields extracted from an industrial inspection field log. "
        "Per-component data (name, stated minimum, current and historical UT readings) "
        "is returned as a single flat marked-up text block, not a nested array, so a "
        "dense report with many readings does not require the model to hold open many "
        "levels of JSON nesting at once."
    ),
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
                "description": "Auto-discovered applicable governing code, e.g. ASME Sec VIII, ASME B31.3, AWS D1.1, API 653.",
            },
            "jurisdiction": {
                "type": ["string", "null"],
                "description": "Applicable regulatory jurisdiction/authority or site/client standard, only if explicitly stated.",
            },
            "confidence_metrics": {
                "type": ["object", "null"],
                "description": "Multi-axis self-assessment, replacing a single ambiguous float score.",
                "properties": {
                    "text_extraction_confidence": {"type": ["string", "null"], "enum": ["HIGH", "MEDIUM", "LOW", None]},
                    "calculation_confidence": {"type": ["string", "null"], "enum": ["HIGH", "MEDIUM", "LOW", None]},
                    "source_conflict_level": {"type": ["string", "null"], "enum": ["NONE", "LOW", "MEDIUM", "HIGH", None]},
                },
            },
            "global_minimum_allowable_thickness": {
                "type": ["object", "null"],
                "description": (
                    "Asset-level (document-wide) minimum allowable thickness, e.g. 'Minimum Allowable "
                    "Shell Thickness (MAT)', only if the text states one as a default applying to the "
                    "whole asset rather than to one specific component. This is a SHELL-CLASS default: "
                    "Python binds it to shell-class components only and NEVER to tubes (put tube and other "
                    "class-specific minimums in component_mat_table)."
                ),
                "properties": {
                    "value": {"type": ["number", "null"]},
                    "unit": {"type": ["string", "null"]},
                },
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
            "pressure_events": {
                "type": "array",
                "description": (
                    "Every distinct logged pressure excursion/startup spike/upset found anywhere in the "
                    "text, including narrative operator logs — not just tabular data. Variance is computed "
                    "in Python afterward, not by the model."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "peak_pressure_psi": {"type": ["number", "null"]},
                        "design_pressure_psi": {"type": ["number", "null"]},
                        "duration_minutes": {
                            "type": ["number", "null"],
                            "description": (
                                "How long the excursion was sustained, in minutes, only if the text "
                                "explicitly states or clearly implies a duration (e.g. a start/end "
                                "timestamp pair, 'for 20 min'). Null if not stated — an event with no "
                                "stated duration is never assumed to be either brief or sustained."
                            ),
                        },
                        "unit": {"type": ["string", "null"]},
                        "timestamp_or_context": {"type": ["string", "null"]},
                        "is_duplicate_or_continuation": {"type": "boolean"},
                        "related_event_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["event_id", "is_duplicate_or_continuation"],
                },
            },
            "component_mat_table": {
                "type": "array",
                "description": (
                    "Minimum allowable thickness per component CLASS (e.g. tube = 0.070 in, shell = "
                    "0.375 in), exactly as the text states them. Python binds each component's MAT from "
                    "this table by component_type, so a shell MAT is never applied to a tube. Every entry "
                    "requires an explicit unit."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "component_type": {"type": "string", "enum": ["shell", "tube", "channel", "flange", "head", "nozzle", "support", "plate", "other"]},
                        "value": {"type": "number"},
                        "unit": {"type": "string", "enum": ["in", "mm"]},
                        "source": {"type": ["string", "null"]},
                    },
                    "required": ["component_type", "value", "unit"],
                },
            },
            "document_registry": {
                "type": "array",
                "description": (
                    "Every distinct source document in the package with its type, date and whether it is a "
                    "signed addendum. Python uses this for precedence (a signed addendum / re-run supersedes "
                    "an earlier final report for the components it re-measured)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "document_name": {"type": "string"},
                        "document_type": {"type": "string"},
                        "document_date": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD if possible."},
                        "is_signed_addendum": {"type": "boolean"},
                        "supersedes": {"type": "array", "items": {"type": "string"}, "description": "Component identifiers the document says it supersedes, e.g. S3, S5, T3."},
                    },
                    "required": ["document_name", "document_type", "is_signed_addendum"],
                },
            },
            "flat_manifest_block": {
                "type": "string",
                "description": (
                    "Every component's data as plain marked-up text (Component: / Component_Type: / MAT: / "
                    "Unit: / MAT_*: / Attached_Reinforcement: / Reading_Set: / UT: / UT_*: / Historical_Set: / "
                    "Historical_UT: / Component_End), one block per physical component. EVERY reading and "
                    "MAT value must carry its own unit (e.g. '9.45 mm'). See system prompt for the exact "
                    "grammar. This replaces a nested JSON components array."
                ),
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
                        "risk_category": {
                            "type": "string",
                            "enum": ["FAIL_BELOW_CRITERION", "CONFLICT", "MISSING_INFORMATION", "INFORMATIONAL", "REQUIRES_VERIFICATION"],
                        },
                        "notes": {"type": ["string", "null"]},
                    },
                    "required": ["finding", "risk_category"],
                },
            },
        },
        "required": ["asset_category", "flat_manifest_block", "field_anomalies"],
    },
}


RETRY_DIRECTIVE = """

IMPORTANT — RETRY DIRECTIVE: On the previous attempt at this exact report, \
you replied with plain conversational text instead of calling the \
extract_compliance_data tool. This retry has exactly one acceptable \
outcome: call extract_compliance_data with whatever you can find in the \
report. Do not describe the report, do not ask a clarifying question, do \
not apologize or explain in text — the tool call is the entire response. \
If the report genuinely contains nothing extractable, still call the tool \
with every field left null/empty and record why in \
missing_engineering_variables_ledger — do not fall back to a text reply \
under any circumstances."""


def _request_structured_extraction(client, report_text, system_prompt):
    """One API call attempt. tool_choice is kept at {"type": "auto"} — this
    model rejects forced tool selection ({"type": "tool"} / {"type": "any"})
    with a 400 error, so forcing is not an available lever here. Returns
    (tool_input, fallback_text): tool_input is the dict of extracted fields
    if a tool_use block came back, else None; fallback_text is whatever
    plain text the model wrote instead, for diagnostics only — it is never
    used as extracted data."""
    response = client.messages.create(
        model=MODEL_NAME,
        # Kept at 6000 (raised from the pre-flattening version's 3500): a
        # single flat string field can legitimately need more room on a
        # dense report than several small nested objects did, since there's
        # no per-object JSON scaffolding splitting up the budget anymore.
        max_tokens=8000,  # raised: Reading_Set / Historical_Set lines make the manifest longer
        system=system_prompt,
        tools=[EXTRACTION_TOOL],
        tool_choice={"type": "auto"},
        messages=[{"role": "user", "content": report_text}],
    )
    fallback_text_parts = []
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_compliance_data":
            return dict(block.input), ""
        if block.type == "text":
            fallback_text_parts.append(block.text)
    return None, "\n".join(fallback_text_parts).strip()


def extract_with_claude(client, report_text):
    """Sends the raw text report to the model and returns the structured
    extraction dict. tool_choice stays {"type": "auto"} throughout — this
    model's API rejects a forced tool_choice ({"type": "tool", "name": ...}
    or {"type": "any"}) with a 400 error, so this function cannot lean on
    that lever the way an earlier version of this file tried to.

    Instead it makes up to two attempts. If the first call comes back with
    plain conversational text instead of a tool_use block, it retries once
    with the same report but a stronger, single-purpose directive appended
    to the system prompt, explicitly naming what went wrong and ruling out
    a text reply. If the retry also fails to produce a tool_use block, this
    raises rather than fabricating a structured extraction from whichever
    text the model wrote — the calling code in the main script body already
    treats that exception as a hard stop, not a silent default."""
    tool_input, fallback_text = _request_structured_extraction(client, report_text, SYSTEM_PROMPT)
    if tool_input is not None:
        return tool_input

    tool_input, retry_fallback_text = _request_structured_extraction(
        client, report_text, SYSTEM_PROMPT + RETRY_DIRECTIVE
    )
    if tool_input is not None:
        return tool_input

    detail = retry_fallback_text or fallback_text or "(model returned no text on either attempt)"
    raise RuntimeError(
        "Model did not return a structured extraction after two attempts — no tool_use block found "
        f"on the initial call or the retry. Last conversational reply from the model: {detail[:500]}"
    )

# ============================================================================
# PASS 2 — NATIVE PYTHON DISPOSITION PARSER
# (turns the model's flat "flat_manifest_block" string into the same
#  per-component structure the rest of this file expects, using plain
#  string splitting/regex only — no second model call, no JSON parsing
#  of model output, so it cannot be broken by an incomplete/truncated
#  JSON array.)
# ============================================================================

_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+")

# Shared label-classification vocabularies used by BOTH the primary
# flat-manifest parser and the fallback table reparser, so a reading is
# classified the same way regardless of which pass recovered it.
NON_BOUNDARY_LABEL_HINTS = ("plate", "pad", "reinforcement", "repair", "doubler")
PRE_PREP_LABEL_HINTS = ("pre-prep", "pre prep", "preprep", "as-found", "asfound", "before-prep", "beforeprep", "pre_prep")
POST_PREP_LABEL_HINTS = ("post-prep", "post prep", "postprep", "as-left", "asleft", "after-prep", "afterprep", "post_prep")


def _label_matches(label, hints):
    if not label:
        return False
    low = label.lower()
    return any(h in low for h in hints)


def _resolve_unit(explicit_unit, global_unit, notices, context_label):
    """Enforces the value/unit pairing rule: a numeric value is never left
    paired with a NULL unit if any unit is recoverable. Falls back to the
    document-wide global MAT unit (when available) and only returns None,
    with a visible notice, when truly nothing is recoverable — so a bare,
    un-unitized number is always surfaced as a flagged gap rather than
    silently treated as usable."""
    if explicit_unit:
        return explicit_unit
    if global_unit:
        notices.append(
            f"{context_label}: no unit was stated alongside this value — used the document-wide "
            f"global MAT unit ('{global_unit}') as a fallback."
        )
        return global_unit
    notices.append(
        f"{context_label}: a numeric value was stated with no recoverable unit anywhere in the "
        "document — flagged as unusable pending verification rather than silently treated as valid."
    )
    return None


def _coerce_float(raw_value):
    """Pulls the first numeric token out of a string; returns None if there
    isn't one. Deliberately tolerant of stray units/whitespace the model
    might still emit inline (e.g. '0.500 in')."""
    if raw_value is None:
        return None
    match = _NUMERIC_RE.search(raw_value)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


_READING_UNIT_RE = re.compile(
    r'(?P<num>[-+]?\d*\.?\d+)\s*(?P<unit>millimet\w*|mm\b|inch(?:es)?\b|in\b|")?',
    re.IGNORECASE,
)


def _normalize_unit(raw):
    """Maps any unit spelling to the canonical 'in' or 'mm'. Returns None for
    anything unrecognized — an unrecognized unit is never guessed at."""
    if raw is None:
        return None
    t = str(raw).strip().lower().strip(".")
    if t in ("in", "inch", "inches", '"', "''"):
        return "in"
    if t in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
        return "mm"
    return None


def _split_value_unit(value_str):
    """Pulls (number, canonical_unit_or_None) out of a reading token such as
    '9.45 mm', '0.072in', or '0.402"'. The unit is whatever is printed next to
    THIS number — never inherited from anywhere else."""
    m = _READING_UNIT_RE.search(value_str or "")
    if not m:
        return None, None
    try:
        return float(m.group("num")), _normalize_unit(m.group("unit"))
    except ValueError:
        return None, None


def _parse_reading_list(line_value, fallback_unit=None):
    """Parses a reading list into [{location_label, value, unit, unit_source,
    prep_phase, is_pad}] dicts. Accepts 'Label=value unit' pairs (required
    form) or bare values (auto-labeled R1, R2, ...).

    UNIT PRIORITY (row-level preservation): the unit printed on the reading
    itself always wins (unit_source='row'). Only when the row carries no unit
    does the block-level fallback_unit apply (unit_source='block_default');
    the caller may then apply a document-wide default (unit_source=
    'global_default'). Both inherited cases are flagged in parse notices and
    still pass through the unit sanity check in audit_and_reconcile."""
    readings = []
    if not line_value:
        return readings
    block_unit = _normalize_unit(fallback_unit)
    for i, part in enumerate([p.strip() for p in line_value.split(",")], start=1):
        if not part:
            continue
        if "=" in part:
            label, _, value_str = part.partition("=")
            label = label.strip() or f"R{i}"
        else:
            label, value_str = f"R{i}", part
        value, row_unit = _split_value_unit(value_str)
        if value is None:
            continue  # skip unparseable tokens rather than fabricate a reading
        if row_unit:
            unit, unit_source = row_unit, "row"
        elif block_unit:
            unit, unit_source = block_unit, "block_default"
        else:
            unit, unit_source = None, "none"
        if _label_matches(label, PRE_PREP_LABEL_HINTS):
            prep_phase = "pre"
        elif _label_matches(label, POST_PREP_LABEL_HINTS):
            prep_phase = "post"
        else:
            prep_phase = None
        readings.append({
            "location_label": label, "value": value, "unit": unit, "unit_source": unit_source,
            "prep_phase": prep_phase, "is_pad": _label_matches(label, NON_BOUNDARY_LABEL_HINTS),
        })
    return readings


def _split_non_boundary_readings(readings):
    """Splits a parsed reading list into (shell_readings, pad_readings) by
    checking each reading's own location label against the shared
    NON_BOUNDARY_LABEL_HINTS vocabulary (pad/reinforcement/plate/doubler/
    repair). A pad/reinforcement reading is evaluated separately from the
    original parent-metal shell — per API 570, a repair pad passing does
    not offset a base-shell exceedance at the same location, so the two
    must never be blended into one 'lowest reading' comparison."""
    shell, pad = [], []
    for r in readings:
        (pad if _label_matches(r.get("location_label"), NON_BOUNDARY_LABEL_HINTS) else shell).append(r)
    return shell, pad


def _parse_float_list(line_value):
    """Parses a plain comma-separated list of numbers (e.g.
    'UT_All_Values_In_Region: 18.2, 18.1, 17.8') into a list of floats,
    silently dropping any token that isn't a number."""
    if not line_value:
        return []
    values = []
    for part in line_value.split(","):
        v = _coerce_float(part)
        if v is not None:
            values.append(v)
    return values


# Line-prefix -> field name for the optional provenance/traceability lines.
# Kept as a table (rather than a long elif chain) since it's the same
# pattern repeated for MAT_* and UT_* — makes it easy to extend later.
_PROVENANCE_LINE_MAP = {
    "mat_authority_tier:": "mat_authority_tier",
    "mat_source:": "mat_source",
    "mat_general_criterion:": "mat_general_criterion",
    "mat_reason:": "mat_reason",
    "ut_authority_tier:": "ut_authority_tier",
    "ut_source:": "ut_source",
    "ut_source_date:": "ut_source_date",
    "ut_all_values_in_region:": "ut_all_values_in_region",
    "ut_reason:": "ut_reason",
}


def _split_pipe_fields(value, n):
    """Splits a pipe-delimited manifest line into exactly n fields. If the model
    supplied fewer metadata fields than expected, the LAST field (the readings)
    stays last and the missing metadata is left blank at the front."""
    parts = [p.strip() for p in (value or "").split("|", n - 1)]
    while len(parts) < n:
        parts.insert(0, "")
    return parts


_TRUTHY = ("yes", "true", "y", "1")


def _parse_single_component_block(block_text, global_mat=None, notices=None):
    """Parses one 'Component: ... ' block (already stripped of its trailing
    Component_End marker) into the component dict shape evaluate_component()
    expects. Returns None if the block has no recognizable component name.

    REFACTOR NOTES:
      * Every reading keeps the unit printed on its own row (see
        _parse_reading_list). Nothing is defaulted to a document-wide unit
        unless the row itself states none, and that case is flagged.
      * Current readings are grouped into reading_sets, one per source
        document (with document type/date/signed-addendum metadata), so
        audit_and_reconcile can apply addendum supersession. Historical
        readings are grouped into historical_sets the same way.
      * MAT is NOT propagated from the document-wide value here. Binding a
        MAT to a component happens in audit_and_reconcile, keyed on
        component_type, so a shell MAT can never leak onto a tube.
    """
    if notices is None:
        notices = []
    component_name = None
    component_type = None
    mat_value = None
    mat_unit = None
    ut_line_value = None
    historical_line_value = None
    attached_raw = None
    reading_set_lines = []
    historical_set_lines = []
    provenance_raw = {}

    for line in block_text.splitlines():
        line = line.strip()
        if not line:
            continue
        line_lower = line.lower()
        if line_lower.startswith("component_type:"):
            component_type = line.split(":", 1)[1].strip() or None
        elif line_lower.startswith("component:"):
            component_name = line.split(":", 1)[1].strip()
        elif line_lower.startswith("attached_reinforcement:"):
            attached_raw = line.split(":", 1)[1].strip()
        elif line_lower.startswith("reading_set:"):
            reading_set_lines.append(line.split(":", 1)[1])
        elif line_lower.startswith("historical_set:"):
            historical_set_lines.append(line.split(":", 1)[1])
        elif line_lower.startswith("mat_"):
            # Check the longer MAT_* provenance prefixes before the bare
            # "mat:" check below, since "mat_reason:" also starts with "mat".
            for prefix, field in _PROVENANCE_LINE_MAP.items():
                if line_lower.startswith(prefix):
                    provenance_raw[field] = line.split(":", 1)[1].strip()
                    break
        elif line_lower.startswith("mat:"):
            mat_value = _coerce_float(line.split(":", 1)[1])
        elif line_lower.startswith("unit:"):
            mat_unit = line.split(":", 1)[1].strip() or None
        elif line_lower.startswith("historical_ut:"):
            historical_line_value = line.split(":", 1)[1]
        elif line_lower.startswith("ut_"):
            for prefix, field in _PROVENANCE_LINE_MAP.items():
                if line_lower.startswith(prefix):
                    provenance_raw[field] = line.split(":", 1)[1].strip()
                    break
        elif line_lower.startswith("ut:"):
            ut_line_value = line.split(":", 1)[1]

    if not component_name:
        return None

    global_unit = _normalize_unit((global_mat or {}).get("unit"))

    # Mandatory unit-pairing rule for the stated MAT value.
    if mat_value is not None:
        mat_unit = _resolve_unit(mat_unit, (global_mat or {}).get("unit"), notices, f"'{component_name}' MAT value")

    # ---- current-cycle reading sets (one per source document) ----
    reading_sets = []
    if ut_line_value:
        reading_sets.append({
            "doc_type": provenance_raw.get("ut_authority_tier"),
            "doc_date": provenance_raw.get("ut_source_date"),
            "is_signed_addendum": False,
            "source": provenance_raw.get("ut_source"),
            "readings": _parse_reading_list(ut_line_value, mat_unit),
        })
    for raw in reading_set_lines:
        dtype, ddate, signed, source, rd = _split_pipe_fields(raw, 5)
        reading_sets.append({
            "doc_type": dtype or None,
            "doc_date": ddate or None,
            "is_signed_addendum": signed.strip().lower() in _TRUTHY,
            "source": source or None,
            "readings": _parse_reading_list(rd, mat_unit),
        })

    # ---- historical sets (one per source document/kind) ----
    historical_sets = []
    if historical_line_value:
        historical_sets.append({
            "doc_date": None, "source": None,
            "readings": _parse_reading_list(historical_line_value, mat_unit),
        })
    for raw in historical_set_lines:
        hdate, hsource, rd = _split_pipe_fields(raw, 3)
        historical_sets.append({
            "doc_date": hdate or None, "source": hsource or None,
            "readings": _parse_reading_list(rd, mat_unit),
        })

    # Unit backstop: a reading with NO recoverable unit from its own row or the
    # block gets the document-wide unit only as a flagged last resort.
    all_readings = [r for s in reading_sets + historical_sets for r in s["readings"]]
    unitless = [r for r in all_readings if not r.get("unit")]
    if unitless:
        if global_unit:
            for r in unitless:
                r["unit"], r["unit_source"] = global_unit, "global_default"
        else:
            notices.append(
                f"'{component_name}': {len(unitless)} reading(s) have no recoverable unit anywhere in the "
                "document — flagged as unusable pending verification."
            )
    inherited = [r for r in all_readings if r.get("unit_source") in ("block_default", "global_default")]
    if inherited:
        notices.append(
            f"'{component_name}': {len(inherited)} reading(s) carried no unit on their own row, so a "
            "block/document-wide default unit was inherited. The row-level unit is authoritative when "
            "present — these inherited units are still checked by the unit sanity audit."
        )

    # Legacy flat lists (kept for compatibility; audit_and_reconcile rebuilds
    # them from the sets after normalization/precedence).
    current_all = [r for s in reading_sets for r in s["readings"]]
    ut_readings, pad_readings = _split_non_boundary_readings(current_all)
    if pad_readings:
        notices.append(
            f"'{component_name}': isolated {len(pad_readings)} reinforcement/repair-pad reading(s) "
            "from the base-shell UT readings — evaluated separately per API 570."
        )
    historical_readings = [r for s in historical_sets for r in s["readings"]]

    attached = None
    if attached_raw:
        m = re.search(r'pad_thickness\s*=\s*([-+]?\d*\.?\d+)\s*(millimet\w*|mm\b|inch(?:es)?\b|in\b|")?',
                      attached_raw, re.IGNORECASE)
        if m:
            attached = {
                "pad_thickness": float(m.group(1)),
                "unit": _normalize_unit(m.group(2)) or global_unit,
                "is_pressure_retaining": bool(re.search(r"pressure_retaining\s*=\s*(true|yes)", attached_raw, re.IGNORECASE)),
            }

    mat_provenance = None
    if any(k in provenance_raw for k in ("mat_authority_tier", "mat_source", "mat_general_criterion", "mat_reason")):
        mat_provenance = {
            "authority_tier": provenance_raw.get("mat_authority_tier"),
            "source": provenance_raw.get("mat_source"),
            "general_criterion": _coerce_float(provenance_raw.get("mat_general_criterion")),
            "reason": provenance_raw.get("mat_reason"),
        }

    ut_provenance = None
    if any(k in provenance_raw for k in ("ut_authority_tier", "ut_source", "ut_source_date", "ut_all_values_in_region", "ut_reason")):
        ut_provenance = {
            "authority_tier": provenance_raw.get("ut_authority_tier"),
            "source": provenance_raw.get("ut_source"),
            "source_date": provenance_raw.get("ut_source_date"),
            "all_values_in_region": _parse_float_list(provenance_raw.get("ut_all_values_in_region")),
            "reason": provenance_raw.get("ut_reason"),
        }

    return {
        "component_name": component_name,
        "component_type": component_type,
        "explicit_minimum_required_mat": (
            {"value": mat_value, "unit": mat_unit} if mat_value is not None else None
        ),
        "mat_provenance": mat_provenance,
        "ut_provenance": ut_provenance,
        "attached_reinforcement": attached,
        "reading_sets": reading_sets,
        "historical_sets": historical_sets,
        "ut_thickness_measurements": ut_readings,
        "historical_ut_thickness_measurements": historical_readings,
        "non_pressure_boundary_measurements": pad_readings,
    }


_KEY_STRIP_WORDS = re.compile(r"\b(shell|tubes?|channel|flange|component|course)\b", re.IGNORECASE)


def _component_key(name):
    """Merge key for one PHYSICAL component. 'S3', 'Shell S3' and 's-3' all map
    to 's3'; anything that isn't a short identifier keeps its full lowercase
    name, so distinct components are never collapsed by accident."""
    raw = (name or "").strip().lower()
    stripped = re.sub(r"[^a-z0-9]", "", _KEY_STRIP_WORDS.sub("", raw))
    if re.fullmatch(r"[a-z]{1,2}\d{1,3}", stripped):
        return stripped
    return raw


def _merge_duplicate_components(components, notices):
    """Relational-linking safety net: merges component dicts that share the
    same physical-component key into ONE entry (one master record per
    component), appending readings/reading sets instead of creating duplicate
    component IDs. Order of first appearance is preserved. When more than one
    merged block states its own MAT, the first non-null one is kept; this
    function does not adjudicate MAT conflicts (MAT binding is decided later,
    by component_type, in audit_and_reconcile)."""
    merged_by_key = {}
    order = []
    list_fields = (
        "ut_thickness_measurements", "historical_ut_thickness_measurements",
        "non_pressure_boundary_measurements", "reading_sets", "historical_sets",
    )
    for comp in components:
        key = _component_key(comp.get("component_name"))
        if not key:
            order.append(comp)
            continue
        if key not in merged_by_key:
            merged_by_key[key] = comp
            order.append(comp)
            continue
        existing = merged_by_key[key]
        for f in list_fields:
            existing[f] = (existing.get(f) or []) + (comp.get(f) or [])
        if not existing.get("component_type") and comp.get("component_type"):
            existing["component_type"] = comp["component_type"]
        if not existing.get("attached_reinforcement") and comp.get("attached_reinforcement"):
            existing["attached_reinforcement"] = comp["attached_reinforcement"]
        if not existing.get("explicit_minimum_required_mat") and comp.get("explicit_minimum_required_mat"):
            existing["explicit_minimum_required_mat"] = comp["explicit_minimum_required_mat"]
            existing["mat_provenance"] = comp.get("mat_provenance")
        if not existing.get("ut_provenance") and comp.get("ut_provenance"):
            existing["ut_provenance"] = comp.get("ut_provenance")
        notices.append(
            f"Relational linking: merged a duplicate '{comp.get('component_name')}' block into the "
            "existing component entry instead of creating a second component ID (same physical "
            "component appearing in more than one table/document)."
        )
    return order


def native_parameter_matrix_parser(manifest_text, raw_report, global_mat=None):
    """Splits the model's flat_manifest_block on its own explicit boundary
    markers and rebuilds the per-component matrix in plain Python. This is
    the piece of the pipeline that is immune to the model dropping or
    truncating a nested JSON array — it only ever operates on plain text
    splitting, so a partially-cut-off block degrades to "one fewer
    component parsed" (visible in parse_notices) rather than an exception
    that blanks the whole matrix.

    `raw_report` is accepted (and currently unused directly) so this
    function's signature matches callers that want to cross-reference the
    original text in future audit passes without changing the call site.

    `global_mat` is the document-wide {"value", "unit"} dict (from the
    model's "global_minimum_allowable_thickness" field, if any) — propagated
    into any component that states no MAT of its own, and used as the unit
    fallback for a component that states a MAT value but no unit.

    Returns: (components: list[dict], parse_notices: list[str])
    """
    components_out = []
    notices = []

    if not manifest_text or not manifest_text.strip():
        notices.append(
            "The model returned an empty flat_manifest_block — no components could be parsed. "
            "Treat this extraction as unverified rather than as 'zero components found in the report'."
        )
        return components_out, notices

    # Split on the explicit end marker. Trailing content after the last
    # marker (e.g. a stray partial block from truncation) is inspected too,
    # so a cut-off final component still surfaces as a notice instead of
    # silently vanishing.
    raw_segments = manifest_text.split("Component_End")
    trailing = raw_segments[-1]
    segments = raw_segments[:-1] if len(raw_segments) > 1 else raw_segments

    for segment in segments:
        block = segment.strip()
        if not block:
            continue
        if "component:" not in block.lower():
            continue
        try:
            parsed = _parse_single_component_block(block, global_mat=global_mat, notices=notices)
        except Exception as e:  # defensive: one bad block must not blank the matrix
            parsed = None
            notices.append(f"Skipped an unparseable component block during flat-manifest parsing: {e}")
        if parsed is not None:
            components_out.append(parsed)
        else:
            notices.append("Found a 'Component:' block with no recoverable component name — skipped it.")

    if trailing.strip() and "component:" in trailing.lower():
        notices.append(
            "The flat manifest block appears to end with an unterminated component (no closing "
            "'Component_End' marker) — this may indicate output was cut short. That component was "
            "still parsed on a best-effort basis if a name and any fields were recoverable."
        )
        try:
            parsed = _parse_single_component_block(trailing.strip(), global_mat=global_mat, notices=notices)
            if parsed is not None:
                components_out.append(parsed)
        except Exception:
            pass

    if not components_out:
        notices.append(
            "No components could be parsed out of a non-empty flat_manifest_block — check the raw "
            "block in the debug expander below; the manifest may not follow the expected grammar."
        )

    # Relational-linking safety net: collapse duplicate component blocks
    # (same name appearing once per table it was mentioned in) into single
    # entries before this matrix ever reaches evaluation.
    components_out = _merge_duplicate_components(components_out, notices)

    return components_out, notices


# ============================================================================
# AUTOMATIC FALLBACK / REPARSE ENGINE
# ============================================================================
# Runs only when the primary path (model flat_manifest_block ->
# native_parameter_matrix_parser above) comes back with zero mapped
# components despite text_extraction_confidence == HIGH and the raw report
# plainly containing table-like structure — i.e. exactly the "high
# confidence, zero components mapped" failure mode this stage exists to
# recover from, rather than a genuine "nothing extractable" report. It never
# calls the model again: it is a second, independent, deterministic reading
# of the SAME raw text the primary path already had, so a truncated or
# malformed flat_manifest_block degrades to a lower-confidence recovery
# instead of an empty CONDITION UNVERIFIED result.

def raw_text_has_components_or_tables(raw_report):
    """Cheap heuristic gate for whether the fallback engine is worth trying
    at all. Intentionally permissive — a false positive just means the
    fallback engine runs and itself finds nothing to recover (harmless);
    it exists only to skip the fallback pass entirely on a report that
    plainly has no structured data (e.g. a single paragraph of prose), so
    that case is still allowed to resolve as a genuine CONDITION UNVERIFIED
    rather than being retried pointlessly."""
    if not raw_report or not raw_report.strip():
        return False
    pipe_rows = [ln for ln in raw_report.splitlines() if ln.count("|") >= 2]
    if len(pipe_rows) >= 2:
        return True
    location_like = re.findall(r"\b[A-Z]{1,3}-?\d{1,3}\b", raw_report)
    decimal_like = re.findall(r"\b\d+\.\d+\b", raw_report)
    return len(set(location_like)) >= 3 and len(decimal_like) >= 3


_FALLBACK_MIN_HEADER_HINTS = ("min", "tmin", "required", "criterion", "limit")
_FALLBACK_HISTORICAL_HEADER_HINTS = ("previous", "prior", "historical", "baseline", "last", "2024")
_FALLBACK_NON_BOUNDARY_HINTS = ("plate", "pad", "reinforcement", "repair", "doubler")
_FALLBACK_CONFIDENCE_HEADER_HINTS = ("confidence",)
_FALLBACK_SKIP_HEADER_HINTS = ("note", "notes", "comment", "remark", "status", "visual")


def _split_pipe_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator_row(cells):
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c)


def _normalize_confidence(text):
    """Maps a free-text confidence cell (e.g. 'Low', 'med.', 'HIGH conf.')
    to the app's HIGH/MEDIUM/LOW vocabulary. Returns None if the cell
    doesn't clearly say one of the three — an unrecognized cell is never
    guessed at, just left unset."""
    if not text:
        return None
    t = text.strip().lower()
    if "low" in t:
        return "LOW"
    if "med" in t:
        return "MEDIUM"
    if "high" in t:
        return "HIGH"
    return None


_INCHES_TOKEN_RE = re.compile(r'\bin(?:ches)?\b|"', re.IGNORECASE)
_MM_TOKEN_RE = re.compile(r'\bmm\b|\bmillimet(?:er|re)s?\b', re.IGNORECASE)


def _fallback_numbers_from(col_indices, cells, header_cells, default_unit=None, notices=None):
    """Pulls every decimal/integer token out of each named column, labeling
    each with its column header (plus a numbered suffix if a single cell
    contains more than one number, e.g. 'Initial 10.9 / Repeat 11.0').

    Unit handling — a value returned from this function is NEVER paired
    with a NULL unit:
      1. If the cell text or its column header contains an inches marker
         ("in", "inches", or a bare double-quote), the number is treated as
         inches and converted to millimeters (x25.4), with a conversion_note
         recording the original value/unit so nothing is silently rewritten.
      2. Else if the cell text or header explicitly says "mm"/"millimeter(s)",
         the number is kept as-is with unit "mm".
      3. Else, if a default_unit was supplied (typically the document-wide
         global MAT unit, or a unit already established elsewhere in the
         same report), that unit is used and a notice is recorded so the
         fallback is visible rather than silent.
      4. Only if none of the above apply is a reading returned with
         unit=None, and a notice is always appended for that case flagging
         it as unusable pending verification — this function never resolves
         that case by guessing a unit with no textual basis."""
    if notices is None:
        notices = []
    out = []
    for i in col_indices:
        if i >= len(cells):
            continue
        cell_text = cells[i]
        header_text = header_cells[i] if i < len(header_cells) else ""
        is_inches = bool(_INCHES_TOKEN_RE.search(cell_text) or _INCHES_TOKEN_RE.search(header_text))
        is_mm = bool(_MM_TOKEN_RE.search(cell_text) or _MM_TOKEN_RE.search(header_text))
        tokens = re.findall(r"-?\d+\.?\d*", cell_text)
        base_label = header_text.strip() if header_text.strip() else f"col{i}"
        for j, tok in enumerate(tokens):
            val = _coerce_float(tok)
            if val is None:
                continue
            conversion_note = None
            unit = None
            label = base_label if len(tokens) == 1 else f"{base_label}_{j + 1}"
            if is_inches:
                converted = round(val * 25.4, 4)
                conversion_note = f"Converted from {val:g} in to {converted:g} mm (x25.4)."
                val = converted
                unit = "mm"
            elif is_mm:
                unit = "mm"
            elif default_unit:
                unit = default_unit
                notices.append(
                    f"Fallback reparse: column '{base_label}' cell '{cell_text}' stated no unit — "
                    f"used the document-wide default unit ('{default_unit}') rather than leaving it "
                    "un-unitized."
                )
            else:
                notices.append(
                    f"Fallback reparse: column '{base_label}' cell '{cell_text}' stated no unit and no "
                    "document-wide default unit is available — flagged as unusable pending verification "
                    "rather than treated as a valid measurement."
                )
            out.append({"location_label": label, "value": val, "unit": unit, "conversion_note": conversion_note})
    return out


def execute_fallback_reparse(raw_report, global_mat=None):
    """Secondary, plain-Python recovery parser. Scans the raw report's own
    Markdown-style pipe tables directly — using each table's own header row
    to decide which column is a location/component label, which columns are
    current-cycle readings, which (if any) are prior-inspection readings,
    and which (if any) states an explicit minimum — and rebuilds component
    dicts in the same shape evaluate_component() expects.

    This is a best-effort recovery path, not a replacement for the primary
    extraction: every component it returns is tagged fallback_extracted=True,
    which downstream evaluation treats conservatively (a failing margin
    still BLOCKS; a passing margin is routed to NEEDS_HUMAN_REVIEW rather
    than confidently CLEARED — see build_evaluation_summary's via_fallback
    handling) rather than as a definitive pass/fail.

    Returns (components: list[dict], notices: list[str])."""
    components = []
    notices = []

    blocks, current = [], []
    for line in raw_report.splitlines():
        if line.count("|") >= 2:
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []
    if len(current) >= 2:
        blocks.append(current)

    if not blocks:
        notices.append(
            "Automatic fallback reparse engine ran but found no Markdown-style table structure to "
            "recover components from."
        )
        return components, notices

    for block in blocks:
        header_cells = _split_pipe_row(block[0])
        data_rows = [r for r in block[1:] if not _is_separator_row(_split_pipe_row(r))]
        if not header_cells or not data_rows:
            continue

        header_lower = [h.lower() for h in header_cells]
        label_col = 0  # first column is the component/location label by convention
        min_cols, hist_cols, reading_cols, non_boundary_cols, confidence_cols = [], [], [], [], []
        for idx, h in enumerate(header_lower):
            if idx == label_col:
                continue
            if any(hint in h for hint in _FALLBACK_SKIP_HEADER_HINTS):
                continue
            if any(hint in h for hint in _FALLBACK_CONFIDENCE_HEADER_HINTS):
                confidence_cols.append(idx)
            elif any(hint in h for hint in _FALLBACK_NON_BOUNDARY_HINTS):
                non_boundary_cols.append(idx)
            elif any(hint in h for hint in _FALLBACK_MIN_HEADER_HINTS):
                min_cols.append(idx)
            elif any(hint in h for hint in _FALLBACK_HISTORICAL_HEADER_HINTS):
                hist_cols.append(idx)
            else:
                reading_cols.append(idx)

        for row in data_rows:
            cells = _split_pipe_row(row)
            if len(cells) <= label_col or not cells[label_col]:
                continue
            component_name = cells[label_col].strip("*` ")
            if not component_name:
                continue

            # A row whose own label names a repair/reinforcement item (e.g. "A3 Reinforcement
            # Plate") is isolated the same way a dedicated column would be — its readings are
            # never pressure-boundary base-metal readings, regardless of which axis names it.
            row_is_non_boundary = any(hint in component_name.lower() for hint in _FALLBACK_NON_BOUNDARY_HINTS)

            confidence_value = None
            for ci in confidence_cols:
                if ci < len(cells):
                    confidence_value = _normalize_confidence(cells[ci]) or confidence_value

            default_unit = (global_mat or {}).get("unit")
            non_boundary_readings = _fallback_numbers_from(non_boundary_cols, cells, header_cells, default_unit, notices)
            if row_is_non_boundary:
                non_boundary_readings += _fallback_numbers_from(reading_cols, cells, header_cells, default_unit, notices)
                ut_readings = []
            else:
                ut_readings = _fallback_numbers_from(reading_cols, cells, header_cells, default_unit, notices)
            historical_readings = _fallback_numbers_from(hist_cols, cells, header_cells, default_unit, notices)
            mat_values = _fallback_numbers_from(min_cols, cells, header_cells, default_unit, notices)

            if not ut_readings and not historical_readings and not mat_values and not non_boundary_readings:
                continue  # this row carried no recoverable numeric data at all

            mat_provenance = None
            if mat_values:
                explicit_minimum_required_mat = {"value": mat_values[0]["value"], "unit": mat_values[0]["unit"]}
            else:
                explicit_minimum_required_mat = None

            components.append({
                "component_name": component_name,
                "explicit_minimum_required_mat": explicit_minimum_required_mat,
                "mat_provenance": mat_provenance,
                "ut_provenance": {
                    "authority_tier": "Supplemental UT Measurement",
                    "source": None,
                    "source_date": None,
                    "all_values_in_region": [],
                    "reason": (
                        "Recovered by the automatic fallback reparse engine directly from a raw "
                        "table structure after the primary extraction mapped zero components — "
                        "treat as lower-confidence pending verification."
                    ),
                },
                "ut_thickness_measurements": ut_readings,
                "historical_ut_thickness_measurements": historical_readings,
                "non_pressure_boundary_measurements": non_boundary_readings,
                "measurement_confidence": confidence_value,
                "fallback_extracted": True,
            })

    # Relational-linking safety net: the same component label commonly
    # appears in both a current-cycle table and a separate historical/
    # prior-year table in the raw report — merge those into one component
    # entry rather than leaving duplicate component IDs.
    components = _merge_duplicate_components(components, notices)

    if components:
        notices.append(
            f"Automatic fallback reparse engine recovered {len(components)} component(s) directly "
            "from raw table structure after the primary extraction mapped zero components. These "
            "readings are tagged as lower-confidence and routed to human review rather than treated "
            "as a definitive pass, even where the computed margin is positive."
        )
    else:
        notices.append(
            "Automatic fallback reparse engine found table structure in the raw report but could not "
            "confidently recover any component rows from it."
        )
    return components, notices


# ============================================================================
# POST-EXTRACTION SEMANTIC AUDITOR — audit_and_reconcile
# ----------------------------------------------------------------------------
# Deterministic, plain-Python pass that runs AFTER parsing and BEFORE any
# pass/fail margin is computed. In order, per component:
#   1. resolve component_type (declared by the model, else inferred from name)
#   2. normalize every reading to dual units (in + mm) and run the unit sanity
#      check (a tube reading > 0.500 in is mm-misidentified -> auto-convert)
#   3. apply document precedence (signed addendum / re-run supersedes the
#      final report) and archive superseded readings
#   4. link any repair pad as non-pressure-retaining reinforcement, never
#      letting pad thickness govern a pressure calculation
#   5. reconcile historical records by physical equivalence (<= 1% after unit
#      conversion) so in/mm duplicates collapse into one lineage entry
#   6. bind the governing MAT by component_type (tube MAT never = shell MAT)
# evaluate() then recomputes every margin from these normalized values.
# ============================================================================

MM_PER_IN = 25.4
TUBE_MAX_PLAUSIBLE_IN = 0.500     # any tube wall reading above this is suspect
OTHER_MAX_PLAUSIBLE_IN = 4.000    # generic ceiling for non-tube components
MIN_PLAUSIBLE_MM = 0.25           # an "mm" value below this is really inches
EQUIVALENCE_TOLERANCE = 0.01      # <= 1% variance after unit conversion
PAD_TOLERANCE_IN = 0.0005
KNOWN_COMPONENT_TYPES = ("shell", "tube", "channel", "flange", "head", "nozzle", "support", "plate", "other")
NO_GLOBAL_MAT_TYPES = ("tube",)   # the document-wide (shell) MAT never binds to these

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _to_in_mm(value, unit):
    if unit == "in":
        return value, value * MM_PER_IN
    if unit == "mm":
        return value / MM_PER_IN, value
    return None, None


def _infer_component_type(name, declared=None):
    """Returns (type, how). A type declared by the extraction wins; otherwise
    it is inferred from the component name."""
    d = (declared or "").strip().lower()
    if d.endswith("s") and d[:-1] in KNOWN_COMPONENT_TYPES:
        d = d[:-1]
    if d in KNOWN_COMPONENT_TYPES:
        return d, "declared"
    n = (name or "").lower()
    if re.search(r"\btubes?\b|\bcoil\b|\bbundle\b", n) or re.fullmatch(r"\s*t-?\d{1,3}\s*", n):
        return "tube", "name"
    for word in ("channel", "flange", "head", "nozzle"):
        if word in n:
            return word, "name"
    if re.search(r"\bshell\b|\bcourse\b", n) or re.fullmatch(r"\s*s-?\d{1,3}\s*", n):
        return "shell", "name"
    return "other", "name"


def _component_id(name):
    m = re.search(r"\b([A-Za-z]{1,2})-?(\d{1,3})\b", name or "")
    return f"{m.group(1).upper()}{m.group(2)}" if m else (name or "UNNAMED")


def _parse_doc_date(text):
    """Best-effort date -> sortable (year, month, day) tuple; None if unreadable.
    A missing year sorts as 0 (only same-year ordering is assumed)."""
    if not text:
        return None
    t = str(text).strip().lower()
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?", t)
    if m:
        return int(m.group(3) or 0), _MONTHS[m.group(1)], int(m.group(2))
    m = re.fullmatch(r"\s*(\d{4})\s*", t)
    if m:
        return int(m.group(1)), 0, 0
    return None


def _date_key(s):
    return _parse_doc_date(s.get("doc_date")) or (-1, -1, -1)


def _doc_tier(s):
    """2 = signed addendum or re-run (supersedes earlier readings);
    1 = everything else (final report, worksheet, notes...)."""
    text = f"{s.get('doc_type') or ''} {s.get('source') or ''}".lower()
    signed_addendum = bool(s.get("is_signed_addendum")) or ("addendum" in text and "signed" in text)
    rerun = bool(re.search(r"re-?run|re-?test|re-?inspect|re-?shoot|re-?measure", text))
    return 2 if (signed_addendum or rerun) else 1


def _doc_identity(s):
    return ((s.get("doc_type") or "").lower(), (s.get("doc_date") or "").lower(), (s.get("source") or "").lower())


def _source_label(s):
    return s.get("source") or s.get("doc_type") or "Unnamed source"


def _enrich_from_registry(s, registry):
    src = (s.get("source") or "").strip().lower()
    if len(src) < 4 or not registry:
        return
    for doc in registry:
        name = (doc.get("document_name") or "").strip().lower()
        if name and (name in src or src in name):
            s["doc_type"] = s.get("doc_type") or doc.get("document_type")
            s["doc_date"] = s.get("doc_date") or doc.get("document_date")
            if doc.get("is_signed_addendum"):
                s["is_signed_addendum"] = True
            return


def _ensure_sets(comp):
    """Returns (reading_sets, historical_sets) as deep-ish copies. Components
    that never went through the flat-manifest parser (e.g. fallback-reparsed
    table rows) get their flat lists wrapped into a single default set."""
    sets = [dict(s, readings=[dict(r) for r in s.get("readings") or []]) for s in comp.get("reading_sets") or []]
    if not sets:
        prov = comp.get("ut_provenance") or {}
        cur = [dict(r) for r in comp.get("ut_thickness_measurements") or []]
        pads = [dict(r, is_pad=True) for r in comp.get("non_pressure_boundary_measurements") or []]
        if cur or pads:
            sets.append({"doc_type": prov.get("authority_tier"), "doc_date": prov.get("source_date"),
                         "is_signed_addendum": False, "source": prov.get("source"), "readings": cur + pads})
    hsets = [dict(s, readings=[dict(r) for r in s.get("readings") or []]) for s in comp.get("historical_sets") or []]
    if not hsets:
        hist = [dict(r) for r in comp.get("historical_ut_thickness_measurements") or []]
        if hist:
            hsets.append({"doc_date": None, "source": None, "readings": hist})
    return sets, hsets


def _normalize_reading(r, comp_type):
    """Normalizes one reading to dual units and runs the unit sanity check.
    Mutates r. Returns (status, message) with status OK / CORRECTED / REVIEW.
    A REVIEW reading is never given an inch value, so it cannot reach a margin."""
    raw_value = r.get("value")
    unit = _normalize_unit(r.get("unit"))
    label = r.get("location_label")
    r["value_raw"] = raw_value
    r["unit_as_extracted"] = r.get("unit")
    if raw_value is None or unit is None:
        r["unit_status"] = "REVIEW"
        return "REVIEW", (f"reading '{label}' = {raw_value} has no recognizable unit — excluded from margin "
                          "calculations pending unit review.")
    v_in, v_mm = _to_in_mm(raw_value, unit)
    limit = TUBE_MAX_PLAUSIBLE_IN if comp_type == "tube" else OTHER_MAX_PLAUSIBLE_IN
    status, message = "OK", None
    if v_in > limit:
        if comp_type == "tube" and unit == "in" and raw_value / MM_PER_IN <= limit:
            unit = "mm"
            v_in, v_mm = _to_in_mm(raw_value, "mm")
            status = "CORRECTED"
            message = (f"reading '{label}' = {raw_value:g} was tagged inches but exceeds the {limit:.3f} in "
                       f"plausibility ceiling for a tube — reinterpreted as {raw_value:g} mm ({v_in:.5f} in).")
        else:
            r["unit_status"] = "REVIEW"
            return "REVIEW", (f"reading '{label}' = {raw_value:g} {unit} ({v_in:.3f} in) exceeds the {limit:.3f} in "
                              f"plausibility ceiling for a {comp_type} and could not be safely auto-corrected — "
                              "excluded from margin calculations pending unit review.")
    elif unit == "mm" and raw_value < MIN_PLAUSIBLE_MM:
        r["unit_status"] = "REVIEW"
        return "REVIEW", (f"reading '{label}' = {raw_value:g} mm is implausibly thin for a wall measurement "
                          "(likely inches mislabeled as mm) — excluded pending unit review.")
    r["value"] = round(v_in, 8)
    r["unit"] = "in"
    r["value_in"] = round(v_in, 8)
    r["value_mm"] = round(v_mm, 4)
    r["unit_stated"] = unit
    r["unit_status"] = "CORRECTED_MM" if status == "CORRECTED" else "OK"
    return status, message


def _mat_to_in(value, unit, comp_type):
    """Converts a MAT value to inches, with the same tube-unit sanity rule."""
    u = _normalize_unit(unit)
    if value is None or u is None:
        return None, "MAT value has no recognizable unit"
    v_in, _ = _to_in_mm(value, u)
    if comp_type == "tube" and u == "in" and v_in > TUBE_MAX_PLAUSIBLE_IN and value / MM_PER_IN <= TUBE_MAX_PLAUSIBLE_IN:
        return value / MM_PER_IN, f"MAT {value:g} 'in' exceeds the tube ceiling — reinterpreted as {value:g} mm"
    return v_in, None


def _apply_precedence(sets):
    """Signed addendum / re-run supersedes earlier documents for this component.
    Returns (active_sets, superseded_sets, info_notes, conflict_notes)."""
    live = [s for s in sets if s.get("readings")]
    info, conflicts = [], []
    if not live:
        return [], [], info, conflicts
    for s in live:
        s["tier"] = _doc_tier(s)
    tier2 = [s for s in live if s["tier"] == 2]
    if not tier2:
        if len({_doc_identity(s) for s in live}) > 1:
            info.append("current readings came from more than one document with no signed addendum or "
                        "re-run — all retained; the lowest reading governs conservatively.")
        return live, [], info, conflicts
    best = max(_date_key(s) for s in tier2)
    winners = [s for s in tier2 if _date_key(s) == best]
    active, superseded = list(winners), []
    for s in live:
        if s in winners:
            continue
        if s["tier"] == 2 or _date_key(s) <= best:
            superseded.append(s)
        else:
            active.append(s)
            conflicts.append(f"'{_source_label(s)}' is dated after the signed addendum/re-run but is not itself "
                             "a signed addendum or re-run — retained as active; confirm which governs.")
    if superseded:
        n = sum(len(s["readings"]) for s in superseded)
        info.append(f"'{_source_label(winners[0])}' (signed addendum/re-run) supersedes {n} earlier reading(s) from "
                    f"{', '.join(sorted({_source_label(s) for s in superseded}))}; archived in superseded_readings.")
    return active, superseded, info, conflicts


def _archive_record(r, s):
    return {"location_label": r.get("location_label"), "value": r.get("value_raw"), "unit": r.get("unit_stated"),
            "value_in": r.get("value_in"), "value_mm": r.get("value_mm"),
            "source_document": _source_label(s), "document_date": s.get("doc_date"),
            "document_type": s.get("doc_type")}


def _labels_compatible(a, b):
    na, nb = re.sub(r"[^a-z0-9]", "", (a or "").lower()), re.sub(r"[^a-z0-9]", "", (b or "").lower())
    auto = lambda x: x == "" or re.fullmatch(r"r\d+", x) is not None
    return na == nb or auto(na) or auto(nb)


def _reconcile_history(entries):
    """Collapses historical entries that are the SAME physical reading recorded
    in different units/sources (<= 1% variance after conversion, same date,
    compatible location label) into one lineage entry. Returns
    (lineage, legacy_readings, notes)."""
    def authority(e):
        src = (e["source"] or "").lower()
        return 1 if re.search(r"spreadsheet|worksheet|xls|csv", src) else 0
    clusters, notes = [], []
    for e in sorted(entries, key=authority):
        placed = False
        for c in clusters:
            rep = c["members"][0]
            same_date = (c["date"] or "") == (e["date"] or "")
            close = abs(e["r"]["value_in"] - rep["r"]["value_in"]) / max(rep["r"]["value_in"], 1e-9) <= EQUIVALENCE_TOLERANCE
            if same_date and close and _labels_compatible(e["r"].get("location_label"), rep["r"].get("location_label")):
                c["members"].append(e)
                placed = True
                break
        if not placed:
            clusters.append({"date": e["date"], "members": [e]})
    lineage, legacy = [], []
    for c in clusters:
        rep = c["members"][0]["r"]
        item = {"date": c["date"], "location_label": rep.get("location_label"),
                "reconciled_value_in": round(rep["value_in"], 5), "reconciled_value_mm": round(rep["value_mm"], 3),
                "members": [{"value": m["r"].get("value_raw"), "unit": m["r"].get("unit_stated"),
                             "value_in": m["r"]["value_in"], "value_mm": m["r"]["value_mm"], "source": m["source"]}
                            for m in c["members"]]}
        for m in c["members"]:
            src = (m["source"] or "").lower()
            kind = "spreadsheet" if re.search(r"spreadsheet|worksheet|xls|csv", src) else "report"
            item.setdefault(f"source_{kind}_value", m["r"].get("value_raw"))
            item.setdefault(f"source_{kind}_unit", m["r"].get("unit_stated"))
        if len(c["members"]) > 1:
            a, b = c["members"][0]["r"], c["members"][1]["r"]
            pct = abs(a["value_in"] - b["value_in"]) / max(a["value_in"], 1e-9) * 100
            item["note"] = (f"{a['value_raw']:g} {a['unit_stated']} and {b['value_raw']:g} {b['unit_stated']} are "
                            f"equivalent within rounding tolerance ({pct:.2f}% variance after unit conversion).")
        else:
            item["note"] = "Single source record."
        lineage.append(item)
        legacy.append({"location_label": rep.get("location_label"), "value": rep["value_in"], "unit": "in",
                       "value_in": rep["value_in"], "value_mm": rep["value_mm"]})
    return lineage, legacy, notes


def audit_and_reconcile(components, extracted):
    """Deterministic post-extraction semantic auditor. Mutates each component
    into its reconciled form and returns (components, notices). See the module
    banner above for the ordered steps. Margins are NOT computed here — they
    are recomputed by evaluate() only after this pass has finished."""
    notices = []
    registry = extracted.get("document_registry") or []
    gm = extracted.get("global_minimum_allowable_thickness") or {}
    g_in, g_note = _mat_to_in(gm.get("value"), gm.get("unit"), "shell") if gm.get("value") is not None else (None, None)
    type_mat = {}
    for row in extracted.get("component_mat_table") or []:
        t = (row.get("component_type") or "").strip().lower()
        if t.endswith("s") and t[:-1] in KNOWN_COMPONENT_TYPES:
            t = t[:-1]
        v_in, note = _mat_to_in(row.get("value"), row.get("unit"), t)
        if t in KNOWN_COMPONENT_TYPES and v_in is not None:
            type_mat[t] = {"value_in": v_in, "source": row.get("source")}
            if note:
                notices.append(f"MAT table ({t}): {note}.")

    for comp in components:
        name = comp.get("component_name") or "Unnamed Component"
        ctype, how = _infer_component_type(name, comp.get("component_type"))
        comp["component_type"] = ctype
        comp["component_identifier"] = _component_id(name)
        if how == "name":
            notices.append(f"'{name}': component_type was not declared by the extraction — inferred '{ctype}' from the component name.")
        review_notes, correction_notes, precedence_notes = [], [], []

        sets, hsets = _ensure_sets(comp)
        for s in sets + hsets:
            _enrich_from_registry(s, registry)

        # ---- 2. unit normalization + sanity (current readings) ----
        for s in sets:
            kept = []
            for r in s["readings"]:
                r["is_pad"] = bool(r.get("is_pad")) or _label_matches(r.get("location_label"), NON_BOUNDARY_LABEL_HINTS)
                status, msg = _normalize_reading(r, ctype)
                if status == "REVIEW":
                    review_notes.append(msg)
                    comp.setdefault("unit_review_readings", []).append(
                        {"location_label": r.get("location_label"), "value": r.get("value_raw"),
                         "unit": r.get("unit_as_extracted"), "source_document": _source_label(s)})
                    continue
                if status == "CORRECTED":
                    correction_notes.append(msg)
                kept.append(r)
            s["readings"] = kept

        # ---- 3. precedence / supersession ----
        active_sets, superseded_sets, info, conflicts = _apply_precedence(sets)
        precedence_notes += info
        review_notes += conflicts
        for s in active_sets:
            for r in s["readings"]:
                r.update({"source_document": _source_label(s), "document_date": s.get("doc_date"),
                          "document_type": s.get("doc_type"), "is_signed_addendum": s.get("tier") == 2})
        active_readings = [r for s in active_sets for r in s["readings"]]
        superseded_records = [_archive_record(r, s) for s in superseded_sets for r in s["readings"]]

        # ---- 4. repair-pad linkage ----
        base = [r for r in active_readings if not r["is_pad"]]
        pads = [r for r in active_readings if r["is_pad"]]
        att = comp.get("attached_reinforcement")
        pad_in = None
        if att and att.get("pad_thickness") is not None:
            pu = _normalize_unit(att.get("unit"))
            pad_in = _to_in_mm(att["pad_thickness"], pu)[0] if pu else None
            if att.get("is_pressure_retaining"):
                review_notes.append("extraction marked the attached pad as pressure-retaining — overridden: a "
                                    "non-replacement reinforcement pad is never credited as pressure-retaining.")
        elif pads:
            pad_in = min(r["value_in"] for r in pads)
        if pad_in is not None:
            leaked = [r for r in base if abs(r["value_in"] - pad_in) <= PAD_TOLERANCE_IN]
            if leaked:
                base = [r for r in base if r not in leaked]
                pads += leaked
                review_notes.append(f"{len(leaked)} base-shell reading(s) matched the reinforcement-pad thickness "
                                    f"({pad_in:.4f} in) and were moved out of the pressure calculation.")
            comp["attached_reinforcement"] = {
                "pad_thickness": round(pad_in, 5), "pad_thickness_mm": round(pad_in * MM_PER_IN, 3),
                "unit": "in", "is_pressure_retaining": False,
                "note": "Non-replacement reinforcement: pressure calculations use the base pressure-retaining shell only.",
            }
        else:
            comp["attached_reinforcement"] = None

        # ---- 5. historical reconciliation ----
        entries = []
        for hs in hsets:
            for r in hs["readings"]:
                status, msg = _normalize_reading(r, ctype)
                if status == "REVIEW":
                    review_notes.append(f"historical {msg}")
                    continue
                if status == "CORRECTED":
                    correction_notes.append(f"historical {msg}")
                entries.append({"date": hs.get("doc_date"), "source": hs.get("source"), "r": r})
        lineage, hist_legacy, _ = _reconcile_history(entries)

        # ---- 6. MAT binding by component_type ----
        own = comp.get("explicit_minimum_required_mat")
        mat_in, binding = None, None
        if own and own.get("value") is not None:
            own_in, note = _mat_to_in(own["value"], own.get("unit"), ctype)
            if note:
                correction_notes.append(note)
            if (ctype == "tube" and own_in is not None and g_in and ctype in type_mat
                    and abs(own_in - g_in) / g_in <= EQUIVALENCE_TOLERANCE
                    and abs(type_mat[ctype]["value_in"] - g_in) / g_in > EQUIVALENCE_TOLERANCE):
                mat_in = type_mat[ctype]["value_in"]
                binding = {"source": "component-type MAT table (overrode a value equal to the shell MAT)",
                           "detail": type_mat[ctype].get("source")}
                review_notes.append(f"stated MAT ({own_in:.4f} in) equals the shell MAT — replaced with the tube MAT "
                                    f"({mat_in:.4f} in).")
            elif own_in is not None:
                mat_in, binding = own_in, {"source": "component-specific MAT", "detail": (comp.get("mat_provenance") or {}).get("source")}
        if mat_in is None and ctype in type_mat:
            mat_in = type_mat[ctype]["value_in"]
            binding = {"source": f"{ctype} MAT from component-type table", "detail": type_mat[ctype].get("source")}
        if mat_in is None and g_in is not None and ctype not in NO_GLOBAL_MAT_TYPES:
            mat_in = g_in
            binding = {"source": "document-wide MAT (shell-class default)", "detail": None}
        if mat_in is None and ctype in NO_GLOBAL_MAT_TYPES and g_in is not None:
            notices.append(f"'{name}': no tube-specific MAT was found — the document-wide shell MAT ({g_in:.4f} in) "
                           "was deliberately NOT applied to this tube.")
        if binding is not None:
            binding["component_type"] = ctype
            binding["value_in"] = round(mat_in, 5)
        else:
            binding = {"source": "none bound", "component_type": ctype, "value_in": None, "detail": None}
        comp["explicit_minimum_required_mat"] = {"value": round(mat_in, 5), "unit": "in"} if mat_in is not None else None
        comp["mat_binding"] = binding
        if binding["source"] != "component-specific MAT" and mat_in is not None:
            comp["mat_provenance"] = {"authority_tier": None, "source": binding["source"],
                                      "general_criterion": None,
                                      "reason": f"MAT bound by component_type='{ctype}' (never a shell MAT on a tube)."}

        # ---- write reconciled data back ----
        comp["ut_thickness_measurements"] = base
        comp["non_pressure_boundary_measurements"] = pads
        comp["historical_ut_thickness_measurements"] = hist_legacy
        comp["superseded_readings"] = superseded_records
        comp["historical_lineage"] = lineage
        comp["review_notes"] = review_notes
        comp["correction_notes"] = correction_notes
        comp["precedence_notes"] = precedence_notes
        comp["governing_document"] = (
            {"source_document": _source_label(active_sets[0]), "document_date": active_sets[0].get("doc_date"),
             "document_type": active_sets[0].get("doc_type"), "tier": active_sets[0].get("tier")}
            if active_sets else None)
        for msg in review_notes + correction_notes + precedence_notes:
            notices.append(f"'{name}': {msg}")
    return components, notices


def build_master_records(components, results):
    """One consolidated master record per physical component: identity, type,
    bound MAT, active governing reading, superseded lineage, historical
    lineage, and the verified calculation (taken from evaluate(), i.e. computed
    only after reconciliation)."""
    records = []
    for comp, res in zip(components, results):
        lowest = res.get("lowest")
        active = comp.get("ut_thickness_measurements") or []
        gdoc = comp.get("governing_document") or {}
        summary = res.get("evaluation_summary") or {}
        workflow = summary.get("workflow_status")
        corrected = bool(comp.get("correction_notes"))
        if res["status"] in ("blocked",):
            pass_fail, validity = "FAIL", "FAILED_BELOW_MAT"
        elif workflow == "CLEARED":
            pass_fail, validity = "PASS", ("VERIFIED_WITH_UNIT_CORRECTION" if corrected else "VERIFIED_SECURE")
        elif comp.get("unit_review_readings") and not active:
            pass_fail, validity = "REVIEW", "UNIT_REVIEW_REQUIRED"
        elif res["status"] in ("insufficient", "no_measurements", "no_criteria_no_measurements"):
            pass_fail, validity = "INSUFFICIENT_DATA", "INSUFFICIENT_DATA"
        else:
            pass_fail, validity = "REVIEW", "REQUIRES_REVIEW"

        active_reading = None
        if active:
            active_reading = {
                "raw_values": [r.get("value_raw") for r in active],
                "selected_governing_value_mm": lowest.get("value_mm") if lowest else None,
                "selected_governing_value_in": round(lowest["value_in"], 5) if lowest else None,
                "unit": lowest.get("unit_stated") if lowest else None,
                "source_document": gdoc.get("source_document"),
                "document_date": gdoc.get("document_date"),
                "document_type": gdoc.get("document_type"),
                "is_signed_addendum": gdoc.get("tier") == 2,
                "is_superseding": bool(comp.get("superseded_readings")) and gdoc.get("tier") == 2,
            }
        mat = res.get("mat")
        margin = res.get("margin")
        records.append({
            "component_identifier": comp.get("component_identifier") or comp.get("component_name"),
            "component_name": comp.get("component_name"),
            "component_type": comp.get("component_type"),
            "governing_mat": {"value": round(mat, 5), "unit": "in"} if mat is not None else None,
            "mat_binding": comp.get("mat_binding"),
            "active_governing_reading": active_reading,
            "attached_reinforcement": comp.get("attached_reinforcement"),
            "superseded_readings": comp.get("superseded_readings") or [],
            "historical_lineage": comp.get("historical_lineage") or [],
            "unit_review_readings": comp.get("unit_review_readings") or [],
            "engineering_calculations": {
                "lowest_active_reading_in": round(lowest["value_in"], 5) if lowest else None,
                "governing_mat_in": round(mat, 5) if mat is not None else None,
                "true_margin_in": round(margin, 5) if margin is not None else None,
                "pass_fail_status": pass_fail,
                "calculation_validity": validity,
            },
        })
    return records


def _fmt_reading(r):
    """'label: 0.3772 in (9.58 mm)' — dual-unit display of one reading."""
    u = r.get("unit") or ""
    s = f"{r.get('location_label')}: {r['value']:.4f} {u}".rstrip()
    if u == "in" and r.get("value_mm") is not None:
        s += f" ({r['value_mm']:.2f} mm)"
    return s


def render_master_records(records):
    st.markdown("#### Consolidated Component Master Records")
    st.caption("One master record per physical component: bound MAT, active governing reading, superseded "
               "readings, historical lineage, and the calculation recomputed after unit normalization.")
    for rec in records:
        calc = rec["engineering_calculations"]
        with st.expander(f"{rec['component_identifier']} ({rec['component_type']}) — {calc['pass_fail_status']} · {calc['calculation_validity']}"):
            st.json(rec)
    st.download_button("Download master records (JSON)", json.dumps(records, indent=2, default=str),
                       file_name="component_master_records.json", mime="application/json")


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


def build_evaluation_summary(status, name, lowest=None, mat=None, mat_unit=None, margin=None,
                              via_fallback=False, low_confidence=False, pad_flag=False,
                              pre_reading=None, post_reading=None):
    """Decouples the mathematical calculation result from the system's
    workflow status, per the requirement that a workflow label (BLOCKED,
    etc.) must never be presented as if it were itself an engineering
    determination. Both fields, and the explanation joining them, are built
    here in plain Python from numbers already computed elsewhere — never
    asked of the model.

    via_fallback marks a component whose readings were recovered by the
    automatic fallback reparse engine (see execute_fallback_reparse) rather
    than the primary structured extraction. low_confidence marks a
    component the source text itself tagged LOW measurement confidence
    (e.g. a table's own "Confidence" column). Either condition triggers the
    same conservative handling: a BLOCKED result stays BLOCKED (a fail is
    never softened just because the read is lower-confidence), but a
    passing result is downgraded to NEEDS_HUMAN_REVIEW rather than
    confidently CLEARED, since a false "pass" from an unverified or
    low-confidence read is the more dangerous failure mode to risk."""
    unconfirmed = via_fallback or low_confidence
    if status == "blocked":
        calculation_result = "BELOW_PROVIDED_MINIMUM"
        workflow_status = "BLOCKED"
        unit = f" {mat_unit}" if mat_unit else ""
        cause = []
        if via_fallback:
            cause.append("recovered by the automatic fallback reparse engine rather than the primary extraction")
        if low_confidence:
            cause.append("tagged LOW measurement confidence in the source text")
        fallback_note = (
            f" This reading was {' and '.join(cause)} — the BLOCKED status is retained regardless, "
            "since a confirmed exceedance is never softened on the strength of a lower-confidence read."
        ) if cause else ""
        pad_note = (
            " A reinforcement/repair pad reading is also on file for this location, evaluated "
            "separately per API 570 — the pad reading does not offset this base-shell exceedance, and "
            "a separate engineering assessment of the pad under API 570 is required."
        ) if pad_flag else ""
        status_explanation = (
            f"Mathematical result is BELOW_PROVIDED_MINIMUM ({name} measured {lowest['value']:.4f}{unit} "
            f"vs. {mat:.4f}{unit} required minimum, margin {margin:+.4f}). System workflow set to BLOCKED "
            f"per safety policy — this is a workflow decision, not itself an engineering disposition."
            f"{fallback_note}{pad_note}"
        )
        risk_category = "FAIL_BELOW_CRITERION"
    elif status == "requires_verification_prep":
        unit = f" {mat_unit}" if mat_unit else ""
        calculation_result = "REQUIRES_VERIFICATION_SURFACE_PREP_RETEST"
        workflow_status = "REQUIRES_VERIFICATION"
        status_explanation = (
            f"{name}: the pre-surface-prep reading ({pre_reading['location_label']}="
            f"{pre_reading['value']:.4f}{unit}) fell below the required minimum ({mat:.4f}{unit}), but "
            f"the post-surface-prep reading ({post_reading['location_label']}={post_reading['value']:.4f}"
            f"{unit}) is at or above it. This is neither auto-passed nor auto-failed: a human reviewer "
            "must confirm whether the pre-prep reading reflected surface scale/coating removed by "
            "prep (in which case the post-prep reading governs) or a return to base-metal at a "
            "thinner point (in which case it does not) before this location can be cleared."
        )
        risk_category = "REQUIRES_VERIFICATION"
    elif status == "verified":
        unit = f" {mat_unit}" if mat_unit else ""
        if unconfirmed:
            # A passing margin from a lower-confidence or fallback-recovered reading is not
            # treated as a definitive pass — route it to human review instead of clearing it.
            cause = []
            if via_fallback:
                cause.append("came from the automatic fallback reparse engine rather than the primary extraction")
            if low_confidence:
                cause.append("is tagged LOW measurement confidence in the source text")
            calculation_result = "WITHIN_SPEC_UNCONFIRMED"
            workflow_status = "NEEDS_HUMAN_REVIEW"
            status_explanation = (
                f"Mathematical result is WITHIN_SPEC ({name} measured {lowest['value']:.4f}{unit} vs. "
                f"{mat:.4f}{unit} required minimum, margin {margin:+.4f}), but this reading "
                f"{' and '.join(cause)}. System workflow set to NEEDS_HUMAN_REVIEW rather than "
                f"CLEARED — a passing margin from a lower-confidence read is not treated as a "
                f"definitive pass."
            )
            risk_category = "REQUIRES_VERIFICATION"
        else:
            calculation_result = "WITHIN_SPEC"
            workflow_status = "CLEARED"
            status_explanation = (
                f"Mathematical result is WITHIN_SPEC ({name} measured {lowest['value']:.4f}{unit} vs. "
                f"{mat:.4f}{unit} required minimum, margin {margin:+.4f}). System workflow set to CLEARED."
            )
            risk_category = None
    else:  # insufficient / no_measurements / no_criteria_no_measurements
        calculation_result = "INSUFFICIENT_DATA"
        workflow_status = "NEEDS_HUMAN_REVIEW"
        status_explanation = (
            f"Mathematical result is INSUFFICIENT_DATA for {name} — a margin cannot be computed from "
            f"what the source text provides. System workflow set to NEEDS_HUMAN_REVIEW, not BLOCKED, "
            f"since this reflects a data gap rather than a confirmed exceedance."
        )
        risk_category = "MISSING_INFORMATION"
    return {
        "calculation_result": calculation_result,
        "workflow_status": workflow_status,
        "status_explanation": status_explanation,
        "risk_category": risk_category,
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
    mat_provenance = component.get("mat_provenance")
    ut_provenance = component.get("ut_provenance")

    if not ut_list and component.get("unit_review_readings"):
        missing = ["Verified unit for reading(s): " + ", ".join(
            f"{r.get('location_label')}={r.get('value')} {r.get('unit') or '(no unit)'}"
            for r in component["unit_review_readings"])]
        evaluation_summary = build_evaluation_summary("insufficient", name)
        return {"name": name, "status": "insufficient", "missing_vars": missing,
                 "ut_measurements": [], "mat_provenance": mat_provenance,
                 "ut_provenance": ut_provenance, "degradation": degradation,
                 "evaluation_summary": evaluation_summary, "raw": component}

    if not ut_list:
        # Distinguish "we have a stated minimum but nothing to test it against"
        # from "we have neither criteria nor measurements at all" — the two
        # are engineering-critical to tell apart in the UI.
        status = "no_measurements" if mat is not None else "no_criteria_no_measurements"
        evaluation_summary = build_evaluation_summary(status, name)
        return {"name": name, "status": status, "mat": mat, "mat_unit": mat_unit,
                 "mat_provenance": mat_provenance, "ut_provenance": ut_provenance,
                 "degradation": degradation, "evaluation_summary": evaluation_summary, "raw": component}

    calc_note = None
    if mat is None:
        if is_piping:
            mat, missing, calc_note = calc_b31_3_mat(piping_vars)
            if mat is None:
                evaluation_summary = build_evaluation_summary("insufficient", name)
                return {"name": name, "status": "insufficient", "missing_vars": missing,
                         "ut_measurements": ut_list, "mat_provenance": mat_provenance,
                         "ut_provenance": ut_provenance, "degradation": degradation,
                         "evaluation_summary": evaluation_summary, "raw": component}
        else:
            evaluation_summary = build_evaluation_summary("insufficient", name)
            return {"name": name, "status": "insufficient",
                     "missing_vars": ["Explicit Minimum Required MAT (not stated for this component)"],
                     "ut_measurements": ut_list, "mat_provenance": mat_provenance,
                     "ut_provenance": ut_provenance, "degradation": degradation,
                     "evaluation_summary": evaluation_summary, "raw": component}

    non_boundary_readings = component.get("non_pressure_boundary_measurements") or []
    via_fallback = bool(component.get("fallback_extracted"))
    low_confidence = (component.get("measurement_confidence") or "").upper() == "LOW"

    # Surface-prep re-test workflow: a component whose current readings
    # include BOTH a pre-prep and a post-prep sample for the same visit is
    # never silently auto-passed (on the post-prep number) or auto-failed
    # (on the pre-prep number). Only when the pre-prep reading is genuinely
    # below MAT and the post-prep reading clears it does this override fire
    # — otherwise (e.g. both fail, or both pass) the normal lowest-reading
    # comparison below already gives the right, conservative answer.
    pre_readings = [r for r in ut_list if r.get("prep_phase") == "pre"]
    post_readings = [r for r in ut_list if r.get("prep_phase") == "post"]
    if pre_readings and post_readings:
        lowest_pre = min(pre_readings, key=lambda r: r["value"])
        lowest_post = min(post_readings, key=lambda r: r["value"])
        if lowest_pre["value"] < mat <= lowest_post["value"]:
            evaluation_summary = build_evaluation_summary(
                "requires_verification_prep", name, mat=mat, mat_unit=mat_unit,
                pre_reading=lowest_pre, post_reading=lowest_post,
            )
            return {"name": name, "status": "requires_verification_prep", "mat": mat, "mat_unit": mat_unit,
                     "lowest": lowest_pre, "pre_reading": lowest_pre, "post_reading": lowest_post,
                     "mat_provenance": mat_provenance, "ut_provenance": ut_provenance,
                     "degradation": degradation, "via_fallback": via_fallback, "low_confidence": low_confidence,
                     "non_boundary_readings": non_boundary_readings,
                     "evaluation_summary": evaluation_summary, "raw": component}

    lowest = min(ut_list, key=lambda r: r["value"])
    margin = round(lowest["value"] - mat, 5)  # values are normalized inches at this point
    status = "blocked" if margin < 0 else "verified"
    pad_flag = status == "blocked" and bool(non_boundary_readings)
    evaluation_summary = build_evaluation_summary(status, name, lowest=lowest, mat=mat, mat_unit=mat_unit,
                                                     margin=margin, via_fallback=via_fallback,
                                                     low_confidence=low_confidence, pad_flag=pad_flag)
    return {"name": name, "status": status, "mat": mat, "mat_unit": mat_unit, "lowest": lowest,
             "margin": margin, "calc_note": calc_note, "mat_provenance": mat_provenance,
             "ut_provenance": ut_provenance, "degradation": degradation, "via_fallback": via_fallback,
             "low_confidence": low_confidence, "pad_flag": pad_flag,
             "non_boundary_readings": non_boundary_readings,
             "evaluation_summary": evaluation_summary, "raw": component}


MIN_QUALIFYING_EXCURSION_MINUTES = 15  # sustained-duration threshold; see evaluate_pressure_events


def evaluate_pressure_events(pressure_events):
    """Plain-Python qualification + variance calc for every logged pressure
    reading. The model supplies event_id/peak/design/duration/timestamp/
    duplicate-flag only — every comparison below (exceeds-design,
    meets-duration, and the resulting qualifies-as-excursion determination)
    is computed here, never trusted from the model, per this file's standing
    rule that safety-relevant arithmetic and thresholding is never asked of
    the LLM.

    An event "qualifies" as a confirmed overpressure excursion only when
    BOTH hold: observed pressure strictly exceeds the stated design
    pressure, AND the stated duration meets/exceeds
    MIN_QUALIFYING_EXCURSION_MINUTES. A pressure exceedance with no stated
    duration is never assumed to be either brief or sustained — it is kept
    in the returned list (nothing is silently dropped) but marked
    qualifies=False and duration_status="UNKNOWN" so the UI can route it to
    REQUIRES_VERIFICATION rather than either clearing it or treating it as a
    confirmed excursion on pressure alone. An exceedance whose stated
    duration falls short of the threshold is marked qualifies=False and
    duration_status="BELOW_THRESHOLD" — it is a real logged exceedance, just
    not one that meets this app's sustained-excursion bar.

    Non-exceedances (observed <= design) are dropped entirely, as before —
    they were never excursions of any kind. Entries flagged by the model as
    a duplicate/continuation of an earlier event are kept (so nothing
    vanishes from the record) but marked is_duplicate so the UI can render
    one banner instead of two for the same physical event."""
    results = []
    for event in pressure_events or []:
        observed = event.get("peak_pressure_psi")
        design = event.get("design_pressure_psi")
        if observed is None or design is None or design == 0:
            continue
        if observed <= design:
            continue
        variance_abs = round(observed - design, 4)
        variance_pct = round((variance_abs / design) * 100, 1)
        unit = event.get("unit") or "psi"
        duration = event.get("duration_minutes")

        if duration is None:
            duration_status = "UNKNOWN"
            qualifies = False
        elif duration >= MIN_QUALIFYING_EXCURSION_MINUTES:
            duration_status = "MEETS_THRESHOLD"
            qualifies = True
        else:
            duration_status = "BELOW_THRESHOLD"
            qualifies = False

        results.append({
            "event_id": event.get("event_id") or f"PE-{len(results)+1:02d}",
            "observed": observed, "design": design, "unit": unit,
            "variance_abs": variance_abs, "variance_pct": variance_pct,
            "duration_minutes": duration,
            "duration_status": duration_status,
            "qualifies": qualifies,
            "timestamp_or_context": event.get("timestamp_or_context"),
            "is_duplicate": bool(event.get("is_duplicate_or_continuation")),
            "related_event_ids": event.get("related_event_ids") or [],
        })
    return results


GENERIC_NEXT_STEPS = [
    "Verify governing code/standard and jurisdictional compliance requirements.",
    "Confirm localized thinning via additional ultrasonic thickness (UT) sweeps.",
    "Determine applicability of a Fitness-for-Service (FFS) evaluation once metallurgy and design basis are confirmed.",
]


def has_governing_context(extracted):
    """True only when governing code, jurisdiction, and metallurgy are ALL
    explicitly stated — the gate the system prompt itself is barred from
    bypassing when recommending anything code-specific."""
    return bool(
        (extracted.get("engineering_framework") or "").strip()
        and (extracted.get("jurisdiction") or "").strip()
        and (extracted.get("metallurgy_specification") or "").strip()
    )


def build_next_steps(component_name, extracted):
    """Gates specific, code-referencing remediation steps behind having a
    governing code + jurisdiction + metallurgy on file. Without all three,
    only the conservative, generic next-steps list is shown — the system
    never recommends a specific code pathway (e.g. "API 579 FFS") when it
    doesn't actually know what code or jurisdiction governs this asset."""
    if has_governing_context(extracted):
        return remediation_steps(component_name)
    return list(GENERIC_NEXT_STEPS)


RISK_TIER_ORDER = ["FAIL_BELOW_CRITERION", "CONFLICT", "MISSING_INFORMATION", "REQUIRES_VERIFICATION", "INFORMATIONAL"]


def build_risk_ledger(results, field_anomalies, pressure_excursions, ledger_items):
    """Assembles the unified 5-tier risk ledger, replacing the old flat
    'UNRESOLVED / Requires Validation' bucket. Every entry is plain-Python
    generated text; the tier a component-level entry lands in is decided by
    evaluate_component()'s deterministic status, not by the model. Model-
    supplied field_anomalies keep whichever of the 5 tiers the model
    assigned (falling back to REQUIRES_VERIFICATION for anything malformed,
    which is the most conservative tier to default into)."""
    entries = {tier: [] for tier in RISK_TIER_ORDER}

    for r in results:
        cat = (r.get("evaluation_summary") or {}).get("risk_category")
        if cat == "FAIL_BELOW_CRITERION":
            unit = f" {r['mat_unit']}" if r.get("mat_unit") else ""
            entries[cat].append(
                f"{r['name']}: measured {r['lowest']['value']:.4f}{unit} vs. required "
                f"{r['mat']:.4f}{unit} (margin {r['margin']:+.4f})"
            )
            if r.get("pad_flag"):
                entries["REQUIRES_VERIFICATION"].append(
                    f"{r['name']}: base-shell reading is below the required minimum independent of the "
                    "on-file reinforcement/repair pad reading at this location — an engineering "
                    "assessment of the pad per API 570 is required in addition to the base-shell "
                    "exceedance above."
                )
        elif r["status"] == "requires_verification_prep":
            entries["REQUIRES_VERIFICATION"].append(r["evaluation_summary"]["status_explanation"])
        elif cat == "MISSING_INFORMATION":
            if r["status"] == "no_criteria_no_measurements":
                entries[cat].append(f"{r['name']}: no design acceptance criteria or wall-thickness examination metrics provided")
            elif r["status"] == "no_measurements":
                entries[cat].append(f"{r['name']}: stated minimum on file but no UT readings extracted")
            else:
                entries[cat].append(f"{r['name']}: margin uncalculable — missing engineering variables")

    for pe in pressure_excursions:
        if pe["is_duplicate"]:
            continue  # same physical event as one already listed — don't double-count
        base = (
            f"Pressure event {pe['event_id']}: observed {pe['observed']:g} {pe['unit']} exceeds design "
            f"threshold {pe['design']:g} {pe['unit']} by +{pe['variance_abs']:g} {pe['unit']} "
            f"(+{pe['variance_pct']:g}%)"
        )
        if pe["qualifies"]:
            # Exceeds design pressure AND sustained >= MIN_QUALIFYING_EXCURSION_MINUTES —
            # a confirmed qualifying overpressure excursion.
            entries["FAIL_BELOW_CRITERION"].append(
                f"{base}, sustained {pe['duration_minutes']:g} min — qualifies as an overpressure excursion."
            )
        elif pe["duration_status"] == "UNKNOWN":
            # Exceeds design pressure but the text gives no duration — can't confirm this was
            # sustained rather than a brief transient, so this is neither cleared nor a
            # confirmed excursion until a human resolves the duration.
            entries["REQUIRES_VERIFICATION"].append(
                f"{base}, duration not stated — cannot confirm whether this meets the "
                f"{MIN_QUALIFYING_EXCURSION_MINUTES}-minute sustained-duration threshold for a "
                f"qualifying overpressure excursion."
            )
        else:  # BELOW_THRESHOLD
            # Exceeds design pressure but the stated duration is below the sustained-excursion
            # bar — a real logged exceedance, just not one this app treats as a confirmed
            # qualifying excursion.
            entries["INFORMATIONAL"].append(
                f"{base}, sustained only {pe['duration_minutes']:g} min — below the "
                f"{MIN_QUALIFYING_EXCURSION_MINUTES}-minute sustained-duration threshold, so not "
                f"treated as a qualifying overpressure excursion."
            )

    for a in field_anomalies:
        cat = a.get("risk_category")
        if cat not in entries:
            cat = "REQUIRES_VERIFICATION"  # conservative default for a malformed/missing category
        label = a.get("finding") or "unspecified finding"
        notes = a.get("notes")
        entries[cat].append(label + (f" — {notes}" if notes else ""))

    for item in ledger_items:
        entries["MISSING_INFORMATION"].append(item)

    return entries


def evaluate(extracted):
    components = extracted.get("components_matrix") or []
    asset_category = (extracted.get("asset_category") or "").lower()
    is_piping = "pip" in asset_category  # covers "piping" / "pipeline" / "process pipe"
    piping_vars = extracted.get("piping_design_variables")

    results = [evaluate_component(c, is_piping, piping_vars) for c in components]
    blocked = [r for r in results if r["status"] == "blocked"]
    unresolved = [r for r in results if r["status"] in ("insufficient", "no_measurements", "no_criteria_no_measurements", "requires_verification_prep")]
    degradations = [r["degradation"] for r in results if r.get("degradation") and r["degradation"]["is_loss"]]
    field_anomalies = extracted.get("field_anomalies") or []
    ledger_items = extracted.get("missing_engineering_variables_ledger") or []
    pressure_excursions = evaluate_pressure_events(extracted.get("pressure_events"))

    risk_ledger = build_risk_ledger(results, field_anomalies, pressure_excursions, ledger_items)
    for c in components:
        nm = c.get("component_name") or "Unnamed Component"
        for msg in c.get("review_notes") or []:
            risk_ledger["REQUIRES_VERIFICATION"].append(f"{nm}: {msg}")
        for msg in (c.get("correction_notes") or []) + (c.get("precedence_notes") or []):
            risk_ledger["INFORMATIONAL"].append(f"{nm}: {msg}")
    has_fail = bool(risk_ledger["FAIL_BELOW_CRITERION"])
    has_other_open_risk = any(risk_ledger[t] for t in ("CONFLICT", "MISSING_INFORMATION", "REQUIRES_VERIFICATION"))

    if not results and not any(risk_ledger.values()):
        global_status = "CONDITION UNVERIFIED"
    elif has_fail:
        global_status = "BLOCKED"
    elif has_other_open_risk:
        global_status = "CONDITION UNVERIFIED"
    else:
        global_status = "VERIFIED SECURE"

    ledger_message = build_status_ledger_message(global_status, risk_ledger)

    raw_confidence = extracted.get("confidence_metrics") or {}
    confidence_metrics = {
        "text_extraction_confidence": raw_confidence.get("text_extraction_confidence") or "UNKNOWN",
        "calculation_confidence": raw_confidence.get("calculation_confidence") or "UNKNOWN",
        "source_conflict_level": raw_confidence.get("source_conflict_level") or "UNKNOWN",
        # Hardcoded, never asked of the model: this app makes no independent
        # engineering determination under any circumstance, so this axis is
        # always the same fixed value rather than something the model could
        # accidentally claim otherwise.
        "engineering_determination": "NOT_AVAILABLE",
    }

    return {"results": results, "blocked": blocked, "unresolved": unresolved,
             "field_anomalies": field_anomalies, "degradations": degradations,
             "pressure_excursions": pressure_excursions, "risk_ledger": risk_ledger,
             "global_status": global_status, "ledger_message": ledger_message,
             "confidence_metrics": confidence_metrics,
             "master_records": build_master_records(components, results)}


def build_status_ledger_message(global_status, risk_ledger):
    """Builds a context-specific diagnostic sentence naming the exact
    parameters driving the restriction, drawn from the tiered risk ledger,
    instead of a bare status word — and explicitly notes that the workflow
    label is a policy decision, not an independent engineering determination
    (requirement: decouple workflow status from mathematical results)."""
    if global_status == "VERIFIED SECURE":
        return ("VERIFIED SECURE: All mapped components carry a stated criteria set and a current "
                "reading at or above minimum. No confirmed exceedances or open risk items on file.")

    fails = risk_ledger.get("FAIL_BELOW_CRITERION", [])
    other_entries = []
    for tier in ("CONFLICT", "MISSING_INFORMATION", "REQUIRES_VERIFICATION"):
        other_entries.extend(risk_ledger.get(tier, []))

    if fails:
        lead = fails[0]
    elif other_entries:
        lead = other_entries[0]
    else:
        return f"{global_status}: No mapped components resolved to a confirmed status; awaiting further data."

    all_entries = fails + other_entries
    trailing = [e for e in all_entries if e != lead]

    if not trailing:
        return (f"{global_status}: {lead}. (Workflow status reflects system safety policy — see the "
                f"risk taxonomy ledger below for how each item was categorized, not an independent "
                f"engineering determination.)")

    entries_text = "; ".join(trailing)
    return (f"{global_status}: {lead}. Unresolved risk entries include: {entries_text}. (Workflow "
            f"status reflects system safety policy, not an independent engineering determination.)")


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
    if "tube" in n or "coil" in n:
        return ["Do not assume acceptability from thickness alone — schedule a wall-thickness examination (UT/eddy current) for this bundle.",
                 "Confirm the design acceptance criteria for this bundle with piping/mechanical engineering before disposition.",
                 "Flag for Authorized Inspector review pending both criteria and readings being established."]
    if "shell" in n or "head" in n or "skirt" in n:
        return ["Perform a local Fitness-for-Service assessment (e.g. API 579 Level 1/2) before continued operation.",
                 "Grid the surrounding area to bound the extent of the thin region.",
                 "Consider a pressure de-rate as an interim measure pending repair."]
    return ["Route this component to Fitness-for-Service / Authorized Inspector review.",
             "Bound the extent of the thin area with supplemental UT grid readings.",
             "Do not return the component to unrestricted service until disposition is issued."]

# ============================================================================
# INTERNAL VALIDATOR PASS (sanity net over the parsed component data — best-
# effort, not a guarantee — plus flat-manifest-specific parse notices)
# ============================================================================

def audit_extraction(report_text, extracted, parse_notices):
    notices = list(parse_notices)
    components = extracted.get("components_matrix") or []

    numeric_tokens = re.findall(r"\b\d+\.\d+\b", report_text)
    total_readings = sum(len(c.get("ut_thickness_measurements") or []) for c in components)
    if numeric_tokens and total_readings < max(1, int(len(numeric_tokens) * 0.5)):
        notices.append(
            f"The source text contains {len(numeric_tokens)} decimal values, but only "
            f"{total_readings} were mapped into components — some readings may not have been assigned."
        )

    keywords = ["column", "plate", "gusset", "flange", "shell", "head", "nozzle", "beam", "brace", "support", "tube", "coil", "skirt"]
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


def render_pressure_alert_banner(pe):
    dup_note = ""
    if pe.get("related_event_ids"):
        dup_note = f" (related to {', '.join(pe['related_event_ids'])})"
    context = f" — {pe['timestamp_or_context']}" if pe.get("timestamp_or_context") else ""

    if pe["qualifies"]:
        headline = f"⚠ Qualifying Overpressure Excursion — {pe['event_id']}{context}"
        duration_note = (
            f"Sustained {pe['duration_minutes']:g} min, meeting the "
            f"{MIN_QUALIFYING_EXCURSION_MINUTES}-minute qualification threshold."
        )
        closing = "Flagged for immediate operational review — do not defer to a footnote."
    else:  # duration_status == "UNKNOWN" — below-threshold events don't reach this banner
        headline = f"⚠ Pressure Exceedance, Duration Unconfirmed — {pe['event_id']}{context}"
        duration_note = (
            "No duration was stated in the source text, so this cannot yet be confirmed as a "
            f"sustained ({MIN_QUALIFYING_EXCURSION_MINUTES}+ min) qualifying excursion."
        )
        closing = "Flagged as REQUIRES_VERIFICATION pending confirmation of how long this was sustained."

    st.markdown(
        f"<div class='pressure-alert-banner'>"
        f"<div class='headline'>{headline}</div>"
        f"<div class='body-text'>Observed peak of {pe['observed']:g} "
        f"{pe['unit']} against a stated design pressure threshold of "
        f"{pe['design']:g} {pe['unit']}. Threshold variance: "
        f"+{pe['variance_abs']:g} {pe['unit']} / "
        f"+{pe['variance_pct']:g}% overload. {duration_note}{dup_note}. {closing}</div></div>",
        unsafe_allow_html=True,
    )


_TIER_CSS = {
    "FAIL_BELOW_CRITERION": "tier-fail",
    "CONFLICT": "tier-conflict",
    "MISSING_INFORMATION": "tier-missing",
    "REQUIRES_VERIFICATION": "tier-verify",
    "INFORMATIONAL": "tier-info",
}
_TIER_LABELS = {
    "FAIL_BELOW_CRITERION": "Fail — Below/Outside Criterion",
    "CONFLICT": "Conflict — Sources Disagree",
    "MISSING_INFORMATION": "Missing Information",
    "REQUIRES_VERIFICATION": "Requires Verification (NDT/Human)",
    "INFORMATIONAL": "Informational",
}


def render_risk_taxonomy_ledger(risk_ledger):
    """Renders the strict 5-tier risk ledger, replacing the old flat
    'UNRESOLVED / Requires Validation' bucket with mutually-exclusive,
    clearly labeled categories."""
    any_entries = any(risk_ledger.values())
    if not any_entries:
        st.markdown(
            "<div class='risk-tier-card tier-info'><div class='tier-label'>Nothing Outstanding</div>"
            "No risk-ledger entries in any of the 5 tiers.</div>",
            unsafe_allow_html=True,
        )
        return
    for tier in RISK_TIER_ORDER:
        items = risk_ledger.get(tier) or []
        if not items:
            continue
        items_html = "".join(f"<div>• {item}</div>" for item in items)
        st.markdown(
            f"<div class='risk-tier-card {_TIER_CSS[tier]}'>"
            f"<div class='tier-label'>{_TIER_LABELS[tier]} ({len(items)})</div>{items_html}</div>",
            unsafe_allow_html=True,
        )


def render_degradation_card(deg):
    st.markdown(
        f"<div class='degradation-card'>"
        f"<div class='headline'>Degradation Anomaly Captured</div>"
        f"<div class='body-text'>{deg['component_name']}: prior lowest reading "
        f"{deg['previous_value']:.4f} {deg['unit']} now measures {deg['current_value']:.4f} {deg['unit']} — "
        f"an observed reduction of -{deg['delta']:.4f} {deg['unit']}. Flagged for remaining-life "
        f"tracking and corrosion-rate evaluation loops.</div></div>",
        unsafe_allow_html=True,
    )


def render_status_ledger_banner(outcome):
    gs = outcome["global_status"]
    css_class = "blocked" if gs == "BLOCKED" else ("secure" if gs == "VERIFIED SECURE" else "unverified")
    st.markdown(
        f"<div class='status-ledger-banner {css_class}'><b>{outcome['ledger_message']}</b></div>",
        unsafe_allow_html=True,
    )


def render_provenance_expander(result):
    """'Why this value?' traceability — only rendered when the model actually
    supplied provenance metadata for this component; never fabricated."""
    mat_prov = result.get("mat_provenance")
    ut_prov = result.get("ut_provenance")
    if not mat_prov and not ut_prov:
        return
    with st.expander("Why this value? (Data Provenance)"):
        if mat_prov:
            st.markdown("**Required Minimum MAT selection**")
            if mat_prov.get("authority_tier"):
                st.markdown(f"- Authority tier: *{mat_prov['authority_tier']}*")
            if mat_prov.get("source"):
                st.markdown(f"- Source: {mat_prov['source']}")
            if mat_prov.get("general_criterion") is not None:
                st.markdown(f"- General/default criterion overridden: {mat_prov['general_criterion']:.4f}")
            if mat_prov.get("reason"):
                st.markdown(f"- Why this value: {mat_prov['reason']}")
        if ut_prov:
            st.markdown("**Measured Minimum UT selection**")
            if ut_prov.get("authority_tier"):
                st.markdown(f"- Authority tier: *{ut_prov['authority_tier']}*")
            if ut_prov.get("source"):
                st.markdown(f"- Source: {ut_prov['source']}" + (f" ({ut_prov['source_date']})" if ut_prov.get("source_date") else ""))
            if ut_prov.get("all_values_in_region"):
                vals = ", ".join(f"{v:g}" for v in ut_prov["all_values_in_region"])
                st.markdown(f"- All candidate values considered in this region: {vals}")
            if ut_prov.get("reason"):
                st.markdown(f"- Why this value: {ut_prov['reason']}")


def render_audit_trail_expander(result, next_steps):
    """End-to-end audit trail: Source Segment -> Extracted Raw Values ->
    Applied Precedence Rule -> Calculation -> Risk Category -> Finding ->
    Action Items, all drawn from data already computed elsewhere in this
    file — nothing new is asserted here."""
    summary = result.get("evaluation_summary") or {}
    mat_prov = result.get("mat_provenance") or {}
    ut_prov = result.get("ut_provenance") or {}
    precedence_bits = []
    if ut_prov.get("authority_tier"):
        precedence_bits.append(f"UT reading: {ut_prov['authority_tier']}")
    if mat_prov.get("authority_tier"):
        precedence_bits.append(f"Criterion: {mat_prov['authority_tier']}")
    precedence_text = "; ".join(precedence_bits) if precedence_bits else "No explicit source-precedence conflict was recorded for this component."

    with st.expander("End-to-End Audit Trail"):
        st.markdown(f"1. **Source Segment** — {result['name']}")
        if result.get("mat") is not None and result.get("lowest"):
            unit = f" {result['mat_unit']}" if result.get("mat_unit") else ""
            lu = f" {result['lowest']['unit']}" if result['lowest'].get('unit') else ""
            st.markdown(f"2. **Extracted Raw Values** — measured minimum {result['lowest']['value']:.4f}{lu}; stated/derived criterion {result['mat']:.4f}{unit}")
        else:
            st.markdown("2. **Extracted Raw Values** — insufficient data extracted to populate this step")
        st.markdown(f"3. **Applied Precedence Rule** — {precedence_text}")
        if result.get("margin") is not None:
            st.markdown(f"4. **Calculation** — margin = measured − criterion = {result['margin']:+.4f}")
        else:
            st.markdown("4. **Calculation** — not computable from available data")
        st.markdown(f"5. **Risk Category** — {summary.get('risk_category') or 'None (within spec)'}")
        st.markdown(f"6. **Finding** — {summary.get('status_explanation', '')}")
        steps_text = "; ".join(next_steps) if next_steps else "None required."
        st.markdown(f"7. **Action Items** — {steps_text}")


def _mat_binding_text(result):
    mb = (result.get("raw") or {}).get("mat_binding") or {}
    if not mb:
        return "n/a"
    return f"{mb.get('component_type', '?')} → {mb.get('source', 'n/a')}"


def render_component_card(result, extracted):
    status = result["status"]
    via_fallback = bool(result.get("via_fallback"))
    low_confidence = bool(result.get("low_confidence"))
    unconfirmed_downgrade = (via_fallback or low_confidence) and status == "verified"

    if status == "blocked":
        css_class = "blocked"
    elif status == "verified" and not unconfirmed_downgrade:
        css_class = "verified"
    else:
        css_class = "unresolved"

    pill_label = {
        "blocked": "Blocked",
        "verified": "Pending Verification" if unconfirmed_downgrade else "Verified",
        "insufficient": "Unresolved",
        "no_measurements": "No Data",
        "no_criteria_no_measurements": "Unverified",
        "requires_verification_prep": "Requires Verification (Surface Prep)",
    }[status]
    summary = result.get("evaluation_summary") or {}

    st.markdown(
        f"<div class='glass-card component-card {css_class}'>"
        f"<div style='font-size:1.05rem;font-weight:700;color:#F9FAFB;'>{result['name']}"
        f"<span class='status-pill {css_class}'>{pill_label}</span></div>",
        unsafe_allow_html=True,
    )

    if via_fallback:
        st.caption(
            "⚙️ Recovered via the automatic fallback reparse engine (raw table structure), not the "
            "primary extraction — treat readings as lower-confidence pending verification."
        )
    if low_confidence:
        st.caption(
            "🔎 Source text tags this reading LOW measurement confidence — not treated as a "
            "definitive pass even where the computed margin is positive."
        )
    if result.get("pad_flag"):
        st.caption(
            "🛠️ Base-shell exceedance confirmed independent of an on-file reinforcement/repair pad "
            "reading at this location — a separate API 570 engineering assessment of the pad is "
            "required in addition to the shell finding."
        )
    if status == "requires_verification_prep":
        unit = f" {result.get('mat_unit')}" if result.get("mat_unit") else ""
        pre_r, post_r = result["pre_reading"], result["post_reading"]
        st.caption(
            f"🧽 Pre-prep reading {pre_r['location_label']}={pre_r['value']:.4f}{unit} was below the "
            f"required minimum; post-prep reading {post_r['location_label']}={post_r['value']:.4f}{unit} "
            "clears it. Routed to human review rather than auto-passed or auto-failed."
        )
    non_boundary = result.get("non_boundary_readings") or []
    if non_boundary:
        items = ", ".join(f"{r['location_label']}={r['value']:g}{r.get('unit') or ''}" for r in non_boundary)
        st.caption(
            f"🛡️ Isolated non-pressure-boundary reading(s) on file for this location ({items}) — "
            "excluded from the base-metal margin calculation above, per the reinforcement/repair "
            "isolation rule."
        )

    # Decoupled math-vs-workflow line: the mathematical calculation_result
    # is never presented as if it were itself the workflow_status decision.
    if summary:
        st.markdown(
            f"<div class='decoupled-status-line'>Calculation result: <b>{summary.get('calculation_result')}</b> "
            f"&nbsp;|&nbsp; System workflow status: <b>{summary.get('workflow_status')}</b></div>",
            unsafe_allow_html=True,
        )

    # Multi-inspection historical trend — surfaced for every component that has
    # a comparable prior reading, regardless of current pass/fail status.
    deg = result.get("degradation")
    if deg and deg["is_loss"]:
        render_degradation_card(deg)

    next_steps = []
    if status in ("blocked", "verified"):
        unit_suffix = f" {result['mat_unit']}" if result.get("mat_unit") else ""
        lowest = result["lowest"]
        lowest_unit = f" {lowest['unit']}" if lowest.get("unit") else ""

        # Auditable calculation-trail table: Component -> Measured Minimum UT ->
        # Required Minimum MAT -> Computed True Margin -> Status, in one row.
        st.markdown(
            "<table class='calc-trail-table'>"
            f"<tr><td class='label'>Measured Minimum UT</td><td>{_fmt_reading(lowest)}</td></tr>"
            f"<tr><td class='label'>Required Minimum MAT</td><td>{result['mat']:.4f}{unit_suffix}</td></tr>"
            f"<tr><td class='label'>MAT Binding</td><td>{_mat_binding_text(result)}</td></tr>"
            f"<tr><td class='label'>Computed True Margin</td><td>{result['margin']:+.4f}</td></tr>"
            f"<tr><td class='label'>Status</td><td>{pill_label}</td></tr>"
            "</table>",
            unsafe_allow_html=True,
        )
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
            next_steps = build_next_steps(result["name"], extracted)
            gated_note = "" if has_governing_context(extracted) else (
                "<div style='color:#FBBF24;font-size:0.78rem;margin-top:4px;'>Governing code, jurisdiction, "
                "and/or metallurgy are not all on file — showing conservative generic next steps only, not a "
                "specific code-level recommendation.</div>"
            )
            steps_html = "".join(f"<li>{s}</li>" for s in next_steps)
            st.markdown(
                f"<div class='disposition-block'><div class='disposition-label'>Potential Next Steps for "
                f"Authorized Engineer — Awaiting Authorized Engineering/Inspector Review</div><ul>{steps_html}</ul>"
                f"{gated_note}"
                f"<div style='color:#9CA3AF;font-size:0.8rem;'>This is a suggested exploration path, not a final "
                f"disposition — the confirmed exceedance above stands independent of whichever pathway is ultimately "
                f"authorized.</div></div>",
                unsafe_allow_html=True,
            )
    elif status == "insufficient":
        st.write("Cannot compute a margin for this component — missing:")
        for m in result["missing_vars"]:
            st.markdown(f"- ❌ **{m}**")
        next_steps = ["Supply the missing engineering variable(s) listed above.", "Route to an authorized engineer to confirm an interim basis if data cannot be obtained promptly."]
    elif status == "no_measurements":
        st.write("A minimum is on file for this component, but no UT readings were extracted for it.")
        next_steps = ["Schedule a wall-thickness examination (UT) for this component.", "Confirm the stated minimum is still the applicable criterion before testing."]
    else:  # no_criteria_no_measurements
        st.warning(
            "No design acceptance criteria or wall-thickness examination metrics provided; "
            "structural condition is unverified."
        )
        next_steps = ["Establish a design acceptance criterion for this component with engineering.", "Schedule a wall-thickness examination (UT) once criteria are established."]

    render_provenance_expander(result)
    render_audit_trail_expander(result, next_steps)

    with st.expander("View Raw Source Extraction Parameters Line"):
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
        with st.spinner("Checking this amazing file"):
            try:
                extracted = extract_with_claude(client, report_text)
                manifest_text = extracted.get("flat_manifest_block", "")
                global_mat = extracted.get("global_minimum_allowable_thickness")
                components_matrix, parse_notices = native_parameter_matrix_parser(manifest_text, report_text, global_mat=global_mat)

                # Automatic Fallback Pipeline Controller: only fires on the specific failure
                # mode this stage exists for — the model's own self-assessment says the text
                # was legible (HIGH), the raw report plainly contains table-like structure, and
                # yet the primary parse still mapped zero components. A report that genuinely
                # has nothing extractable (low confidence, or no table structure at all) is
                # left to resolve as CONDITION UNVERIFIED as before, rather than retried.
                text_extraction_confidence = (extracted.get("confidence_metrics") or {}).get("text_extraction_confidence")
                if (
                    text_extraction_confidence == "HIGH"
                    and len(components_matrix) == 0
                    and raw_text_has_components_or_tables(report_text)
                ):
                    fallback_components, fallback_notices = execute_fallback_reparse(report_text, global_mat=global_mat)
                    components_matrix = fallback_components
                    parse_notices = list(parse_notices) + fallback_notices

                # Deterministic semantic audit: unit sanity, addendum precedence, pad linkage,
                # historical reconciliation and MAT-by-component-type binding. Margins are only
                # computed afterwards (evaluate), from the normalized values.
                components_matrix, recon_notices = audit_and_reconcile(components_matrix, extracted)
                parse_notices = list(parse_notices) + recon_notices

                extracted["components_matrix"] = components_matrix
                st.session_state["extracted"] = extracted
                st.session_state["parse_notices"] = parse_notices
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
    parse_notices = st.session_state.get("parse_notices", [])
    outcome = evaluate(extracted)
    audit_notices = audit_extraction(report_text, extracted, parse_notices)
    cm = outcome["confidence_metrics"]

    # Pressure-excursion alerts always sit at the very top of the matrix panel —
    # one banner per distinct event; duplicates/continuations are folded in.
    for pe in outcome["pressure_excursions"]:
        if not pe["is_duplicate"] and pe["duration_status"] != "BELOW_THRESHOLD":
            render_pressure_alert_banner(pe)

    st.markdown("#### Global Engineering Status Ledger")
    render_status_ledger_banner(outcome)

    st.markdown("#### 5-Tier Risk Taxonomy Ledger")
    render_risk_taxonomy_ledger(outcome["risk_ledger"])

    st.markdown("#### Multi-Dimensional Confidence Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_html("Text Extraction Confidence", cm["text_extraction_confidence"], "accent-cyan"), unsafe_allow_html=True)
    m2.markdown(metric_html("Calculation Confidence", cm["calculation_confidence"], "accent-cyan"), unsafe_allow_html=True)
    m3.markdown(metric_html("Source Conflict Level", cm["source_conflict_level"]), unsafe_allow_html=True)
    m4.markdown(metric_html("Engineering Determination", cm["engineering_determination"]), unsafe_allow_html=True)
    st.caption("Engineering Determination is always NOT_AVAILABLE — this system extracts, calculates, and flags; it never issues an independent engineering sign-off.")

    st.markdown("#### Global Engineering Safety Status")
    gs = outcome["global_status"]
    gs_accent = "accent-gold" if gs == "BLOCKED" else None
    st.markdown(metric_html("Workflow Status (system policy, not an engineering determination)", gs, gs_accent), unsafe_allow_html=True)

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
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_html("Asset Category", extracted.get("asset_category") or "Not stated"), unsafe_allow_html=True)
        c2.markdown(metric_html("Metallurgy", extracted.get("metallurgy_specification") or "Not stated"), unsafe_allow_html=True)
        c3.markdown(metric_html("Governing Code", extracted.get("engineering_framework") or "Not determined"), unsafe_allow_html=True)
        c4.markdown(metric_html("Jurisdiction", extracted.get("jurisdiction") or "Not stated"), unsafe_allow_html=True)
        if not has_governing_context(extracted):
            st.caption("Governing code, jurisdiction, and metallurgy are not all on file — code-specific recommendations are suppressed in favor of conservative generic next steps.")
        gmat = extracted.get("global_minimum_allowable_thickness") or {}
        if gmat.get("value") is not None:
            st.caption(
                f"📐 Document-wide global MAT on file: {gmat['value']:g} {gmat.get('unit') or ''} — "
                "applied to shell-class components only, never to tubes."
            )

        st.markdown("#### Calculation Trail Ledger")
        st.caption("Component Name → Measured Minimum UT → Required Minimum MAT → Computed True Margin → Calculation Result / Workflow Status → Potential Next Steps")
        for result in outcome["results"]:
            render_component_card(result, extracted)

        render_master_records(outcome["master_records"])

        ledger = extracted.get("missing_engineering_variables_ledger") or []
        if ledger:
            st.markdown("#### 📒 Data Deficit Ledger")
            for item in ledger:
                st.markdown(f"- {item}")

        anomalies = outcome["field_anomalies"]
        if anomalies:
            st.markdown("#### ⚠️ Field Anomalies")
            for a in anomalies:
                cat = a.get("risk_category") or "REQUIRES_VERIFICATION"
                css = _TIER_CSS.get(cat, "unresolved")
                label = _TIER_LABELS.get(cat, cat)
                st.markdown(
                    f"<div class='glass-card'><b>{a.get('finding')}</b> "
                    f"<span class='status-pill {'blocked' if cat == 'FAIL_BELOW_CRITERION' else 'unresolved'}'>{label}</span>"
                    f"<div style='color:#D1D5DB;margin-top:6px;'>{a.get('notes') or ''}</div></div>",
                    unsafe_allow_html=True,
                )

    with st.expander("Raw flat_manifest_block returned by the model (pre-parse)"):
        st.text(extracted.get("flat_manifest_block", ""))

    with st.expander("Raw structured response from the model (post-parse)"):
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
