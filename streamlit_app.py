"""
TraceLink AI — Dynamic Parameter Discovery Pipeline
-----------------------------------------------------
Single-file Streamlit application. Fully generic: there is no
`if asset_type == "..."` branching anywhere in this file. Every
extraction is driven by keyword-weight maps and numerical proximity
scoring against a flat token stream, so a heat exchanger, storage tank,
or structural truss report is handled by the exact same code path as
a pressure vessel or piping report — nothing here has to change to
support a new asset class in the field.

Architectural pillars:
  1. Universal Tokenization Engine   -> tokenize()
  2. Dynamic ceiling/threshold + design-variable discovery
                                      -> find_closest_number() + *_KEYWORDS maps
  3. Core Evaluation Logic Matrix     -> run_dynamic_analysis()
  4. Extraction-boundary exception
     handling (auto B31.3 fallback)  -> inside run_dynamic_analysis()
  5. Field Anomaly Harvester          -> harvest_anomalies()
  6. Enterprise layout                -> UI section at bottom

No `re` import (manual character scanning instead) and no vector-DB
import (the reference/RAG panel is a flat Python list).
"""

import streamlit as st

st.set_page_config(
    page_title="TraceLink AI — Dynamic Compliance Engine",
    layout="wide",
    page_icon="🛠️",
)

# ============================================================================
# 1. UNIVERSAL TOKENIZATION ENGINE
# ============================================================================

_STRIP_TO_SPACE = set(':,\t\r()[];"\'')


def tokenize(text):
    """Strip colons/commas/tabs/carriage-returns/etc. and split into a flat
    token stream. Newlines collapse like any other whitespace — proximity
    scoring works purely on token distance, not line boundaries, exactly
    as a token-stream design should."""
    out_chars = []
    for ch in text:
        out_chars.append(" " if ch in _STRIP_TO_SPACE else ch)
    return "".join(out_chars).split()


UNIT_TOKENS = {
    "in", "in.", "mm", "cm", "m", "ft", "psi", "kpa", "mpa", "bar", "ksi",
    "°f", "°c", "f", "c", "lb", "lbs", "kg",
}


def split_leading_number(token):
    """If `token` starts with a signed float, return (value, remainder)
    where remainder is whatever trailing characters follow (e.g. a unit
    fused directly onto the number, like '5.82mm'). Returns (None, token)
    if the token doesn't start with a number. No regex — manual scan."""
    n = len(token)
    idx = 0
    if idx < n and token[idx] == "-":
        idx += 1
    seen_digit = False
    seen_dot = False
    start = idx
    while idx < n:
        c = token[idx]
        if c.isdigit():
            seen_digit = True
            idx += 1
        elif c == "." and not seen_dot:
            seen_dot = True
            idx += 1
        else:
            break
    if not seen_digit:
        return None, token
    num_str = token[:idx]
    remainder = token[idx:]
    try:
        return float(num_str), remainder
    except ValueError:
        return None, token


def looks_like_point_label(token):
    """Matches inspection-label tokens like 'F1', 'A12', 'W2', or a bare
    'A' / 'B' used as a column-style label. Deliberately narrow so it
    doesn't swallow material grades like 'A106' unless a number actually
    follows it shortly afterward (checked by the caller)."""
    if not token or not token[0].isalpha() or not token[0].isupper():
        return False
    rest = token[1:]
    if rest == "":
        return True
    return rest.isdigit() and len(rest) <= 3


def keyword_matches(token_lower, keyword):
    """Short keywords (<=2 chars, e.g. bare variable letters like 'p', 's',
    'e', 'y') must match the *whole* token to avoid matching every word
    that merely contains that letter. Longer keywords match as a
    substring so 'thickness', 'allowable', etc. still catch fused forms."""
    if len(keyword) <= 2:
        return token_lower.strip(".") == keyword
    return keyword in token_lower


# ============================================================================
# 2. DYNAMIC THRESHOLD / VARIABLE DISCOVERY (keyword-weight + proximity)
# ============================================================================

CEILING_KEYWORDS = {"minimum": 3, "allowable": 3, "permitted": 2, "mat": 3, "limit": 2, "thickness": 1}
PRESSURE_KEYWORDS = {"pressure": 3, "design": 1, "p": 2}
DIAMETER_KEYWORDS = {"diameter": 3, "outside": 1, "od": 2}
STRESS_KEYWORDS = {"stress": 3, "allowable": 1, "s": 1}
QUALITY_KEYWORDS = {"quality": 3, "joint": 2, "factor": 1, "e": 1}
Y_COEFF_KEYWORDS = {"coefficient": 2, "y": 3}
CORROSION_KEYWORDS = {"corrosion": 3, "allowance": 2, "ca": 2}

VARIABLE_KEYWORD_MAPS = {
    "Design Pressure (P)": PRESSURE_KEYWORDS,
    "Outside Diameter (D)": DIAMETER_KEYWORDS,
    "Allowable Stress (S)": STRESS_KEYWORDS,
    "Quality / Joint Factor (E)": QUALITY_KEYWORDS,
    "Y Coefficient": Y_COEFF_KEYWORDS,
    "Corrosion Allowance (CA)": CORROSION_KEYWORDS,
}


def find_closest_number(tokens, keyword_weights, exclude_indices=None, min_score=0.0):
    """Locate every keyword hit in the token stream, weight it, then find
    the numeric token with the highest cumulative weight/(1+distance)
    score across all hits. This is the generic engine behind MAT
    discovery AND every design-variable lookup — same function, no
    branching on what concept is being searched for.

    `min_score` is a confidence floor: a single stray keyword hit sitting
    a couple of tokens away from an unrelated number (e.g. the word
    'Allowable' inside 'Allowable Stress S: 15000' bleeding onto a nearby
    diameter value) should not be accepted as a genuine match. Real
    threshold/variable phrases cluster multiple keyword hits tightly
    around their number and comfortably clear a modest floor; isolated,
    distant, single-keyword coincidences don't."""
    exclude_indices = exclude_indices or set()
    lowered = [t.lower() for t in tokens]

    keyword_hits = []
    for i, tok in enumerate(lowered):
        for kw, weight in keyword_weights.items():
            if keyword_matches(tok, kw):
                keyword_hits.append((i, weight))

    if not keyword_hits:
        return None, None

    best_score, best_idx, best_val = 0.0, None, None
    for i, tok in enumerate(tokens):
        if i in exclude_indices:
            continue
        val, _ = split_leading_number(tok)
        if val is None:
            continue
        score = sum(weight / (1 + abs(i - kpos)) for kpos, weight in keyword_hits)
        if score > best_score:
            best_score, best_idx, best_val = score, i, val

    if best_idx is None or best_score < min_score:
        return None, None
    return best_val, best_idx


# ============================================================================
# MEASUREMENT ARRAY DISCOVERY (dynamic, unit- and label-aware)
# ============================================================================

MEASUREMENT_HINT_WORDS = {"ut", "reading", "readings", "point", "points", "measured", "gauge", "inspection"}


def collect_labeled_points(tokens, exclude_indices=None):
    """Column-style measurement tables: a label token (F1, A12, W2, A, B...)
    immediately followed by a numeric token. `exclude_indices` keeps this
    from re-claiming a number already assigned to a discovered design
    variable — a bare 'S', 'E', 'Y', or 'P' is indistinguishable from a
    single-letter column label by shape alone, so a number that the
    variable-discovery pass already owns (e.g. the 15000 in
    'Allowable Stress S: 15000') must not also become a fake reading
    labeled 'S'."""
    exclude_indices = exclude_indices or set()
    points, used = [], set()
    for i, tok in enumerate(tokens):
        if not looks_like_point_label(tok):
            continue
        for j in range(i + 1, min(i + 4, len(tokens))):
            if j in exclude_indices:
                continue
            val, unit = split_leading_number(tokens[j])
            if val is not None:
                points.append({"label": tok, "value": val, "unit": unit or None, "index": j})
                used.add(j)
                break
    return points, used


def collect_hinted_points(tokens, exclude_indices):
    """Fallback for reports with no letter+digit label column: a numeric
    token near a measurement-ish word (UT, reading, gauge, point...) is
    treated as a field measurement, with the nearest earlier word used
    as its label."""
    points = []
    for i, tok in enumerate(tokens):
        if i in exclude_indices:
            continue
        val, unit = split_leading_number(tok)
        if val is None:
            continue
        window = [w.lower().strip(".") for w in tokens[max(0, i - 5):i + 3]]
        if any(w in MEASUREMENT_HINT_WORDS for w in window):
            label = next((w for w in reversed(tokens[max(0, i - 4):i]) if w[:1].isalpha()), None)
            points.append({"label": label or f"reading@{i}", "value": val, "unit": unit or None, "index": i})
    return points


def collect_measurement_array(tokens, exclude_indices):
    points, used = collect_labeled_points(tokens, exclude_indices)
    exclude = exclude_indices | used
    hinted = collect_hinted_points(tokens, exclude | {p["index"] for p in points})
    existing_idx = {p["index"] for p in points}
    for hp in hinted:
        if hp["index"] not in existing_idx:
            points.append(hp)
    return points


# ============================================================================
# ASSET / METALLURGY EXTRACTION (phrase-level, still keyword-driven)
# ============================================================================

ASSET_HEADERS = ("asset category", "asset:", "equipment id", "equipment:", "system:", "type:")


def extract_asset_category(text):
    for raw_line in text.splitlines():
        low = raw_line.lower()
        if any(h in low for h in ASSET_HEADERS) and ":" in raw_line:
            return raw_line.split(":", 1)[1].strip()
    return "Unclassified — no Asset/Equipment/System/Type header located"


def extract_metallurgy(text):
    for raw_line in text.splitlines():
        low = raw_line.lower()
        if ("metallurgy" in low or "material" in low) and ":" in raw_line:
            return raw_line.split(":", 1)[1].strip()
    return None


# ============================================================================
# 5. FIELD ANOMALY HARVESTER — asset-type agnostic
# ============================================================================

ANOMALY_KEYWORDS = (
    "field note", "inspector note", "recommend", "indication", "flaw",
    "ndt", "un-torqued", "untorqued", "corroded", "crack", "anomaly", "variance",
)


def harvest_anomalies(text):
    chunks = []
    for line in text.splitlines():
        chunks.extend(s.strip() for s in line.split(".") if s.strip())

    found, seen = [], set()
    for chunk in chunks:
        low = chunk.lower()
        if not any(kw in low for kw in ANOMALY_KEYWORDS):
            continue
        content = chunk
        if ":" in content:
            prefix, _, rest = content.partition(":")
            if any(tag in prefix.lower() for tag in ("field note", "inspector note")):
                content = rest.strip()
        if content and content not in seen:
            seen.add(content)
            found.append(content)
    return found


# ============================================================================
# 3 & 4. CORE EVALUATION LOGIC MATRIX + EXTRACTION-BOUNDARY EXCEPTION HANDLING
# ============================================================================

def run_dynamic_analysis(text):
    tokens = tokenize(text)
    asset = extract_asset_category(text)
    metallurgy = extract_metallurgy(text)

    # min_score guards against a single stray keyword (e.g. a lone
    # 'Allowable' inside an unrelated 'Allowable Stress' line) being
    # mistaken for a genuine minimum/allowable-thickness phrase.
    ceiling_val, ceiling_idx = find_closest_number(tokens, CEILING_KEYWORDS, min_score=1.3)
    exclude = {ceiling_idx} if ceiling_idx is not None else set()

    calc_note = None
    design_vars, missing_vars = {}, []

    if ceiling_val is None:
        # --- Extraction-boundary exception: try the B31.3 math fallback ---
        var_exclude = set()
        for name, kw_map in VARIABLE_KEYWORD_MAPS.items():
            val, idx = find_closest_number(tokens, kw_map, var_exclude)
            design_vars[name] = val
            if idx is not None:
                var_exclude.add(idx)
        missing_vars = [k for k, v in design_vars.items() if v is None]

        if not missing_vars:
            P = design_vars["Design Pressure (P)"]
            D = design_vars["Outside Diameter (D)"]
            S = design_vars["Allowable Stress (S)"]
            E = design_vars["Quality / Joint Factor (E)"]
            Y = design_vars["Y Coefficient"]
            CA = design_vars["Corrosion Allowance (CA)"]
            t_design = (P * D) / (2 * (S * E + P * Y))
            ceiling_val = t_design + CA
            calc_note = (
                f"No explicit minimum/allowable threshold string was found in the report, so the engine "
                f"dynamically executed the ASME B31.3 straight-pipe formula from the discovered design "
                f"variables: t_design = (P×D)/(2×(S×E+P×Y)) = {t_design:.4f}. Adding the discovered "
                f"corrosion allowance ({CA:.3f}) yields calculated_safety_ceiling = {ceiling_val:.4f}."
            )
            exclude |= var_exclude
        else:
            exclude |= var_exclude
            points = collect_measurement_array(tokens, exclude)
            return {
                "status": "insufficient",
                "asset": asset, "metallurgy": metallurgy,
                "design_vars": design_vars, "missing_vars": missing_vars,
                "points": points, "anomalies": harvest_anomalies(text),
            }

    points = collect_measurement_array(tokens, exclude)
    readings = [p["value"] for p in points]
    lowest = min(readings) if readings else None
    margin = round(lowest - ceiling_val, 4) if lowest is not None else None

    if lowest is None:
        status = "no_measurements"
    elif margin < 0:
        status = "blocked"
    else:
        status = "verified"

    return {
        "status": status,
        "asset": asset, "metallurgy": metallurgy,
        "ceiling": ceiling_val, "lowest": lowest, "margin": margin,
        "points": points, "calc_note": calc_note,
        "anomalies": harvest_anomalies(text),
    }


# ============================================================================
# REFERENCE CONTEXT PANEL (static list — no vector DB, no chromadb import)
# ============================================================================

REFERENCE_CONTEXT = [
    "ASME BPVC Sec. VIII Div.1 UG-25 — minimum thickness after corrosion shall not be less than the design MAT.",
    "ASME B31.3 §304.1.2 — pressure design thickness (t) is calculated from P, D, S, E, Y before a minimum-thickness verdict can be issued.",
    "ASME B31.3 §304.1.1 — total minimum wall = t_design + corrosion allowance (CA).",
    "API 510 §6.5 — components below MAT require immediate engineering disposition before return to service.",
    "AWS D1.1 Table 6.1 — acceptance criteria for visual weld discontinuities; suspect indications require NDT disposition.",
    "AISC 360 §J3 — bolted connections must meet specified pretension; corrosion or under-torque requires re-inspection.",
    "API 579 (FFS) — governs fitness-for-service evaluation once a component is found below its allowable limit.",
]

# ============================================================================
# SAMPLE REPORTS — pure demo data. The pipeline has zero awareness these
# represent different "kinds" of report; nothing below branches on them.
# ============================================================================

SAMPLE_REPORTS = {
    "Pressure vessel (below MAT)": """Asset: Horizontal Hydraulic Pressure Vessel
Metallurgy: SA-516 Grade 70 carbon steel
Nominal Shell Thickness: 0.500 in
Minimum Allowable Shell Thickness (MAT): 0.375 in
UT Point A1: 0.382
UT Point A2: 0.371
UT Point A3: 0.364
UT Point A4: 0.379
""",
    "Process piping, no stated MAT (triggers dynamic B31.3 calc)": """System: Process Piping System
Metallurgy: ASTM A106 Grade B carbon steel
Design Pressure: 500 psi
Outside Diameter: 4.500 in
Allowable Stress S: 15000 psi
Quality Factor E: 1.00
Y Coefficient: 0.40
Corrosion Allowance: 0.050 in
UT Point F1: 0.070
UT Point F2: 0.082
UT Point F3: 0.079
""",
    "Structural steel frame with unresolved anomalies": """Type: Welded Structural Support Frame / HSS Assembly
Metallurgy: ASTM A500 Grade B
Minimum Allowable Thickness: 5.50 mm
West HSS UT Reading: 5.82 mm
Gusset UT Reading: 10.6 mm
Field Note: Possible weld flaw at north joint - NDT required.
Inspector Note: Corroded and un-torqued 19mm anchor bolt at base plate.
Field Note: Unmapped 3mm alignment variance on brace connection.
""",
    "Heat exchanger (unlisted asset class — no code changes needed)": """Equipment ID: Shell-and-Tube Heat Exchanger HX-204
Metallurgy: SA-240 Type 316L stainless steel
Minimum Allowable Tube Wall Thickness: 0.065 in
UT Point T1: 0.071
UT Point T2: 0.058
UT Point T3: 0.069
Inspector Note: Tube T2 shows localized pitting indication near baffle 3, recommend eddy-current retest.
""",
}

# ============================================================================
# 6. ENTERPRISE UI
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


st.title("🛠️ TraceLink AI — Dynamic Parameter Discovery Pipeline")
st.caption(
    "Track-agnostic engine: keyword-weighted proximity scoring over a flat token stream. "
    "No per-asset-type branching — new asset classes need no code changes."
)

with st.sidebar:
    st.header("Input")
    sample_choice = st.selectbox("Load a sample report", ["— none —"] + list(SAMPLE_REPORTS.keys()))
    if "report_text" not in st.session_state:
        st.session_state["report_text"] = SAMPLE_REPORTS["Pressure vessel (below MAT)"]
    if sample_choice != "— none —" and st.button("Load selected sample", use_container_width=True):
        st.session_state["report_text"] = SAMPLE_REPORTS[sample_choice]

    report_text = st.text_area("Field report text", value=st.session_state["report_text"], height=360)
    run = st.button("▶ Run Dynamic Analysis", type="primary", use_container_width=True)

if run:
    result = run_dynamic_analysis(report_text)

    st.subheader("📋 Data Extracted From the Inspection Report")
    c1, c2 = st.columns(2)
    metric_box(c1, "Discovered Asset Category", result["asset"])
    metric_box(c2, "Discovered Metallurgy", result["metallurgy"] or "Not stated in report")

    if result["status"] == "insufficient":
        st.warning("**CONDITION UNVERIFIED — INSUFFICIENT BOUNDARY DATA INPUTS**")
        st.write("No minimum/allowable threshold string was found, and the following mechanical "
                 "variables needed to calculate one dynamically are also missing:")
        for name in result["missing_vars"]:
            st.markdown(f"- ❌ **{name}** — not found in report")
        found_vars = {k: v for k, v in result["design_vars"].items() if v is not None}
        if found_vars:
            st.markdown("Variables that *were* successfully discovered:")
            cols = st.columns(len(found_vars))
            for col, (name, val) in zip(cols, found_vars.items()):
                metric_box(col, name, val)

    else:
        m1, m2, m3, m4 = st.columns(4)
        metric_box(m1, "Calculated Safety Ceiling", f"{result['ceiling']:.4f}" if result["ceiling"] is not None else "—")
        metric_box(m2, "Lowest Captured Reading", f"{result['lowest']:.4f}" if result["lowest"] is not None else "—")
        metric_box(m3, "Margin", f"{result['margin']:+.4f}" if result["margin"] is not None else "—")
        metric_box(m4, "Measurement Points Found", len(result["points"]))

        if result.get("calc_note"):
            st.info(result["calc_note"])

        if result["status"] == "blocked":
            st.error("🔴 **CRITICAL BOUNDARY DEFECT — COMPLIANCE STATUS: BLOCKED**")
            st.markdown(
                "- Lowest captured field reading is below the calculated safety ceiling.\n"
                "- Component is BLOCKED from continued service pending engineering disposition.\n"
                "- Route to Fitness-for-Service (FFS) / Authorized Inspector review before any return-to-service decision."
            )
        elif result["status"] == "verified":
            st.success("🟢 **COMPLIANCE STATUS: VERIFIED SECURE**")
            st.markdown("- All captured readings are at or above the calculated safety ceiling. Continue routine monitoring interval.")
        elif result["status"] == "no_measurements":
            st.warning("A safety ceiling was discovered, but no measurement readings could be located in this report.")

        if result["points"]:
            with st.expander(f"Measurement array — {len(result['points'])} point(s) captured"):
                for p in sorted(result["points"], key=lambda p: p["value"]):
                    unit = f" {p['unit']}" if p["unit"] else ""
                    st.write(f"- {p['label']}: {p['value']:.4f}{unit}")

    # --- Anomaly harvester: always shown, independent of the pass/fail path ---
    if result["anomalies"]:
        st.markdown("### ⚠️ Unresolved Mechanical Anomaly Logs")
        for note in result["anomalies"]:
            st.markdown(f"<div class='anomaly-box'>{note}</div>", unsafe_allow_html=True)

    with st.expander("📚 Reference / Simulated RAG Context (supporting code clauses)"):
        for clause in REFERENCE_CONTEXT:
            st.write(f"- {clause}")

else:
    st.info("Paste or load a field report on the left, then click **Run Dynamic Analysis**.")
