"""
TraceLink AI — Engineering Compliance Validation Engine
--------------------------------------------------------
Single-file Streamlit application.

Design notes (read before modifying):
  * No `re` module anywhere. All value extraction is done with plain string
    scanning (`extract_number`) so there is no catastrophic-backtracking or
    silent-mismatch risk that regex-based extraction was producing.
  * No `chromadb` or any vector-DB import. The "RAG" reference context is a
    plain Python dict of lists (SIMULATED_RAG_LOG) — zero native deps, zero
    sqlite version conflicts on Streamlit Cloud.
  * No live AI API calls. The "structured AI extraction" step is a
    deterministic label -> value parser (line_dict / find_number /
    collect_numbered_points) that mimics the JSON a structured-output LLM
    call would have returned, without the latency, cost, or key-management
    surface area.
  * Each compliance jurisdiction (track) has its own analyzer function that
    only applies the formulas valid for that asset class — a pressure-vessel
    MAT check can never leak into a piping report and vice versa.
"""

import streamlit as st

st.set_page_config(
    page_title="TraceLink AI — Compliance Engine",
    layout="wide",
    page_icon="🛠️",
)

# ============================================================================
# 1. CORE PARSER (replaces regex entirely)
# ============================================================================

NUMERIC_CHARS = set("0123456789.-")


def extract_number(raw):
    """Pull the first signed decimal number out of a string, character by
    character. Returns None if no number is present. No regex."""
    if not raw:
        return None
    token = ""
    started = False
    for ch in raw:
        if ch in NUMERIC_CHARS:
            token += ch
            started = True
        elif started:
            break
    if token in ("", ".", "-", "-."):
        return None
    try:
        return float(token)
    except ValueError:
        return None


def line_dict(text):
    """Turn 'Label: value' lines into a {lowercase label: value} map."""
    fields = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" in line:
            label, _, value = line.partition(":")
            fields[label.strip().lower()] = value.strip()
    return fields


def find_value(fields, *keywords):
    """First field whose label contains every keyword (order-independent)."""
    for label, value in fields.items():
        if all(k.lower() in label for k in keywords):
            return value
    return None


def find_number(fields, *keywords):
    val = find_value(fields, *keywords)
    return extract_number(val) if val is not None else None


def collect_numbered_points(text, label_keywords):
    """Collect (label, value) pairs from every 'Label: number' line whose
    label contains one of label_keywords. Used for UT reading arrays."""
    points = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        label_low = label.strip().lower()
        if any(k in label_low for k in label_keywords):
            num = extract_number(value)
            if num is not None:
                points.append((label.strip(), num))
    return points


def collect_notes(text, tag="field note"):
    """Collect free-text anomaly notes (weld flaws, bolt condition, etc.)."""
    notes = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        if tag in label.strip().lower():
            notes.append(value.strip())
    return notes


# ============================================================================
# 2. TRACK AUTO-DETECTION
# ============================================================================

def detect_track(text):
    fields = line_dict(text)
    t = text.lower()

    is_structural = any(k in t for k in ("d1.1", "aisc", "hss", "gusset", "structural"))
    is_vessel = any(k in t for k in ("section viii", "pressure vessel", "sa-516"))
    is_piping = any(k in t for k in ("b31.3", "process piping"))

    has_full_math_vars = (
        find_number(fields, "design pressure") is not None
        and find_number(fields, "outside diameter") is not None
        and find_number(fields, "allowable stress") is not None
    )

    if is_structural:
        return "T4"
    if is_vessel:
        return "T1"
    if is_piping and has_full_math_vars:
        return "T3"
    if is_piping:
        return "T2"
    return "T1"


TRACK_LABELS = {
    "T1": "Jurisdiction 1 — ASME Sec. VIII Pressure Vessel",
    "T2": "Jurisdiction 2 — ASME B31.3 Piping (MAT Missing)",
    "T3": "Jurisdiction 3 — ASME B31.3 Piping (Math Engine)",
    "T4": "Jurisdiction 4 — AWS D1.1 / AISC Structural Steel",
}

# ============================================================================
# 3. SAMPLE REPORTS (used to seed the input box / demo the parser)
# ============================================================================

SAMPLES = {
    "T1": """ASME SECTION VIII PRESSURE VESSEL — FIELD INSPECTION REPORT
Asset Category: Horizontal Hydraulic Pressure Vessel
Metallurgy: SA-516 Grade 70 carbon steel
Nominal Shell Thickness: 0.500 in
Minimum Allowable Shell Thickness (MAT): 0.375 in
UT Point A1: 0.382
UT Point A2: 0.371
UT Point A3: 0.364
UT Point A4: 0.379
UT Point A5: 0.388
""",
    "T2": """ASME B31.3 PROCESS PIPING — FIELD INSPECTION REPORT
Asset Category: Process Piping System
Metallurgy: ASTM A106 Grade B carbon steel
Design Pressure: 500 psi
Outside Diameter: 4.500 in
UT Point P1: 0.095
UT Point P2: 0.088
UT Point P3: 0.091
""",
    "T3": """ASME B31.3 PROCESS PIPING — MATH ENGINE VALIDATION REPORT
Asset Category: Process Piping System
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
    "T4": """AWS D1.1 / AISC STRUCTURAL STEEL FRAMING — FIELD INSPECTION REPORT
Asset Category: Welded Structural Support Frame / HSS Assembly
Metallurgy: ASTM A500 Grade B
Drawing HSS Minimum Thickness: 5.50 mm
Drawing Gusset Minimum Thickness: 9.50 mm
West HSS UT Reading: 5.82 mm
Gusset UT Reading: 10.6 mm
Field Note: Possible weld flaw at north joint - NDT required
Field Note: Corroded and un-torqued 19mm anchor bolt at base plate
Field Note: Unmapped 3mm alignment variance on brace connection
""",
}

# ============================================================================
# 4. SIMULATED RAG REFERENCE CONTEXT (plain dict — no vector DB dependency)
# ============================================================================

SIMULATED_RAG_LOG = {
    "T1": [
        "ASME BPVC Sec. VIII Div.1 UG-25: minimum thickness after corrosion "
        "shall not be less than the design MAT.",
        "API 510 6.5: components below MAT require immediate engineering "
        "disposition before return to service.",
    ],
    "T2": [
        "ASME B31.3 304.1.2: pressure design thickness (t) must be "
        "calculated from P, D, S, E, Y before a minimum-thickness verdict "
        "can be issued.",
        "Piping MAT is asset-specific and is NOT interchangeable with "
        "vessel MAT — do not carry over Sec. VIII limits.",
    ],
    "T3": [
        "ASME B31.3 Eq. 3a: t = (P x D) / (2 x (S x E + P x Y)).",
        "Total minimum wall = t_design + corrosion allowance (CA), per "
        "B31.3 304.1.1.",
    ],
    "T4": [
        "AWS D1.1 Table 6.1: acceptance criteria for visual weld "
        "discontinuities; suspect indications require NDT disposition.",
        "AISC 360 J3: bolted connections must meet specified pretension; "
        "corrosion or under-torque requires re-inspection before load "
        "rating is confirmed.",
    ],
}

# ============================================================================
# 5. TRACK ANALYZERS
# ============================================================================

def analyze_t1(text):
    fields = line_dict(text)
    asset = find_value(fields, "asset category") or "Unspecified Asset"
    metallurgy = find_value(fields, "metallurgy") or "Unspecified"
    nominal = find_number(fields, "nominal", "thickness")
    mat = find_number(fields, "minimum allowable") or find_number(fields, "mat")
    points = collect_numbered_points(text, ["ut point", "point"])
    readings = [v for _, v in points]
    min_ut = min(readings) if readings else None

    result = {
        "track": "T1", "asset": asset, "metallurgy": metallurgy,
        "nominal": nominal, "mat": mat, "points": points, "min_ut": min_ut,
        "margin": None, "verdict": "UNKNOWN", "severity": "warning",
        "remediation": [],
    }

    if min_ut is None or mat is None:
        result["verdict"] = "INSUFFICIENT DATA — MAT or UT readings missing."
        result["severity"] = "warning"
        return result

    result["margin"] = round(min_ut - mat, 3)
    if min_ut < mat:
        result["verdict"] = "CRITICAL — ASME SEC VIII BLOCKED"
        result["severity"] = "error"
        result["remediation"] = [
            "Lowest UT reading is below the Minimum Allowable Thickness (MAT).",
            "Vessel is BLOCKED from further pressurization pending disposition.",
            "Notify the Authorized Inspector (AI) and initiate a Fitness-for-Service (FFS) review per API 579.",
            "Do not apply piping (B31.3) formulas to this asset — vessel MAT governs exclusively.",
        ]
    else:
        result["verdict"] = "PASS — Shell thickness above MAT"
        result["severity"] = "success"
        result["remediation"] = [
            "No immediate action required; continue routine UT monitoring interval.",
        ]
    return result


def analyze_t2(text):
    fields = line_dict(text)
    asset = find_value(fields, "asset category") or "Unspecified Asset"
    metallurgy = find_value(fields, "metallurgy") or "Unspecified"
    design_p = find_number(fields, "design pressure")
    outside_d = find_number(fields, "outside diameter")
    mat = find_number(fields, "minimum allowable") or find_number(fields, "mat")
    points = collect_numbered_points(text, ["ut point", "point"])
    readings = [v for _, v in points]
    min_ut = min(readings) if readings else None

    result = {
        "track": "T2", "asset": asset, "metallurgy": metallurgy,
        "design_p": design_p, "outside_d": outside_d, "points": points,
        "min_ut": min_ut, "mat": mat, "verdict": "UNKNOWN", "severity": "warning",
        "remediation": [],
    }

    if mat is None:
        result["verdict"] = "CONDITION UNVERIFIED — INSUFFICIENT BOUNDARY DATA"
        result["severity"] = "warning"
        result["remediation"] = [
            "No Minimum Allowable Thickness (MAT) is specified in this report.",
            "The system will NOT assume failure and will NOT apply pressure-vessel equations to a piping asset.",
            "A B31.3 straight-pipe wall calculation (t = PD / 2(SE+PY) + CA) must be executed "
            "using the design pressure, diameter, allowable stress, quality factor, Y coefficient, "
            "and corrosion allowance before any safety verdict can be issued.",
            "Route this report to Jurisdiction 3 (Math Engine) once the missing variables are supplied.",
        ]
        return result

    # MAT was supplied after all — fall through to a normal comparison.
    if min_ut is not None:
        result["margin"] = round(min_ut - mat, 3)
        if min_ut < mat:
            result["verdict"] = "CRITICAL — ASME B31.3 VIOLATION"
            result["severity"] = "error"
        else:
            result["verdict"] = "PASS — Above MAT"
            result["severity"] = "success"
    return result


def analyze_t3(text):
    fields = line_dict(text)
    asset = find_value(fields, "asset category") or "Unspecified Asset"
    metallurgy = find_value(fields, "metallurgy") or "Unspecified"

    P = find_number(fields, "design pressure")
    D = find_number(fields, "outside diameter")
    S = find_number(fields, "allowable stress")
    E = find_number(fields, "quality factor")
    Y = find_number(fields, "y coefficient") or find_number(fields, "y ")
    CA = find_number(fields, "corrosion allowance")

    points = collect_numbered_points(text, ["ut point", "point"])
    readings = [v for _, v in points]
    min_ut = min(readings) if readings else None

    result = {
        "track": "T3", "asset": asset, "metallurgy": metallurgy,
        "P": P, "D": D, "S": S, "E": E, "Y": Y, "CA": CA,
        "points": points, "min_ut": min_ut,
        "t_design": None, "total_mat": None, "margin": None,
        "verdict": "UNKNOWN", "severity": "warning", "remediation": [],
    }

    vars_present = all(v is not None for v in (P, D, S, E, Y, CA))
    if not vars_present:
        result["verdict"] = "INSUFFICIENT DATA — one or more of P, D, S, E, Y, CA missing"
        result["severity"] = "warning"
        return result

    t_design = (P * D) / (2 * (S * E + P * Y))
    total_mat = t_design + CA
    result["t_design"] = round(t_design, 4)
    result["total_mat"] = round(total_mat, 4)

    if min_ut is None:
        result["verdict"] = "MATH COMPLETE — no UT readings to compare"
        result["severity"] = "warning"
        return result

    result["margin"] = round(min_ut - total_mat, 4)
    if min_ut < total_mat:
        result["verdict"] = "CRITICAL — ASME B31.3 VIOLATION"
        result["severity"] = "error"
        result["remediation"] = [
            f"Lowest field UT ({min_ut:.3f} in) is below the calculated Total MAT ({total_mat:.4f} in).",
            "Pipe segment fails ASME B31.3 minimum wall requirement — remove from service or de-rate pressure.",
            "Confirm P, D, S, E, Y inputs against the current line list before repair scope is finalized.",
        ]
    else:
        result["verdict"] = "PASS — Above calculated Total MAT"
        result["severity"] = "success"
        result["remediation"] = [
            "Segment meets calculated minimum wall; continue scheduled UT monitoring interval.",
        ]
    return result


def analyze_t4(text):
    fields = line_dict(text)
    asset = find_value(fields, "asset category") or "Unspecified Asset"
    metallurgy = find_value(fields, "metallurgy") or "Unspecified"

    hss_limit = find_number(fields, "drawing hss")
    gusset_limit = find_number(fields, "drawing gusset")
    hss_ut = find_number(fields, "hss ut") or find_number(fields, "west hss")
    gusset_ut = find_number(fields, "gusset ut")

    notes = collect_notes(text)

    thickness_ok = None
    if None not in (hss_limit, gusset_limit, hss_ut, gusset_ut):
        thickness_ok = (hss_ut > hss_limit) and (gusset_ut > gusset_limit)

    has_anomalies = len(notes) > 0

    if has_anomalies:
        verdict = "CONDITION NOT FULLY VERIFIED — ENGINEERING REVIEW REQUIRED"
        severity = "warning"
    elif thickness_ok:
        verdict = "PASS — Thickness Criteria: MET"
        severity = "success"
    elif thickness_ok is False:
        verdict = "CRITICAL — Thickness Criteria: NOT MET"
        severity = "error"
    else:
        verdict = "INSUFFICIENT DATA"
        severity = "warning"

    return {
        "track": "T4", "asset": asset, "metallurgy": metallurgy,
        "hss_limit": hss_limit, "gusset_limit": gusset_limit,
        "hss_ut": hss_ut, "gusset_ut": gusset_ut,
        "thickness_ok": thickness_ok, "notes": notes,
        "verdict": verdict, "severity": severity,
        "remediation": (
            [f"Unresolved field anomaly: {n}" for n in notes]
            + ["Bypasses pressure-vessel and piping pressure equations entirely — thickness-only structural check."]
        ) if has_anomalies else [
            "Bypasses pressure-vessel and piping pressure equations entirely — thickness-only structural check.",
        ],
    }


ANALYZERS = {"T1": analyze_t1, "T2": analyze_t2, "T3": analyze_t3, "T4": analyze_t4}

# ============================================================================
# 6. UI
# ============================================================================

st.markdown(
    """
    <style>
    .metric-box {background:#111827;border:1px solid #374151;border-radius:10px;
        padding:14px 18px;margin-bottom:10px;}
    .metric-label {color:#9CA3AF;font-size:0.78rem;text-transform:uppercase;letter-spacing:.04em;}
    .metric-value {color:#F9FAFB;font-size:1.35rem;font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛠️ TraceLink AI — Engineering Compliance Validation Engine")
st.caption(
    "Programmatic parser + rules engine — no regex, no vector-DB import, no live AI key required."
)

with st.sidebar:
    st.header("Input")
    mode = st.radio(
        "Jurisdiction routing",
        ["Auto-Detect from Report Text"] + [TRACK_LABELS[k] for k in ("T1", "T2", "T3", "T4")],
    )
    label_to_key = {v: k for k, v in TRACK_LABELS.items()}
    preload_key = label_to_key.get(mode, "T1")
    st.caption("Load a sample report, or paste your own below.")
    if st.button("Load sample for this selection", use_container_width=True):
        st.session_state["report_text"] = SAMPLES[preload_key]

    if "report_text" not in st.session_state:
        st.session_state["report_text"] = SAMPLES["T1"]

    report_text = st.text_area(
        "Field report text", value=st.session_state["report_text"], height=340, key="report_input"
    )
    run = st.button("▶ Run Compliance Analysis", type="primary", use_container_width=True)

if run:
    track_key = detect_track(report_text) if mode == "Auto-Detect from Report Text" else preload_key
    result = ANALYZERS[track_key](report_text)

    st.subheader(TRACK_LABELS[track_key])
    top1, top2, top3 = st.columns(3)
    top1.markdown(f"<div class='metric-box'><div class='metric-label'>Asset Category</div>"
                   f"<div class='metric-value'>{result['asset']}</div></div>", unsafe_allow_html=True)
    top2.markdown(f"<div class='metric-box'><div class='metric-label'>Metallurgy</div>"
                   f"<div class='metric-value'>{result['metallurgy']}</div></div>", unsafe_allow_html=True)
    top3.markdown(f"<div class='metric-box'><div class='metric-label'>Track</div>"
                   f"<div class='metric-value'>{track_key}</div></div>", unsafe_allow_html=True)

    # --- Track-specific metric grids -------------------------------------
    if track_key == "T1":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nominal Thickness (in)", f"{result['nominal']:.3f}" if result['nominal'] is not None else "—")
        c2.metric("MAT (in)", f"{result['mat']:.3f}" if result['mat'] is not None else "—")
        c3.metric("Lowest UT (in)", f"{result['min_ut']:.3f}" if result['min_ut'] is not None else "—")
        c4.metric("Margin (in)", f"{result['margin']:+.3f}" if result['margin'] is not None else "—")
        with st.expander("UT reading log"):
            for label, val in result["points"]:
                st.write(f"- {label}: {val:.3f} in")

    elif track_key == "T2":
        c1, c2, c3 = st.columns(3)
        c1.metric("Design Pressure (psi)", result["design_p"] if result["design_p"] is not None else "—")
        c2.metric("Outside Diameter (in)", result["outside_d"] if result["outside_d"] is not None else "—")
        c3.metric("MAT specified?", "No" if result["mat"] is None else f"{result['mat']:.3f} in")
        if result["points"]:
            with st.expander("UT reading log (context only — not yet evaluated)"):
                for label, val in result["points"]:
                    st.write(f"- {label}: {val:.3f} in")

    elif track_key == "T3":
        st.markdown("**Calculation Parameter Grid**")
        g1, g2, g3, g4, g5, g6 = st.columns(6)
        g1.metric("P (psi)", result["P"])
        g2.metric("D (in)", result["D"])
        g3.metric("S (psi)", result["S"])
        g4.metric("E", result["E"])
        g5.metric("Y", result["Y"])
        g6.metric("CA (in)", result["CA"])
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("t_design (in)", result["t_design"])
        r2.metric("Total MAT (in)", result["total_mat"])
        r3.metric("Lowest UT (in)", f"{result['min_ut']:.3f}" if result['min_ut'] is not None else "—")
        r4.metric("Margin (in)", f"{result['margin']:+.4f}" if result['margin'] is not None else "—")
        st.latex(r"t_{design} = \frac{P \times D}{2 \times (S \times E + P \times Y)}, \quad \text{Total MAT} = t_{design} + CA")
        with st.expander("UT reading log"):
            for label, val in result["points"]:
                st.write(f"- {label}: {val:.3f} in")

    elif track_key == "T4":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HSS Limit (mm)", result["hss_limit"])
        c2.metric("HSS UT (mm)", result["hss_ut"])
        c3.metric("Gusset Limit (mm)", result["gusset_limit"])
        c4.metric("Gusset UT (mm)", result["gusset_ut"])
        st.write(f"Thickness Criteria: **{'MET' if result['thickness_ok'] else 'NOT MET'}**")
        if result["notes"]:
            with st.expander("Unresolved field anomalies", expanded=True):
                for n in result["notes"]:
                    st.write(f"- {n}")

    # --- Verdict banner -----------------------------------------------------
    banner = {"error": st.error, "warning": st.warning, "success": st.success}[result["severity"]]
    banner(f"**Verdict:** {result['verdict']}")

    # --- Remediation ----------------------------------------------------
    if result["remediation"]:
        st.markdown("**Remediation / Next Steps**")
        for item in result["remediation"]:
            st.markdown(f"- {item}")

    # --- Simulated RAG context ------------------------------------------
    with st.expander("Simulated RAG Retrieval Log (internal reference context)"):
        for clause in SIMULATED_RAG_LOG.get(track_key, []):
            st.write(f"- {clause}")

else:
    st.info("Paste or load a field report on the left, then click **Run Compliance Analysis**.")
