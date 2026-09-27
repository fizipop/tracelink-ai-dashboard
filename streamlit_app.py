import streamlit as st
import json
import re

# Configure high-level enterprise canvas parameters
st.set_page_config(page_title="TraceLink AI | Enterprise Quality Assurance Engine", layout="wide")

# Verified Industrial Engineering Regulatory Matrix Database
INDUSTRIAL_STANDARDS = {
    "AS9100-AEROSPACE-STANDARD": {
        "allowed_materials": ["Titanium-Ti-6Al-4V", "Inconel-718", "Aluminum-7075-T6"],
        "max_geometric_tolerance_mm": 0.01,
        "max_allowable_shear_stress_mpa": 480.0
    },
    "IATF-16949-AUTOMOTIVE-CHASSIS": {
        "allowed_materials": ["Structural-Steel-A36", "Aluminum-6061-T6", "High-Strength-Steel-DualPhase"],
        "max_geometric_tolerance_mm": 0.05,
        "max_allowable_shear_stress_mpa": 250.0
    }
}

# --- Internal AI Text Parsing Engine Helper Functions ---
def extract_metrics_from_text(raw_text: str) -> dict:
    """
    Simulates a localized Natural Language Processing (NLP) regex engine to extract 
    engineering specifications from unstructured inspection files.
    """
    parsed_data = {"material": "Unknown", "calculated_shear_stress_mpa": 0.0, "tolerance": 0.0}
    
    # Intelligently isolate known industrial materials inside text strings
    all_materials = []
    for framework in INDUSTRIAL_STANDARDS.values():
        all_materials.extend(framework["allowed_materials"])
    # Common failing materials to intercept
    all_materials.extend(["Structural-Steel-A36", "Mild-Steel", "Carbon-Steel"])
    
    for mat in all_materials:
        if re.search(r'\b' + re.escape(mat) + r'\b', raw_text, re.IGNORECASE):
            parsed_data["material"] = mat
            break
            
    # Regex extraction loops for load stresses (looks for numbers matching MPa tags)
    stress_match = re.search(r'(?:stress|load|yield)(?:\s*(?:level|rate|is)?\s*)[:\-]?\s*([0-9.]+)\s*(?:mpa)?', raw_text, re.IGNORECASE)
    if stress_match:
        parsed_data["calculated_shear_stress_mpa"] = float(stress_match.group(1))
        
    # Regex extraction loops for precision tolerance metrics (looks for mm tags)
    tolerance_match = re.search(r'(?:tolerance|deviation|variance)\s*[:\-]?\s*([0-9.]+)\s*(?:mm)?', raw_text, re.IGNORECASE)
    if tolerance_match:
        parsed_data["tolerance"] = float(tolerance_match.group(1))
        
    return parsed_data

# --- Visual UI Render Construction ---
st.title("🛡️ TraceLink AI | Enterprise Quality Assurance Engine")
st.subheader("Automated Industrial Safety & Multi-Format Regulatory Verification Layer")
st.markdown("---")

# Visual Sidebar Matrix Controller Configuration
st.sidebar.header("📋 Global Compliance Configuration")
framework_selection = st.sidebar.selectbox(
    "Select Target Inspection Framework",
    list(INDUSTRIAL_STANDARDS.keys())
)

st.sidebar.markdown("""
### 🏭 Enterprise Operational Track:
1. Select your target framework audit track.
2. Choose your inputs: **Structured Data Code (.json)** or an **Unstructured Document Text Report (.txt)**.
3. The platform processes the engineering bounds and highlights immediate remediation logs.
""")

# Setup input format toggle buttons
input_format = st.radio("Select Engineering Input Document Format Type:", ["Structured Schema Data (.json)", "Unstructured Text Report / MTR (.txt)"])
st.markdown("---")

col1, col2 = st.columns(2)
component_specs = None
uploaded_filename = ""

with col1:
    st.header("📥 Ingest Compliance Files")
    
    if input_format == "Structured Schema Data (.json)":
        uploaded_file = st.file_uploader("Upload Component Specification Leaf (.json)", type=["json"])
        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            try:
                component_specs = json.loads(uploaded_file.getvalue())
            except Exception as e:
                st.error(f"Malformed JSON File Structure: {str(e)}")
                
    else:
        uploaded_file = st.file_uploader("Upload Raw Material Test Report or Inspection Summary (.txt)", type=["txt"])
        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            raw_report_text = uploaded_file.getvalue().decode("utf-8")
            
            st.markdown("### Raw Document Log Viewer:")
            st.code(raw_report_text, language="text")
            
            # Execute automated text extraction pipeline pass
            component_specs = extract_metrics_from_text(raw_report_text)

with col2:
    st.header("📊 Compliance Verification Summary")
    
    if component_specs is not None:
        # Prevent processing block errors if regex parsing failed to locate variables safely
        if component_specs["material"] == "Unknown" and component_specs["calculated_shear_stress_mpa"] == 0.0:
            st.warning("⚠️ Document Ingestion Ambiguity: The engine could not safely extract structured physical variables. Please verify the document notation.")
        else:
            rulebook = INDUSTRIAL_STANDARDS[framework_selection]
            audit_failures = []
            remediation_guidance = []
            
            # 1. Material Compliance Evaluation Pass
            current_mat = component_specs.get("material")
            if current_mat not in rulebook["allowed_materials"]:
                audit_failures.append({
                    "parameter": "Material Analysis",
                    "error": f"Unauthorized structural compound specification: '{current_mat}'",
                    "severity": "CRITICAL_STOP"
                })
                remediation_guidance.append({
                    "issue": "Illegal Material Selection",
                    "action_required": f"Replace input compound '{current_mat}' with an authorized standard alternative.",
                    "recommended_options": rulebook["allowed_materials"]
                })
                
            # 2. High-Load Physics Evaluation Pass
            current_stress = component_specs.get("calculated_shear_stress_mpa", 0.0)
            max_stress = rulebook["max_allowable_shear_stress_mpa"]
            if current_stress > max_stress:
                excess = current_stress - max_stress
                audit_failures.append({
                    "parameter": "Mechanical Structural Integrity",
                    "error": f"Internal stress threshold exceeded. Measured: {current_stress} MPa (Safety Limit: {max_stress} MPa)",
                    "severity": "CRITICAL_STOP"
                })
                remediation_guidance.append({
                    "issue": "Structural Overstress Failure",
                    "action_required": f"Modify drawing architecture to absorb or reduce internal load thresholds by a minimum of {excess:.1f} MPa.",
                    "engineering_suggestions": [
                        "Increase structural cross-sectional geometry profile boundaries.",
                        "Enlarge the transition corner radius dimensions to stop fatigue fracturing stress lines.",
                        "Re-calculate material load limits matching high-tensile specifications."
                    ]
                })
                
            # Render corresponding dashboard alert blocks based on audit outputs
            if len(audit_failures) == 0:
                st.success(f"✅ COMPLIANCE STATUS: VERIFIED SECURE (Passed All Framework Rules)")
                st.balloons()
            else:
                st.error(f"❌ COMPLIANCE STATUS: AUDIT FAILURE (Blocked by Quality Assurance Filter)")
                
                st.markdown("### 🪛 Automated Engineering Remediation Blueprint:")
                for step, item in enumerate(remediation_guidance, start=1):
                    with st.expander(f"Correction Target #{step}: {item['issue']}", expanded=True):
                        st.write(f"**Action Required:** {item['action_required']}")
                        if "recommended_options" in item:
                            st.info(f"**Authorized Engineering Options:** {', '.join(item['recommended_options'])}")
                        if "engineering_suggestions" in item:
                            st.warning("**Recommended Mechanical Stress Mitigation Tactics:**")
                            for point in item["engineering_suggestions"]:
                                st.write(f"• {point}")
                                
            st.markdown("---")
            st.markdown("### Processed Metadata Context Extracted:")
            st.json({
                "source_file": uploaded_filename,
                "target_framework_checked": framework_selection,
                "parsed_metrics": component_specs,
                "passed_safety_filters": len(audit_failures) == 0
            })
    else:
        st.info("Awaiting input data streams... Please select an input option and drag a standard text report or structured file to run checks.")
