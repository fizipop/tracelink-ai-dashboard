import streamlit as st
import json
import re

# Configure high-level enterprise canvas parameters
st.set_page_config(page_title="TraceLink AI | Enterprise Quality Assurance Engine", layout="wide")

# Production Material Database with Allowed Chemical Class Keywords instead of rigid string lists
REGULATORY_MATRIX = {
    "AS9100-AEROSPACE-STANDARD": {
        "allowed_material_classes": ["titanium", "inconel", "cobalt", "nickel"],
        "max_geometric_tolerance_mm": 0.01,
        "max_allowable_shear_stress_mpa": 480.0
    },
    "IATF-16949-AUTOMOTIVE-CHASSIS": {
        "allowed_material_classes": ["steel", "aluminum", "iron"],
        "max_geometric_tolerance_mm": 0.05,
        "max_allowable_shear_stress_mpa": 250.0
    }
}

def extract_metrics_from_text(raw_text: str) -> dict:
    """
    NLP parser using flexible token matching to extract engineering specifications 
    from completely unformatted text records.
    """
    parsed_data = {"material": "Unknown", "calculated_shear_stress_mpa": 0.0, "tolerance": 0.0}
    
    # Capture any word sequence preceding common manufacturing descriptor tags
    material_match = re.search(r'(?:material|alloy|compound)\s*(?:is|type|selection)?\s*[:\-]?\s*([a-zA-Z0-9\-_\s]+)', raw_text, re.IGNORECASE)
    if material_match:
        # Clean up whitespace syntax
        parsed_data["material"] = material_match.group(1).strip().split('\n')[0]
        
    stress_match = re.search(r'(?:stress|load|yield)\s*[:\-]?\s*([0-9.]+)\s*(?:mpa)?', raw_text, re.IGNORECASE)
    if stress_match:
        parsed_data["calculated_shear_stress_mpa"] = float(stress_match.group(1))
        
    tolerance_match = re.search(r'(?:tolerance|deviation)\s*[:\-]?\s*([0-9.]+)\s*(?:mm)?', raw_text, re.IGNORECASE)
    if tolerance_match:
        parsed_data["tolerance"] = float(tolerance_match.group(1))
        
    return parsed_data

def evaluate_material_compliance(input_material: str, allowed_classes: list) -> tuple:
    """
    Executes a flexible semantic keyword verification algorithm.
    Allows variations like 'Ti-6Al-4V Grade 5 Titanium' to pass an 'titanium' rule class safely.
    """
    clean_input = input_material.lower()
    for material_class in allowed_classes:
        if material_class in clean_input:
            return True, material_class
    return False, None

# --- Visual UI Render Construction ---
st.title("🛡️ TraceLink AI | Enterprise Quality Assurance Engine")
st.subheader("Automated Industrial Safety & Multi-Format Regulatory Verification Layer")
st.markdown("---")

st.sidebar.header("📋 Global Compliance Configuration")
framework_selection = st.sidebar.selectbox(
    "Select Target Inspection Framework",
    list(REGULATORY_MATRIX.keys())
)

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
        uploaded_file = st.file_uploader("Upload Raw Material Test Report (.txt)", type=["txt"])
        if uploaded_file is not None:
            uploaded_filename = uploaded_file.name
            raw_report_text = uploaded_file.getvalue().decode("utf-8")
            
            st.markdown("### Raw Document Log Viewer:")
            st.code(raw_report_text, language="text")
            component_specs = extract_metrics_from_text(raw_report_text)

with col2:
    st.header("📊 Compliance Verification Summary")
    
    if component_specs is not None:
        rulebook = REGULATORY_MATRIX[framework_selection]
        audit_failures = []
        remediation_guidance = []
        
        # Execute Flexible Material Audit Pass
        input_mat_string = component_specs.get("material", "Unknown")
        is_compliant_material, matched_class = evaluate_material_compliance(
            input_mat_string, 
            rulebook["allowed_material_classes"]
        )
        
        if not is_compliant_material:
            audit_failures.append({
                "parameter": "Material Class Validation",
                "error": f"Material structure '{input_mat_string}' does not align with authorized framework compounds.",
                "severity": "CRITICAL_STOP"
            })
            remediation_guidance.append({
                "issue": "Unauthorized Base Metal Class",
                "action_required": f"Change component material allocation to a confirmed compound within these verified structural families: {rulebook['allowed_material_classes']}"
            })
            
        # Execute High-Load Structural Pass
        current_stress = component_specs.get("calculated_shear_stress_mpa", 0.0)
        max_stress = rulebook["max_allowable_shear_stress_mpa"]
        if current_stress > max_stress:
            excess = current_stress - max_stress
            audit_failures.append({
                "parameter": "Mechanical Structural Integrity",
                "error": f"Internal stress threshold exceeded. Measured: {current_stress} MPa (Limit: {max_stress} MPa)",
                "severity": "CRITICAL_STOP"
            })
            remediation_guidance.append({
                "issue": "Structural Overstress Failure",
                "action_required": f"Modify engineering drawing coordinates to absorb or reduce localized internal load thresholds by a minimum of {excess:.1f} MPa."
            })
            
        # Draw UI Alert Components Based on Dynamic Outcomes
        if len(audit_failures) == 0:
            st.success(f"✅ COMPLIANCE STATUS: VERIFIED SECURE")
            st.info(f"**Audit Context:** Material matched authorized family class: '{matched_class.upper()}'")
            st.balloons()
        else:
            st.error(f"❌ COMPLIANCE STATUS: AUDIT FAILURE (Blocked by Quality Filter)")
            
            st.markdown("### 🪛 Automated Engineering Remediation Blueprint:")
            for step, item in enumerate(remediation_guidance, start=1):
                st.warning(f"**Correction target #{step}: {item['issue']}**")
                st.write(f"• {item['action_required']}")
                
        st.markdown("---")
        st.markdown("### Extracted Ingestion Context Meta:")
        st.json({
            "source_file": uploaded_filename,
            "target_framework_checked": framework_selection,
            "extracted_metrics": component_specs,
            "passed_safety_filters": len(audit_failures) == 0
        })
