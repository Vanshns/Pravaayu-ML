
import pandas as pd
import json

# ─────────────────────────────────────────────────────────────────────────────
# CRF Feature Extraction — Full Pipeline
# Flattens a patient JSON into a fixed-width feature dict for ML / decision tree
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared helpers ─────────────────────────────────────────────────────────

def normalize(val):
    if val is None:
        return ""
    return str(val).strip().lower()


def yn(val):
    """'Yes'/'No' → 1/0. Returns 0 for missing."""
    return 1 if normalize(val) == "yes" else 0


# def extract_duration(durations: dict, key: str) -> dict:
#     d = durations.get(key, {})
#     return {
#         "duration": int(d.get("val", 0) or 0),
#         "duration_type": normalize(d.get("unit", "days"))
#     }

def extract_duration(durations: dict, key: str) -> dict:
    # 1. Get the data for the key
    d = durations.get(key, {})
    
    # 2. Check if 'd' is a dictionary. If it's a string or something else, 
    # we convert it to a format the rest of the function expects.
    if not isinstance(d, dict):
        # If it's a standalone number/string, we treat it as the 'val'
        d = {"val": d, "unit": "days"}

    # 3. Safely extract and cast the duration
    try:
        raw_val = d.get("val", 0)
        # Handle cases where val might be None or an empty string
        duration_val = int(raw_val) if raw_val and str(raw_val).isdigit() else 0
    except (ValueError, TypeError):
        duration_val = 0

    return {
        "duration": duration_val,
        "duration_type": normalize(d.get("unit", "days"))
    }


def pain_15(val):
    """'Less than 15 mins' → 1,  'More than 15 mins' → 2,  missing → 0."""
    v = normalize(val)
    if v == "less than 15 mins":
        return 1
    if v == "more than 15 mins":
        return 2
    return 0


# def dur_val(d: dict) -> int:
#     """Extract numeric value from a duration dict. Returns 0 if missing."""
#     return int(d.get("val", 0) or 0) if isinstance(d, dict) else 0

def dur_val(d):
    if not isinstance(d, dict):
        return 0
    
    val = d.get("val", 0)

    try:
        return float(val)   # 👈 handles 1, 1.5, "2", etc.
    except:
        return 0


def lifting_kg(val):
    """'2kg'→2, '3kg'→3 … 'More than 5kg'→6, missing→0."""
    v = normalize(val)
    mapping = {"2kg": 2, "3kg": 3, "4kg": 4, "5kg": 5, "more than 5kg": 6}
    return mapping.get(v, 0)


def rom_encode(val):
    """'Free'→0, 'Restricted'→1, missing→-1."""
    v = normalize(val)
    if v == "free":      return 0
    if v == "restricted": return 1
    return -1


def restriction_encode(val):
    """'Mild'→1, 'Moderate'→2, 'Heavy'→3, missing→0."""
    v = normalize(val)
    mapping = {"mild": 1, "moderate": 2, "heavy": 3}
    return mapping.get(v, 0)


def pain_encode(val):
    """'Painful'→1, 'Non-Painful'→0, missing→-1."""
    v = normalize(val)
    if v == "painful":     return 1
    if v == "non-painful": return 0
    return -1


def temp_encode(val):
    """'Normal'→0, 'Ushna'→1, missing→-1."""
    v = normalize(val)
    if v == "normal": return 0
    if v == "ushna":  return 1
    return -1


def creps_encode(val):
    """'Grade 1'→1, 'Grade 2'→2, 'Grade 3'→3, missing→0."""
    v = normalize(val)
    mapping = {"grade 1": 1, "grade 2": 2, "grade 3": 3}
    return mapping.get(v, 0)


def slr_val(val):
    """Extract numeric degree from SLR string, e.g. '50°' → 50. Returns 0 for missing/negative."""
    if not val:
        return 0
    v = str(val).replace("°", "").replace("-ve", "").strip()
    try:
        return int(v)
    except ValueError:
        return 0


def is_negative_slr(val):
    """Returns 1 if SLR is explicitly -ve."""
    return 1 if "-ve" in str(val).lower() else 0


# ── Step 1: Basic patient / systemic info ──────────────────────────────────

def transform_patient_systemic(raw: dict) -> dict:
    f = {}

    f["age"]    = int(raw.get("age", 0) or 0)
    f["weight"] = float(raw.get("weight", 0) or 0)

    sex = normalize(raw.get("sex"))
    f["sex_male"]   = 1 if sex == "male"   else 0
    f["sex_female"] = 1 if sex == "female" else 0

    f["acidity"] = yn(raw.get("acidity"))
    f["gas"]     = yn(raw.get("gas"))

    stool = normalize(raw.get("stool"))
    f["stool_satisfactory"]   = 1 if stool == "satisfactory"   else 0
    f["stool_unsatisfactory"] = 1 if stool == "unsatisfactory" else 0

    appetite = normalize(raw.get("appetite"))
    for opt in ["increase", "decrease", "irregular", "normal"]:
        f[f"appetite_{opt}"] = 1 if appetite == opt else 0

    urine = normalize(raw.get("urine"))
    urine_options = ["normal", "dribbling", "increase", "nocturnal", "burning"]
    for opt in urine_options:
        f[f"urine_{opt}"] = 1 if urine == opt else 0
    f["urine_unknown"] = 1 if urine not in urine_options else 0

    tongue = normalize(raw.get("tongue"))
    f["tongue_nirama"] = 1 if tongue == "nirama" else 0
    f["tongue_sama"]   = 1 if tongue in ("sama", "saamha") else 0

    nidra = normalize(raw.get("nidra"))
    f["nidra_sound"]    = 1 if nidra == "sound"    else 0
    f["nidra_disturbed"] = 1 if nidra == "disturbed" else 0

    return f


# ── Step 2: Medical history ─────────────────────────────────────────────────

def transform_medical_history_features(raw: dict) -> dict:
    f = {}

    comorb   = raw.get("comorbidities_json", {})
    selected = set(comorb.get("selected", []))
    durations = comorb.get("durations", {})

    comorb_map = {
        "Hypertension":     "hypertension",
        "Diabetes Type 1":  "diabetes_type_1",
        "Diabetes Type 2":  "diabetes_type_2",
        "Hyper Cholesterol":"hyper_cholesterol",
        "Hypothyroid":      "hypothyroid",
        "Any Heart Disease":"heart_disease",
        "PCOD":             "pcod",
        "Cancer":           "cancer",
        "GERD":             "gerd",
        "Skin Infections":  "skin_infection",
        "Hyperacidity":     "hyperacidity",
        "Other":            "comorb_other",
    }
    for raw_name, key in comorb_map.items():
        f[key] = 1 if raw_name in selected else 0
        dur = extract_duration(durations, raw_name)
        f[f"{key}_duration_days"] = dur["duration"]

    med = normalize(comorb.get("medication", ""))
    f["medication_none"]    = 1 if med in ("", "none") else 0
    f["medication_ongoing"] = 0 if med in ("", "none") else 1

    surg      = raw.get("surgical_history_json", {})
    selected  = set(surg.get("selected", []))
    durations = surg.get("durations", {})

    surg_map = {
        "Operation for Hernia":  "hernia_operation",
        "K-Wire Fixation":       "k_wire_fixation",
        "Angioplasty":           "angioplasty",
        "Open Heart Surgery":    "open_heart_surgery",
        "Appendicectomy":        "appendicectomy",
        "Laminectomy (Spine)":   "laminectomy_spine",
        "Decompression (Spine)": "decompression_spine",
        "Other":                 "surgery_other",
    }
    for raw_name, key in surg_map.items():
        f[key] = 1 if raw_name in selected else 0
        dur = extract_duration(durations, raw_name)
        f[f"{key}_duration_days"] = dur["duration"]

    return f


# ── Step 3: Ailment feature extractors ─────────────────────────────────────
# Each function accepts the ailment's `details` dict and a `prefix` string.
# All features are written with that prefix so multiple ailments don't clash.

def _side_prefix(prefix, side):
    """e.g. prefix='knee', side='left'  →  'knee_l_'"""
    s = "l" if side == "left" else "r"
    return f"{prefix}_{s}_"



# ── Lumbar ──────────────────────────────────────────────────────────────────

def extract_lumbar(details: dict, prefix: str = "lumbar") -> dict:
    f = {}
    p = f"{prefix}_"

    f[f"{p}duration_val"]  = dur_val(details.get("durationCombined", {}))

    f[f"{p}pain_walking"]        = pain_15(details.get("pain_walking"))
    f[f"{p}pain_walking_dur"]    = dur_val(details.get("pain_walking_duration", {}))
    f[f"{p}pain_lowsitting"]     = pain_15(details.get("pain_lowSitting"))
    f[f"{p}pain_lowsitting_dur"] = dur_val(details.get("pain_lowSitting_duration", {}))
    f[f"{p}pain_highsitting"]    = pain_15(details.get("pain_highSitting"))
    f[f"{p}pain_highsitting_dur"]= dur_val(details.get("pain_highSitting_duration", {}))
    f[f"{p}pain_stairclimbing"]   = pain_15(details.get("pain_stairClimbing"))
    f[f"{p}pain_stairclimbing_dur"]= dur_val(details.get("pain_stairClimbing_duration", {}))
    f[f"{p}pain_standing"]       = pain_15(details.get("pain_standing"))
    f[f"{p}pain_standing_dur"]   = dur_val(details.get("pain_standing_duration", {}))

    f[f"{p}morning_stiffness"]          = yn(details.get("morningStiffness"))
    f[f"{p}morning_stiffness_duration"] = int(details.get("stiffnessDuration", 0) or 0)

    reason = normalize(details.get("ailmentReason", ""))
    f[f"{p}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{p}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(details.get("traumaType", ""))
    f[f"{p}trauma_fall"]        = 1 if trauma_type == "fall"        else 0
    f[f"{p}trauma_slip_impact"] = 1 if trauma_type == "slip impact" else 0
    f[f"{p}trauma_duration_val"]= dur_val(details.get("traumaDurationCombined", {}))

    related = [normalize(x) for x in details.get("painRelated", [])]
    f[f"{p}related_radiating"] = 1 if "radiating" in related else 0
    f[f"{p}related_tingling"]  = 1 if "tingling"  in related else 0

    le = details.get("localExam", {})
    f[f"{p}exam_tenderness"] = yn(le.get("tenderness"))
    f[f"{p}exam_swelling"]   = yn(le.get("swelling"))
    f[f"{p}exam_creps"]      = creps_encode(le.get("creps"))

    # --- Updated Diagnosis Logic (One-Hot Encoding) ---
    diag_list = [
        "herniated disc", "spinal stenosis", "spondylisthesis",
        "muscle spasm", "ankylosing spondylitis", "sciatica",
        "lumbar spondylosis", "arthritis"
    ]
    
    # Get user selection (handles string or list)
    user_diag = details.get("diagnosis", [])
    if isinstance(user_diag, str):
        user_diag = [user_diag]
    
    selected_diags = [normalize(d) for d in user_diag]

    # Create binary columns for every possible lumbar diagnosis
    for d_name in diag_list:
        # Standardize "athritis" to "arthritis" if needed for the feature name
        clean_name = d_name.replace(" ", "_")
        f[f"{p}diag_{clean_name}"] = 1 if d_name in selected_diags else 0

    # Ensure this module is flagged as present in the final vector
    f[f"{prefix}_present"] = 1 if details else 0

    return f




# ── Cervical ─────────────────────────────────────────────────────────────────

def extract_cervical(details: dict, prefix: str = "cervical") -> dict:
    f = {}
    p = f"{prefix}_"

    f[f"{p}duration_val"] = dur_val(details.get("durationCombined", {}))

    f[f"{p}pain_lifting_weight"]     = lifting_kg(details.get("pain_liftingWeight"))
    f[f"{p}pain_lifting_weight_dur"] = dur_val(details.get("pain_liftingWeight_duration", {}))
    f[f"{p}pain_movement"]           = pain_15(details.get("pain_movement"))
    f[f"{p}pain_movement_dur"]       = dur_val(details.get("pain_movement_duration", {}))

    f[f"{p}morning_stiffness"]          = yn(details.get("morningStiffness"))
    f[f"{p}morning_stiffness_duration"] = int(details.get("stiffnessDuration", 0) or 0)

    reason = normalize(details.get("ailmentReason", ""))
    f[f"{p}reason_natural"]     = 1 if reason == "natural degradation" else 0
    f[f"{p}reason_trauma"]      = 1 if reason == "trauma"               else 0

    trauma_type = normalize(details.get("traumaType", ""))
    f[f"{p}trauma_working_desk"] = 1 if trauma_type == "working desk" else 0
    f[f"{p}trauma_impact"]       = 1 if trauma_type == "impact"       else 0
    f[f"{p}trauma_duration_val"] = dur_val(details.get("traumaDurationCombined", {}))

    related = [normalize(x) for x in details.get("painRelated", [])]
    f[f"{p}related_restricted_movement"] = 1 if "restricted movement" in related else 0
    f[f"{p}related_tingling"]             = 1 if "tingling"             in related else 0
    f[f"{p}related_giddiness"]            = 1 if "giddiness"            in related else 0

    le = details.get("localExam", {})
    f[f"{p}exam_tenderness"] = yn(le.get("tenderness"))
    f[f"{p}exam_slr_left"]   = slr_val(le.get("slrLeft"))
    f[f"{p}exam_slr_right"]  = slr_val(le.get("slrRight"))

    # --- Updated Diagnosis Logic (One-Hot Encoding) ---
    diag_list = [
        "herniated disc", "stenosis", "spondylisthesis", 
        "muscle spasm", "ankylosing spondylitis", 
        "cervical spondylosis", "athritis"
    ]
    
    # Get user selection (handles string or list)
    user_diag = details.get("diagnosis", [])
    if isinstance(user_diag, str):
        user_diag = [user_diag]
    
    selected_diags = [normalize(d) for d in user_diag]

    # Create one column for each possible diagnosis
    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{p}diag_{clean_name}"] = 1 if d_name in selected_diags else 0

    return f



# ── Knee (bilateral) ─────────────────────────────────────────────────────────

def _extract_knee_side(side_data: dict, sp: str) -> dict:
    """sp = side prefix e.g. 'knee_l_'"""
    f = {}

    f[f"{sp}duration_val"]  = dur_val(side_data.get("durationCombined", {}))

    f[f"{sp}pain_walking"]           = pain_15(side_data.get("pain_walking"))
    f[f"{sp}pain_walking_dur"]       = dur_val(side_data.get("pain_walking_duration", {}))
    f[f"{sp}pain_lowsitting"]        = pain_15(side_data.get("pain_lowSitting"))
    f[f"{sp}pain_lowsitting_dur"]    = dur_val(side_data.get("pain_lowSitting_duration", {}))
    f[f"{sp}pain_highsitting"]       = pain_15(side_data.get("pain_highSitting"))
    f[f"{sp}pain_highsitting_dur"]   = dur_val(side_data.get("pain_highSitting_duration", {}))
    f[f"{sp}pain_stairclimbing"]     = pain_15(side_data.get("pain_stairClimbing"))
    f[f"{sp}pain_stairclimbing_dur"] = dur_val(side_data.get("pain_stairClimbing_duration", {}))
    f[f"{sp}pain_standing"]          = pain_15(side_data.get("pain_standing"))
    f[f"{sp}pain_standing_dur"]      = dur_val(side_data.get("pain_standing_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_fall"]        = 1 if trauma_type == "fall"        else 0
    f[f"{sp}trauma_twist"]       = 1 if trauma_type == "twist"       else 0
    f[f"{sp}trauma_slip_impact"] = 1 if trauma_type == "slip impact" else 0
    f[f"{sp}trauma_duration_val"]= dur_val(side_data.get("traumaDurationCombined", {}))

    le = side_data.get("localExam", {})
    tenderness_list = le.get("tenderness", []) or []
    f[f"{sp}exam_tenderness_shin"]  = 1 if any("shin"  in x.lower() for x in tenderness_list) else 0
    f[f"{sp}exam_tenderness_knee"]  = 1 if any("knee"  in x.lower() for x in tenderness_list) else 0

    swelling_list = le.get("swelling", []) or []
    f[f"{sp}exam_swelling_pitting"]     = 1 if any("pitting"     in x.lower() and "non" not in x.lower() for x in swelling_list) else 0
    f[f"{sp}exam_swelling_non_pitting"] = 1 if any("non-pitting" in x.lower() for x in swelling_list) else 0

    f[f"{sp}exam_creps_right"]       = creps_encode(le.get("crepsRight"))
    f[f"{sp}exam_creps_left"]        = creps_encode(le.get("crepsLeft"))
    f[f"{sp}exam_rom"]               = rom_encode(le.get("rom"))
    f[f"{sp}exam_rom_right_restrict"]= restriction_encode(le.get("romRightRestricted"))
    f[f"{sp}exam_rom_left_restrict"] = restriction_encode(le.get("romLeftRestricted"))
    f[f"{sp}exam_rom_right_pain"]    = pain_encode(le.get("romRightPain"))
    f[f"{sp}exam_rom_left_pain"]     = pain_encode(le.get("romLeftPain"))
    f[f"{sp}exam_temperature"]       = temp_encode(le.get("temperature"))
    f[f"{sp}exam_slr_right"]         = slr_val(le.get("slrRight"))
    f[f"{sp}exam_slr_left"]          = slr_val(le.get("slrLeft"))
    f[f"{sp}exam_slr_right_neg"]     = is_negative_slr(le.get("slrRight", ""))
    f[f"{sp}exam_slr_left_neg"]      = is_negative_slr(le.get("slrLeft", ""))
    f[f"{sp}exam_varicosity"]        = yn(le.get("varicosity"))

    # --- Updated Diagnosis Logic (One-Hot Encoding) ---
    diag_list = [
        "oa knee", "lateral meniscus tear", "mcl sprain", "tendonitis",
        "patellar tendonitis", "acl tear", "medial meniscus tear",
        "pcl tear", "high riding patella", "bakers cyst",
        "ra knee", "itb syndrome", "chondromalacia patella"
    ]
    
    # Get user selection (handles string or list)
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str):
        user_diag = [user_diag]
    
    selected_diags = [normalize(d) for d in user_diag]

    # Create binary columns for every possible knee diagnosis
    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected_diags else 0

    return f


def extract_knee(details: dict, prefix: str = "knee") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        side_data = details.get(side, {})
        f.update(_extract_knee_side(side_data, sp))

    # Ailment-level present flag (1 if any side has data)
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f



# ── Shoulder (bilateral) ──────────────────────────────────────────────────────

def _extract_shoulder_side(side_data: dict, sp: str) -> dict:
    f = {}

    f[f"{sp}duration_val"]             = dur_val(side_data.get("durationCombined", {}))
    f[f"{sp}pain_lifting_weight"]      = lifting_kg(side_data.get("pain_liftingWeight"))
    f[f"{sp}pain_lifting_weight_dur"]  = dur_val(side_data.get("pain_liftingWeight_duration", {}))
    f[f"{sp}pain_dailyroutine"]        = pain_15(side_data.get("pain_dailyroutine"))
    f[f"{sp}pain_dailyroutine_dur"]    = dur_val(side_data.get("pain_dailyroutine_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_impact"]      = 1 if trauma_type == "impact" else 0
    f[f"{sp}trauma_fall"]        = 1 if trauma_type == "fall"   else 0
    f[f"{sp}trauma_duration_val"]= dur_val(side_data.get("traumaDurationCombined", {}))

    related = [normalize(x) for x in side_data.get("painRelated", [])]
    f[f"{sp}related_tingling_left"]  = 1 if "tingling left"  in related else 0
    f[f"{sp}related_tingling_right"] = 1 if "tingling right" in related else 0
    f[f"{sp}related_numbness_left"]  = 1 if "numbness left"  in related else 0
    f[f"{sp}related_numbness_right"] = 1 if "numbness right" in related else 0
    f[f"{sp}related_giddiness"]      = 1 if "giddiness"       in related else 0

    le = side_data.get("localExam", {})
    f[f"{sp}exam_tenderness_left"]    = yn(le.get("tendernessLeft"))
    f[f"{sp}exam_tenderness_right"]   = yn(le.get("tendernessRight"))
    for meas in ["flexionLeft", "flexionRight", "extensionLeft", "extensionRight",
                 "abductionLeft", "abductionRight", "adductionLeft", "adductionRight",
                 "medialRotationLeft", "medialRotationRight",
                 "lateralRotationLeft", "lateralRotationRight"]:
        v = le.get(meas, "")
        try:
            f[f"{sp}exam_{meas.lower()}"] = int(str(v).replace("°", "").strip()) if v else 0
        except ValueError:
            f[f"{sp}exam_{meas.lower()}"] = 0

    # --- Updated Diagnosis Logic (One-Hot) ---
    diag_list = [
        "frozen shoulder", "rotator cuff tear", "adhesive capsulitis",
        "arthritis", "tendinitis", "cervical spondylosis", "bursitis"
    ]
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str): user_diag = [user_diag]
    selected = [normalize(d) for d in user_diag]

    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected else 0

    return f

def extract_shoulder(details: dict, prefix: str = "shoulder") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        f.update(_extract_shoulder_side(details.get(side, {}), sp))
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Wrist (bilateral) ─────────────────────────────────────────────────────────

def _extract_wrist_side(side_data: dict, sp: str) -> dict:
    f = {}

    f[f"{sp}duration_val"]             = dur_val(side_data.get("durationCombined", {}))
    f[f"{sp}pain_writing"]             = pain_15(side_data.get("pain_writing"))
    f[f"{sp}pain_writing_dur"]         = dur_val(side_data.get("pain_writing_duration", {}))
    f[f"{sp}pain_gripping"]            = pain_15(side_data.get("pain_gripping"))
    f[f"{sp}pain_gripping_dur"]        = dur_val(side_data.get("pain_gripping_duration", {}))
    f[f"{sp}pain_lifting_weight"]      = lifting_kg(side_data.get("pain_liftingWeight"))
    f[f"{sp}pain_lifting_weight_dur"]  = dur_val(side_data.get("pain_liftingWeight_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_impact"]      = 1 if trauma_type == "impact" else 0
    f[f"{sp}trauma_fall"]        = 1 if trauma_type == "fall"   else 0
    f[f"{sp}trauma_duration_val"]= dur_val(side_data.get("traumaDurationCombined", {}))

    le = side_data.get("localExam", {})
    swelling_list = le.get("swelling", []) or []
    f[f"{sp}exam_swelling_left"]  = 1 if "left"  in [x.lower() for x in swelling_list] else 0
    f[f"{sp}exam_swelling_right"] = 1 if "right" in [x.lower() for x in swelling_list] else 0
    f[f"{sp}exam_numbness"]       = yn(le.get("numbness"))

    # --- Updated Diagnosis Logic (One-Hot) ---
    diag_list = ["arthritis", "carpel tunnel syndrome", "tendinitis", "traumatic injury"]
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str): user_diag = [user_diag]
    selected = [normalize(d) for d in user_diag]

    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected else 0

    return f

def extract_wrist(details: dict, prefix: str = "wrist") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        f.update(_extract_wrist_side(details.get(side, {}), sp))
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Elbow (bilateral) ─────────────────────────────────────────────────────────

def _extract_elbow_side(side_data: dict, sp: str) -> dict:
    f = {}

    f[f"{sp}duration_val"]             = dur_val(side_data.get("durationCombined", {}))
    f[f"{sp}pain_lifting_weight"]      = lifting_kg(side_data.get("pain_liftingWeight"))
    f[f"{sp}pain_lifting_weight_dur"]  = dur_val(side_data.get("pain_liftingWeight_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_impact"]      = 1 if trauma_type == "impact" else 0
    f[f"{sp}trauma_fall"]        = 1 if trauma_type == "fall"   else 0
    f[f"{sp}trauma_duration_val"]= dur_val(side_data.get("traumaDurationCombined", {}))

    related = [normalize(x) for x in side_data.get("painRelated", [])]
    f[f"{sp}related_numbness"] = 1 if "numbness" in related else 0
    f[f"{sp}related_swelling"] = 1 if "swelling" in related else 0

    le = side_data.get("localExam", {})
    v = le.get("flexion", "")
    try:    f[f"{sp}exam_flexion"]   = int(str(v).replace("°", "").strip()) if v else 0
    except: f[f"{sp}exam_flexion"]   = 0
    v = le.get("extension", "")
    try:    f[f"{sp}exam_extension"] = int(str(v).replace("°", "").strip()) if v else 0
    except: f[f"{sp}exam_extension"] = 0

    # --- Updated Diagnosis Logic (One-Hot) ---
    diag_list = ["tendonitis", "bursitis", "tennis elbow", "arthritis", "traumatic injury"]
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str): user_diag = [user_diag]
    selected = [normalize(d) for d in user_diag]

    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected else 0

    return f

def extract_elbow(details: dict, prefix: str = "elbow") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        f.update(_extract_elbow_side(details.get(side, {}), sp))
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Ankle (bilateral) ─────────────────────────────────────────────────────────

def _extract_ankle_side(side_data: dict, sp: str) -> dict:
    f = {}

    f[f"{sp}duration_val"]               = dur_val(side_data.get("durationCombined", {}))
    f[f"{sp}pain_walking"]               = pain_15(side_data.get("pain_walking"))
    f[f"{sp}pain_walking_dur"]           = dur_val(side_data.get("pain_walking_duration", {}))
    f[f"{sp}pain_stairclimbing"]         = pain_15(side_data.get("pain_stairClimbing"))
    f[f"{sp}pain_stairclimbing_dur"]     = dur_val(side_data.get("pain_stairClimbing_duration", {}))
    f[f"{sp}pain_standing"]              = pain_15(side_data.get("pain_standing"))
    f[f"{sp}pain_standing_dur"]          = dur_val(side_data.get("pain_standing_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_impact"]       = 1 if trauma_type == "impact" else 0
    f[f"{sp}trauma_fall"]         = 1 if trauma_type == "fall"   else 0
    f[f"{sp}trauma_twist"]        = 1 if trauma_type == "twist"  else 0
    f[f"{sp}trauma_duration_val"] = dur_val(side_data.get("traumaDurationCombined", {}))

    le = side_data.get("localExam", {})
    tenderness_list = [x.lower() for x in (le.get("tenderness", []) or [])]
    swelling_list   = [x.lower() for x in (le.get("swelling",   []) or [])]
    f[f"{sp}exam_tenderness_left"]  = 1 if "left"  in tenderness_list else 0
    f[f"{sp}exam_tenderness_right"] = 1 if "right" in tenderness_list else 0
    f[f"{sp}exam_swelling_left"]    = 1 if "left"  in swelling_list   else 0
    f[f"{sp}exam_swelling_right"]   = 1 if "right" in swelling_list   else 0

    v = le.get("extension", "")
    try:    f[f"{sp}exam_extension"] = int(str(v).replace("°","").strip()) if v else 0
    except: f[f"{sp}exam_extension"] = 0
    v = le.get("flexion", "")
    try:    f[f"{sp}exam_flexion"]   = int(str(v).replace("°","").strip()) if v else 0
    except: f[f"{sp}exam_flexion"]   = 0

    rotation = normalize(le.get("rotation", ""))
    f[f"{sp}exam_rotation_clockwise"]     = 1 if rotation == "clockwise"     else 0
    f[f"{sp}exam_rotation_anticlockwise"] = 1 if rotation == "anticlockwise" else 0

    # --- Updated Diagnosis Logic (One-Hot Encoding) ---
    diag_list = [
        "arthritis", "gout", "bursitis", 
        "tendinitis", "traumatic injury", "calcaneal spur"
    ]
    
    # Get user selection (handles string or list)
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str):
        user_diag = [user_diag]
    
    selected_diags = [normalize(d) for d in user_diag]

    # Create one column for every possible ankle diagnosis
    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected_diags else 0

    return f


def extract_ankle(details: dict, prefix: str = "ankle") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        f.update(_extract_ankle_side(details.get(side, {}), sp))
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Hip (bilateral) ───────────────────────────────────────────────────────────

def _extract_hip_side(side_data: dict, sp: str) -> dict:
    f = {}

    f[f"{sp}duration_val"]               = dur_val(side_data.get("durationCombined", {}))
    f[f"{sp}pain_walking"]               = pain_15(side_data.get("pain_walking"))
    f[f"{sp}pain_walking_dur"]           = dur_val(side_data.get("pain_walking_duration", {}))
    f[f"{sp}pain_lowsitting"]            = pain_15(side_data.get("pain_lowSitting"))
    f[f"{sp}pain_lowsitting_dur"]        = dur_val(side_data.get("pain_lowSitting_duration", {}))
    f[f"{sp}pain_highsitting"]           = pain_15(side_data.get("pain_highSitting"))
    f[f"{sp}pain_highsitting_dur"]       = dur_val(side_data.get("pain_highSitting_duration", {}))
    f[f"{sp}pain_stairclimbing"]         = pain_15(side_data.get("pain_stairClimbing"))
    f[f"{sp}pain_stairclimbing_dur"]     = dur_val(side_data.get("pain_stairClimbing_duration", {}))
    f[f"{sp}pain_standing"]              = pain_15(side_data.get("pain_standing"))
    f[f"{sp}pain_standing_dur"]          = dur_val(side_data.get("pain_standing_duration", {}))

    f[f"{sp}morning_stiffness"]          = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"] = int(side_data.get("stiffnessDuration", 0) or 0)

    reason = normalize(side_data.get("ailmentReason", ""))
    f[f"{sp}reason_natural"] = 1 if reason == "natural degradation" else 0
    f[f"{sp}reason_trauma"]  = 1 if reason == "trauma"               else 0

    trauma_type = normalize(side_data.get("traumaType", ""))
    f[f"{sp}trauma_impact"]      = 1 if trauma_type == "impact" else 0
    f[f"{sp}trauma_fall"]        = 1 if trauma_type == "fall"   else 0
    f[f"{sp}trauma_duration_val"]= dur_val(side_data.get("traumaDurationCombined", {}))

    le = side_data.get("localExam", {})
    f[f"{sp}exam_tenderness"]          = yn(le.get("tenderness"))
    f[f"{sp}exam_swelling"]            = yn(le.get("swelling"))
    f[f"{sp}exam_slr_left"]            = slr_val(le.get("slrLeft"))
    f[f"{sp}exam_slr_right"]           = slr_val(le.get("slrRight"))

    def _rom_deg(v):
        try: return int(str(v).replace("°","").strip()) if v else 0
        except: return 0

    f[f"{sp}exam_rom_left_restricted"]  = _rom_deg(le.get("romLeftRestricted"))
    f[f"{sp}exam_rom_right_restricted"] = _rom_deg(le.get("romRightRestricted"))
    f[f"{sp}exam_rom_left_painful"]     = _rom_deg(le.get("romLeftPainful"))
    f[f"{sp}exam_rom_right_painful"]    = _rom_deg(le.get("romRightPainful"))

    # --- Updated Diagnosis Logic (One-Hot Encoding) ---
    diag_list = ["oa hip", "avn", "labral tear", "tendinitis"]
    
    # Get user selection (handles string or list)
    user_diag = side_data.get("diagnosis", [])
    if isinstance(user_diag, str):
        user_diag = [user_diag]
    
    selected_diags = [normalize(d) for d in user_diag]

    # Create binary columns for each hip diagnosis
    for d_name in diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}diag_{clean_name}"] = 1 if d_name in selected_diags else 0

    return f


def extract_hip(details: dict, prefix: str = "hip") -> dict:
    f = {}
    for side in ("left", "right"):
        sp = _side_prefix(prefix, side)
        # We pass the data for the specific side
        f.update(_extract_hip_side(details.get(side, {}), sp))
    
    # Flag to indicate if hip data exists at all
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Small Joints (bilateral, split hand/feet per side) ───────────────────────

HAND_FINGERS = ["index finger", "middle finger", "ring finger", "thumb", "little finger"]
FEET_TOES    = ["index toe", "middle toe", "ring toe", "big toe", "little toe"]


def _extract_small_joints_side(side_data: dict, sp: str, hand_key: str, feet_key: str) -> dict:
    f = {}

    f[f"{sp}duration_val"] = dur_val(side_data.get("durationCombined", {}))

    # Affected joints — one binary flag per finger/toe
    hand_joints = [normalize(x) for x in side_data.get(hand_key, [])]
    for finger in HAND_FINGERS:
        safe = finger.replace(" ", "_")
        f[f"{sp}hand_{safe}"] = 1 if finger in hand_joints else 0

    feet_joints = [normalize(x) for x in side_data.get(feet_key, [])]
    for toe in FEET_TOES:
        safe = toe.replace(" ", "_")
        f[f"{sp}feet_{safe}"] = 1 if toe in feet_joints else 0

    # Pain activities
    f[f"{sp}pain_gripping_hands"]          = pain_15(side_data.get("pain_grippingHands"))
    f[f"{sp}pain_gripping_hands_dur"]      = dur_val(side_data.get("pain_grippingHands_duration", {}))
    f[f"{sp}pain_lifting_hands"]           = pain_15(side_data.get("pain_liftingHands"))
    f[f"{sp}pain_lifting_hands_dur"]       = dur_val(side_data.get("pain_liftingHands_duration", {}))
    f[f"{sp}pain_stiffness_hands"]         = pain_15(side_data.get("pain_stiffnessHands"))
    f[f"{sp}pain_stiffness_hands_dur"]     = dur_val(side_data.get("pain_stiffnessHands_duration", {}))
    f[f"{sp}pain_standing_feet"]           = pain_15(side_data.get("pain_standingFeet"))
    f[f"{sp}pain_standing_feet_dur"]       = dur_val(side_data.get("pain_standingFeet_duration", {}))
    f[f"{sp}pain_walking_feet"]            = pain_15(side_data.get("pain_walkingFeet"))
    f[f"{sp}pain_walking_feet_dur"]        = dur_val(side_data.get("pain_walkingFeet_duration", {}))

    f[f"{sp}morning_stiffness"]            = yn(side_data.get("morningStiffness"))
    f[f"{sp}morning_stiffness_duration"]   = int(side_data.get("stiffnessDuration", 0) or 0)

    # Hand reason / trauma
    hand_reason = normalize(side_data.get("handReason", ""))
    f[f"{sp}hand_reason_natural"] = 1 if hand_reason == "natural degradation" else 0
    f[f"{sp}hand_reason_trauma"]  = 1 if hand_reason == "trauma"               else 0
    hand_trauma = normalize(side_data.get("handTraumaType", ""))
    f[f"{sp}hand_trauma_impact"]      = 1 if hand_trauma == "impact" else 0
    f[f"{sp}hand_trauma_fall"]        = 1 if hand_trauma == "fall"   else 0
    f[f"{sp}hand_trauma_duration_val"]= dur_val(side_data.get("handTraumaDuration", {}))

    # Feet reason / trauma
    feet_reason = normalize(side_data.get("feetReason", ""))
    f[f"{sp}feet_reason_natural"] = 1 if feet_reason == "natural degradation" else 0
    f[f"{sp}feet_reason_trauma"]  = 1 if feet_reason == "trauma"               else 0
    feet_trauma = normalize(side_data.get("feetTraumaType", ""))
    f[f"{sp}feet_trauma_impact"]      = 1 if feet_trauma == "impact" else 0
    f[f"{sp}feet_trauma_fall"]        = 1 if feet_trauma == "fall"   else 0
    f[f"{sp}feet_trauma_duration_val"]= dur_val(side_data.get("feetTraumaDuration", {}))

    # Local exam
    le = side_data.get("localExam", {})
    hand_sw_key  = "swellingLeftHand"   if "left" in sp  else "swellingRightHand"
    hand_te_key  = "tendernessLeftHand" if "left" in sp  else "tendernessRightHand"
    feet_sw_key  = "swellingLeftFeet"   if "left" in sp  else "swellingRightFeet"
    feet_te_key  = "tendernessLeftFeet" if "left" in sp  else "tendernessRightFeet"
    f[f"{sp}exam_swelling_hand"]   = yn(le.get(hand_sw_key))
    f[f"{sp}exam_tenderness_hand"] = yn(le.get(hand_te_key))
    f[f"{sp}exam_swelling_feet"]   = yn(le.get(feet_sw_key))
    f[f"{sp}exam_tenderness_feet"] = yn(le.get(feet_te_key))

    # --- Updated Diagnosis Logic (One-Hot for Hands) ---
    hand_diag_list = [
        "arthritis", "carpel tunnel syndrome", "trigger finger",
        "nerve compression", "traumatic injury", "nerve injury"
    ]
    user_hand_diag = side_data.get("diagnosis", [])
    if isinstance(user_hand_diag, str): user_hand_diag = [user_hand_diag]
    selected_hand = [normalize(d) for d in user_hand_diag]

    for d_name in hand_diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}hand_diag_{clean_name}"] = 1 if d_name in selected_hand else 0

    # --- Updated Diagnosis Logic (One-Hot for Feet) ---
    feet_diag_list = ["arthritis", "gout", "nerve injury", "traumatic injury"]
    user_feet_diag = side_data.get("feetDiagnosis", [])
    if isinstance(user_feet_diag, str): user_feet_diag = [user_feet_diag]
    selected_feet = [normalize(d) for d in user_feet_diag]

    for d_name in feet_diag_list:
        clean_name = d_name.replace(" ", "_")
        f[f"{sp}feet_diag_{clean_name}"] = 1 if d_name in selected_feet else 0

    return f


def extract_small_joints(details: dict, prefix: str = "smalljoints") -> dict:
    f = {}
    sp_l = _side_prefix(prefix, "left")
    sp_r = _side_prefix(prefix, "right")
    f.update(_extract_small_joints_side(details.get("left", {}),  sp_l, "joints_leftHand",  "joints_leftFeet"))
    f.update(_extract_small_joints_side(details.get("right", {}), sp_r, "joints_rightHand", "joints_rightFeet"))
    f[f"{prefix}_present"] = 1 if (details.get("left") or details.get("right")) else 0
    return f


# ── Ailment dispatcher ───────────────────────────────────────────────────────

AILMENT_EXTRACTORS = {
    "lumbar":      extract_lumbar,
    "cervical":    extract_cervical,
    "knee":        extract_knee,
    "shoulder":    extract_shoulder,
    "wrist":       extract_wrist,
    "elbow":       extract_elbow,
    "ankle":       extract_ankle,
    "hip":         extract_hip,
    "smallJoints": extract_small_joints,
}

ALL_AILMENT_TYPES = list(AILMENT_EXTRACTORS.keys())


def transform_ailments(raw: dict) -> dict:
    """
    Walk ailments_json array and dispatch to the right extractor.
    Also sets  <ailmentType>_present = 0  for every ailment NOT in the record.
    """
    f = {}

    # Mark all ailments absent by default (will be overwritten if present)
    for atype in ALL_AILMENT_TYPES:
        safe = atype.lower()
        f[f"{safe}_present"] = 0

    ailments = raw.get("ailments_json", [])
    for entry in ailments:
        atype   = entry.get("ailmentType", "")
        details = entry.get("details", {})
        extractor = AILMENT_EXTRACTORS.get(atype)
        if extractor:
            ailment_features = extractor(details, prefix=atype.lower())
            f.update(ailment_features)
            f[f"{atype.lower()}_present"] = 1  # ensure flag is set

    return f


# ── Master transform ─────────────────────────────────────────────────────────

def transform_full(raw: dict) -> dict:
    f1 = transform_patient_systemic(raw)
    f2 = transform_medical_history_features(raw)
    f3 = transform_ailments(raw)
    return {**f1, **f2, **f3}

def process_patients(patients: list) -> pd.DataFrame:
    rows = []

    for patient in patients:
        result = transform_full(patient)   # 👈 uses your function
        # rows.append(result)
        rows.append({
            "patient_name": patient.get("patient_name", "Unknown"),
            "patient_index": len(rows),  # optional but useful
            **result
        })

    df = pd.DataFrame(rows)
    df = df.fillna(0)

    return df

# ── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
  # 🔹 Load patients from external file
    with open("patients.json", "r") as f:
        patients = json.load(f)

    # # 🔹 Process all patients
    df = process_patients(patients)

    # 🔹 View output
    print(df.head())     # shows table
    print(df.shape)      # (num_patients, num_features)

    # # 🔹 Save for ML
    df.to_csv("flattened_patients.csv", index=False)
    
    import json, pprint

    sample = {
        "patient_name": "Banwarilal Sharma",
        "age": 68, "weight": 83.05, "sex": "Male",
        "acidity": "Yes", "gas": "Yes", "stool": "Unsatisfactory",
        "urine": "Nocturnal", "appetite": "Normal",
        "tongue": "Saamha", "nidra": "Disturbed",
        "comorbidities_json": {
            "selected": ["Diabetes Type 2", "Hypertension", "Any Heart Disease"],
            "durations": {}, "medication": "taking allopathy T/t for prostate & low back pain"
        },
        "surgical_history_json": {
            "selected": ["Angioplasty", "Operation for Hernia"],
            "durations": {
                "Angioplasty":          {"val": "2024", "unit": "Years"},
                "Operation for Hernia": {"val": "2013", "unit": "Years"}
            },
            "additionalInfo": "Angioplasty on 23/04/24"
        },
        "ailments_json": [
            {
                "ailmentType": "lumbar",
                "details": {
                    "durationCombined": {"val": "2018", "unit": "Years"},
                    "pain_walking": "Less than 15 mins",
                    "pain_walking_duration": {"val": "300", "unit": "Days"},
                    "pain_lowSitting": "More than 15 mins",
                    "pain_standing": "More than 15 mins",
                    "morningStiffness": "Yes",
                    "ailmentReason": "Natural Degradation",
                    "painRelated": ["Tingling"],
                    "localExam": {"tenderness": "Yes", "swelling": "Yes", "creps": "Grade 1"},
                    "diagnosis": "Lumbar spondylosis"
                }
            },
            {
                "ailmentType": "knee",
                "details": {
                    "left": {
                        "localExam": {
                            "tenderness": ["Left Knee"],
                            "swelling":   ["Left Non-Pitting"],
                            "rom": "Free", "slrLeft": "20°"
                        }
                    },
                    "right": {
                        "localExam": {
                            "tenderness": ["Right Knee"],
                            "swelling":   ["Right Non-Pitting"],
                            "rom": "Free", "slrRight": "20°"
                        }
                    }
                }
            }
        ]
    }

