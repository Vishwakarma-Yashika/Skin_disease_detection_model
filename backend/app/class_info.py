"""
Static reference info for the 7 HAM10000 diagnosis classes the trained model
predicts. This is informational copy only -- NOT a medical source of truth.
Always keep the disclaimer attached to any prediction shown to a user.
"""

DISCLAIMER = (
    "This tool is an experimental research demo, not a medical device. It has not "
    "been clinically validated and must never be used to diagnose, rule out, or "
    "guide treatment of any condition. If you're worried about a skin lesion, "
    "please see a licensed dermatologist."
)

# key -> metadata. Keys match the DIAGNOSIS_CLASSES / dx codes used during training.
CLASS_INFO = {
    "akiec": {
        "full_name": "Actinic Keratosis / Bowen's Disease",
        "category": "malignant",
        "risk_level": "high",
        "description": (
            "A rough, scaly patch caused by sun damage. Actinic keratoses are "
            "considered pre-cancerous and can progress to squamous cell carcinoma "
            "if untreated; Bowen's disease is an early, in-situ form of squamous "
            "cell carcinoma."
        ),
        "typical_action": "Should be evaluated by a dermatologist -- these are usually treated once identified.",
    },
    "bcc": {
        "full_name": "Basal Cell Carcinoma",
        "category": "malignant",
        "risk_level": "high",
        "description": (
            "The most common form of skin cancer. Basal cell carcinomas grow "
            "slowly and rarely spread elsewhere in the body, but can cause "
            "significant local damage if left untreated."
        ),
        "typical_action": "Warrants prompt evaluation and likely removal by a dermatologist.",
    },
    "bkl": {
        "full_name": "Benign Keratosis-like Lesion",
        "category": "benign",
        "risk_level": "low",
        "description": (
            "A group of non-cancerous growths including seborrheic keratoses, "
            "solar lentigines, and lichen-planus-like keratoses. Very common, "
            "especially with age and sun exposure."
        ),
        "typical_action": "Usually no treatment needed; monitor for changes in size, color, or shape.",
    },
    "df": {
        "full_name": "Dermatofibroma",
        "category": "benign",
        "risk_level": "low",
        "description": (
            "A firm, benign nodule made of fibrous tissue, often on the legs. "
            "Thought to sometimes follow minor skin trauma like an insect bite."
        ),
        "typical_action": "Harmless; removal is only cosmetic if ever needed.",
    },
    "mel": {
        "full_name": "Melanoma",
        "category": "malignant",
        "risk_level": "critical",
        "description": (
            "The most dangerous form of skin cancer. Melanoma can spread "
            "(metastasize) to other organs if not caught early, but is highly "
            "treatable when detected at an early stage."
        ),
        "typical_action": (
            "Needs urgent evaluation by a dermatologist -- do not wait, "
            "regardless of what this tool predicts."
        ),
    },
    "nv": {
        "full_name": "Melanocytic Nevus (Mole)",
        "category": "benign",
        "risk_level": "low",
        "description": (
            "A common, ordinary mole made of pigment-producing cells. The vast "
            "majority of moles are harmless and stable over time."
        ),
        "typical_action": "No action needed unless it changes -- use the ABCDE rule (Asymmetry, Border, Color, Diameter, Evolving) to watch for change.",
    },
    "vasc": {
        "full_name": "Vascular Lesion",
        "category": "benign",
        "risk_level": "low",
        "description": (
            "Blood-vessel-related lesions such as angiomas, angiokeratomas, "
            "pyogenic granulomas, or small hemorrhages under the skin. Almost "
            "always benign."
        ),
        "typical_action": "Usually harmless; a doctor can confirm and treat if it bleeds easily or grows quickly.",
    },
}


def get_class_info(dx_key: str) -> dict:
    info = CLASS_INFO.get(dx_key)
    if info is None:
        return {
            "full_name": dx_key,
            "category": "unknown",
            "risk_level": "unknown",
            "description": "No reference information available for this class.",
            "typical_action": "Consult a dermatologist.",
        }
    return info
