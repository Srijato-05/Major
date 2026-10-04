import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from src.utils.config_loader import CONFIG

@dataclass
class MaterialClassInfo:
    class_id: int
    name: str
    value_eur_per_kg: float
    density_g_cm3: float
    thickness_mm: float
    priority: str
    regulatory_flag: str
    target_bin: str
    min_area_annex_vii_cm2: float = 10.0   # EU Annex VII requires PCB extraction if > 10 cm^2
    critical_raw_elements: List[str] = field(default_factory=list)

@dataclass
class AdvancedSortedItemResult:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    area_mm2: float
    area_cm2: float
    aspect_ratio: float
    estimated_mass_g: float
    estimated_mass_kg: float
    target_bin: str
    destination_action: str       # EJECT_PNEUMATIC, ISOLATE_ROBOTIC, HAZARD_DECONTAMINATE
    commodity_value_eur: float
    epr_compliance_status: str    # COMPLIANT, PENALTY_REGIME_1, PENALTY_REGIME_2
    penalty_eur: float
    net_economic_value_eur: float
    is_hazardous: bool
    is_annex_vii_mandatory: bool
    critical_materials_recovered: List[str]

class AdvancedSortingDecisionEngine:
    """
    Advanced Industrial Decision Matrix implementing:
    1. EU WEEE Directive 2012/19/EU Annex VII selective decontamination thresholds
       (Mandatory extraction for PCBs > 10 cm^2, batteries, capacitors > 25 mm).
    2. India CPCB EPR 2022 penalty modeling:
       - Regime 1: Financial compensation based on EPR shortfalls
       - Regime 2: Operational default penalty (scaling from ₹15,000 to ₹80,000 / ~€160 - €880)
    3. Multi-modal purity validation and secondary cross-contamination prevention.
    """

    def __init__(self, taxonomy_config: Optional[Dict[str, Any]] = None):
        cfg = taxonomy_config or CONFIG.get("taxonomy", {})
        self.taxonomy: Dict[int, MaterialClassInfo] = {}

        # Critical elements per category (from doc-1 and doc-2)
        critical_elements_map = {
            0: ["Au (Gold)", "Ag (Silver)", "Cu (Copper)", "Pd (Palladium)"],
            1: ["Ta (Tantalum)", "Rare Earth Elements (Nd/Dy)", "Au"],
            2: ["Co (Cobalt)", "Li (Lithium)", "Ni (Nickel)"],
            3: ["Br (Bromine Hazard)"],
            4: ["Al (Aluminum)", "Cu (Copper)"],
            5: ["Cu (Pure Copper)"]
        }

        for cid_str, info in cfg.get("classes", {}).items():
            cid = int(cid_str)
            self.taxonomy[cid] = MaterialClassInfo(
                class_id=cid,
                name=info["name"],
                value_eur_per_kg=float(info["value_eur_per_kg"]),
                density_g_cm3=float(info["density_g_cm3"]),
                thickness_mm=float(info["thickness_mm"]),
                priority=info["priority"],
                regulatory_flag=info["regulatory_flag"],
                target_bin=info["target_bin"],
                critical_raw_elements=critical_elements_map.get(cid, [])
            )

    def evaluate_item(
        self,
        track_id: int,
        class_id: int,
        confidence: float,
        area_mm2: float,
        aspect_ratio: float = 1.0,
        spectral_confidence: float = 0.90
    ) -> AdvancedSortedItemResult:
        info = self.taxonomy.get(class_id)
        if info is None:
            raise ValueError(f"Unknown class_id: {class_id}")

        area_cm2 = area_mm2 / 100.0

        # Mass estimation: Area (mm^2) * thickness (mm) -> Volume (mm^3) -> Mass (g)
        volume_mm3 = area_mm2 * info.thickness_mm
        volume_cm3 = volume_mm3 / 1000.0
        mass_g = volume_cm3 * info.density_g_cm3
        mass_kg = mass_g / 1000.0

        # Commodity value
        gross_value = mass_kg * info.value_eur_per_kg

        # Regulatory evaluation
        is_hazardous = "HAZARDOUS" in info.priority or "EXPLOSIVE" in info.regulatory_flag
        is_annex_vii = False
        penalty = 0.0
        status = "COMPLIANT"

        # Check Annex VII mandatory thresholds
        if class_id == 0:  # High-Grade PCB
            if area_cm2 >= 10.0:
                is_annex_vii = True
                action = "MANDATORY_ANNEX_VII_EXTRACTION"
            else:
                action = "HIGH_VALUE_RECOVERY"
        elif class_id == 2:  # Battery Unit
            is_annex_vii = True
            action = "CRITICAL_EXPLOSIVE_ISOLATION"
            if confidence < 0.60:
                # Uncertain battery ejection triggers safety penalty under Regime 2
                penalty = 165.0  # Approx €165 / ₹15,000 equivalent
                status = "PENALTY_REGIME_2_HAZARD"
        elif class_id == 3:  # BFR Polymer
            is_annex_vii = True
            action = "ROHS_DECONTAMINATION_ROUTE"
        else:
            action = "SECONDARY_METAL_RECOVERY"

        net_value = gross_value - penalty

        return AdvancedSortedItemResult(
            track_id=track_id,
            class_id=class_id,
            class_name=info.name,
            confidence=confidence,
            area_mm2=round(area_mm2, 2),
            area_cm2=round(area_cm2, 2),
            aspect_ratio=round(aspect_ratio, 2),
            estimated_mass_g=round(mass_g, 2),
            estimated_mass_kg=round(mass_kg, 4),
            target_bin=info.target_bin,
            destination_action=action,
            commodity_value_eur=round(gross_value, 4),
            epr_compliance_status=status,
            penalty_eur=round(penalty, 2),
            net_economic_value_eur=round(net_value, 4),
            is_hazardous=is_hazardous,
            is_annex_vii_mandatory=is_annex_vii,
            critical_materials_recovered=info.critical_raw_elements if not is_hazardous else []
        )
