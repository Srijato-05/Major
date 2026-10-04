from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

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

@dataclass
class SortedItemResult:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    area_mm2: float
    estimated_mass_kg: float
    target_bin: str
    commodity_value_eur: float
    is_hazardous: bool
    is_recycled: bool

class SortingDecisionEngine:
    """
    Evaluates segmented e-waste fragments and assigns them to specific target bins
    based on economic value, material density, and regulatory compliance
    (EU WEEE Annex VII selective treatment & India CPCB EPR rules).
    """

    def __init__(self, taxonomy_config: Dict[str, Any]):
        self.taxonomy: Dict[int, MaterialClassInfo] = {}
        for cid_str, info in taxonomy_config.get("classes", {}).items():
            cid = int(cid_str)
            self.taxonomy[cid] = MaterialClassInfo(
                class_id=cid,
                name=info["name"],
                value_eur_per_kg=float(info["value_eur_per_kg"]),
                density_g_cm3=float(info["density_g_cm3"]),
                thickness_mm=float(info["thickness_mm"]),
                priority=info["priority"],
                regulatory_flag=info["regulatory_flag"],
                target_bin=info["target_bin"]
            )

    def evaluate_item(
        self,
        track_id: int,
        class_id: int,
        confidence: float,
        area_mm2: float
    ) -> SortedItemResult:
        """
        Determines target bin, estimates mass from mask area and density,
        and computes economic value or hazard penalty.
        """
        info = self.taxonomy.get(class_id)
        if info is None:
            raise ValueError(f"Unknown class_id: {class_id}")

        # Mass estimation: Area (mm^2) * thickness (mm) -> Volume (mm^3)
        # Volume (cm^3) = Volume (mm^3) / 1000.0
        # Mass (g) = Volume (cm^3) * Density (g/cm^3)
        # Mass (kg) = Mass (g) / 1000.0
        volume_mm3 = area_mm2 * info.thickness_mm
        volume_cm3 = volume_mm3 / 1000.0
        mass_g = volume_cm3 * info.density_g_cm3
        mass_kg = mass_g / 1000.0

        commodity_value = mass_kg * info.value_eur_per_kg
        is_hazardous = "HAZARDOUS" in info.priority or "EXPLOSIVE" in info.regulatory_flag
        is_recycled = not is_hazardous and info.value_eur_per_kg > 0

        return SortedItemResult(
            track_id=track_id,
            class_id=class_id,
            class_name=info.name,
            confidence=confidence,
            area_mm2=area_mm2,
            estimated_mass_kg=mass_kg,
            target_bin=info.target_bin,
            commodity_value_eur=commodity_value,
            is_hazardous=is_hazardous,
            is_recycled=is_recycled
        )
