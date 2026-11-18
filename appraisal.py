"""Estimate a lot's True Value (TV) from its Sample, using the formula the market publishes.

The formula and its numbers come from GET /v1/calendar -> "rules", so nothing is hard-coded here.
"""


def grade(rules: dict, defect_pct: float) -> tuple[str, float]:
    for step in rules["defect_steps"]:
        if defect_pct <= step["max_defect_pct"]:
            return step["grade"], step["step"]
    last = rules["defect_steps"][-1]
    return last["grade"], last["step"]


def grade_change_reason(rules: dict, sample_defect_pct: float, true_defect_pct: float) -> str:
    """Explain grade changes by identifying the defect threshold crossed."""
    steps = rules["defect_steps"]

    def index_for(defect_pct: float) -> int:
        return next((i for i, step in enumerate(steps) if defect_pct <= step["max_defect_pct"]), len(steps) - 1)

    sample_i, true_i = index_for(sample_defect_pct), index_for(true_defect_pct)
    sample_grade, true_grade = steps[sample_i]["grade"], steps[true_i]["grade"]
    if sample_i == true_i:
        return f"เกรดไม่เปลี่ยน ({sample_grade}); defect จริงยังอยู่ในช่วงเกณฑ์เดียวกับ sample"
    low, high = sorted((sample_i, true_i))
    thresholds = [steps[i]["max_defect_pct"] for i in range(low, high)]
    crossed = ", ".join(f"{value:g}%" for value in thresholds)
    direction = "เกรดลดลง" if true_i > sample_i else "เกรดดีขึ้น"
    return (f"{direction}จาก {sample_grade} เป็น {true_grade}: defect เปลี่ยนจาก "
            f"{sample_defect_pct:g}% เป็น {true_defect_pct:g}% และข้ามเส้นเกณฑ์ {crossed}")


def moisture_factor(rules: dict, m: float) -> float:
    mo = rules["moisture"]
    if m <= mo["ref"]:
        return 1.0
    if m <= mo["hard_limit"]:
        return (1 - m / 100) / (1 - mo["ref"] / 100) - mo["penalty_per_pp"] * (m - mo["ref"])
    return mo["hard_factor"]


def screen_factor(rules: dict, screen: int) -> float:
    factors = {int(k): v for k, v in rules["screen_factors"].items()}
    codes = sorted(factors)
    return factors[min(max(int(screen), codes[0]), codes[-1])]


def true_value(rules: dict, sample: dict) -> float:
    """TV = P0 × G(d) × M(m) × S(s), baht per kg."""
    d, m = sample["defect_pct"], sample["moisture_pct"]
    g = (1 - d / 100) * grade(rules, d)[1]
    return rules["p0"] * g * moisture_factor(rules, m) * screen_factor(rules, sample["screen"])


def estimate_lots(rules: dict, lots: list[dict]) -> dict[str, float]:
    # TODO: the sample is noisy (350 g only). A lot whose sample sits near a grade step
    #       may really be one grade better or worse — คิดเผื่อ noise ตรงนี้จะได้คะแนน Appraisal ดีขึ้น
    return {lot["lot_id"]: round(true_value(rules, lot["sample"]), 2) for lot in lots}


def appraisal_items(rules: dict, lots: list[dict], tv: dict[str, float]) -> list[dict]:
    return [{"lot_id": lot["lot_id"], "grade": grade(rules, lot["sample"]["defect_pct"])[0],
             "est_value_per_kg": tv[lot["lot_id"]]} for lot in lots]
