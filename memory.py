"""What the agent remembers between rounds, saved as JSON so a restart does not forget."""
import json
from collections import defaultdict
from pathlib import Path

from appraisal import grade, grade_change_reason


class Memory:
    def __init__(self, path: str):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.sent: dict = data.get("sent", {})         # round_id -> offers we sent (with markup and buyer type)
        self.rounds: list = data.get("rounds", [])     # results of cleared rounds, oldest first
        self.reveal: dict | None = data.get("reveal")
        self.spec_failures: dict = data.get("spec_failures", {})  # lot_id -> inferred tight dimensions from SPEC_MISMATCH

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"sent": self.sent, "rounds": self.rounds, "reveal": self.reveal,
                                         "spec_failures": self.spec_failures},
                                        ensure_ascii=False, indent=1),
                             encoding="utf-8")

    def record_sent(self, round_id: str, offers: list[dict], state: dict, tv: dict,
                    rules: dict | None = None, ctx=None) -> None:
        rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
        lots = {l["lot_id"]: l for l in state["lots"]}
        llm_guarded_kg = getattr(ctx, "llm_guarded_kg", {})
        records = []
        for o in offers:
            rfq, lot = rfqs[o["rfq_id"]], lots[o["lot_id"]]
            llm_kg = min(o["kg"], llm_guarded_kg.get((o["rfq_id"], o["lot_id"]), 0))
            rules_kg = o["kg"] - llm_kg
            sample = lot["sample"]
            records.append({**o, "buyer": rfq.get("buyer"), "buyer_type": rfq["buyer_type"],
                            "markup": round(o["price_per_kg"] / tv[o["lot_id"]], 3),
                            "sample": sample, "sample_grade": grade(rules, sample["defect_pct"])[0] if rules else None,
                            "tv_estimate": tv[o["lot_id"]], "spec": rfq["spec"],
                            "llm_kg": llm_kg, "rules_kg": rules_kg,
                            "source": "LLM + rules" if llm_kg and rules_kg else ("LLM" if llm_kg else "rules")})
        self.sent[round_id] = records
        self.save()

    def record_results(self, results: dict | None) -> bool:
        if not results or not results.get("round_id") or any(r["round_id"] == results["round_id"] for r in self.rounds):
            return False
        sent = {(o["rfq_id"], o["lot_id"]): o for o in self.sent.get(results["round_id"], [])}
        for offer in results.get("my_offers", []):
            if offer.get("status") != "SPEC_MISMATCH":
                continue
            saved = sent.get((offer.get("rfq_id"), offer.get("lot_id")))
            if not saved or not saved.get("sample") or not saved.get("spec"):
                continue
            risk = infer_spec_risk(saved["sample"], saved["spec"])
            record = {"round_id": results["round_id"], "rfq_id": offer.get("rfq_id"),
                      "spec": saved["spec"], "sample": saved["sample"], **risk}
            self.spec_failures.setdefault(offer["lot_id"], []).append(record)
        self.rounds.append(results)
        self.save()
        return True

    def record_reveal(self, reveal: dict, rules: dict) -> None:
        """Attach this game's true lot quality to its saved offers; memory is game-scoped."""
        self.reveal = reveal
        lots = {lot["lot_id"]: lot for lot in reveal.get("lots", [])}
        for offers in self.sent.values():
            for offer in offers:
                lot = lots.get(offer.get("lot_id"))
                if lot:
                    offer["true_quality"] = lot.get("true")
                    offer["true_grade"] = lot.get("true_grade")
                    offer["true_value_per_kg"] = lot.get("true_value_per_kg")
                    sample = offer.get("sample", {})
                    true = lot.get("true", {})
                    if "defect_pct" in sample and "defect_pct" in true:
                        offer["grade_change_reason"] = grade_change_reason(
                            rules, sample["defect_pct"], true["defect_pct"])
        self.save()

    def summary(self, last: int = 3) -> str:
        """A few short lines for the LLM prompt."""
        # TODO: this only looks at the last few rounds. Trends over the whole game, per buyer persona,
        #       may tell you more (ผู้ซื้อแต่ละรายมีนิสัยต่างกัน)
        lines = []
        for rnd in self.rounds[-last:]:
            sent = {(o["rfq_id"], o["lot_id"]): o for o in self.sent.get(rnd["round_id"], [])}
            mine = []
            for o in rnd["my_offers"]:
                s = sent.get((o["rfq_id"], o["lot_id"]), {})
                detail = f"{s.get('buyer_type', '?')} x{s.get('markup', '?')} {o['status']}"
                if o.get("status") == "SPEC_MISMATCH":
                    failures = self.spec_failures.get(o.get("lot_id"), [])
                    failure = next((x for x in failures if x.get("round_id") == rnd.get("round_id")), None)
                    if failure:
                        axes = ",".join(failure.get("likely_axes", [])) or "unclear"
                        detail += f" (likely spec risk: {axes})"
                mine.append(detail)
            by_type = defaultdict(lambda: [0, 0])
            for row in rnd["market_summary"]:
                by_type[row["buyer_type"]][0] += row["wanted_kg"]
                by_type[row["buyer_type"]][1] += row["bought_kg"]
            market = ", ".join(f"{t} bought {b * 100 // max(w, 1)}% of wanted" for t, (w, b) in by_type.items())
            lines.append(f"{rnd['round_id']}: my offers [{'; '.join(mine) or 'none'}]; market: {market}")
        return "\n".join(lines) or "none yet"


def infer_spec_risk(sample: dict, spec: dict) -> dict:
    """Infer which sample dimension was closest to its RFQ boundary when a mismatch occurred."""
    margins = {
        "defect": (spec["max_defect_pct"] - sample["defect_pct"], 0.75),
        "moisture": (spec["max_moisture_pct"] - sample["moisture_pct"], 0.5),
    }
    if spec.get("min_screen") is not None:
        margins["screen"] = (sample["screen"] - spec["min_screen"], 1.0)
    normalized = {axis: max(0.0, margin) / tolerance for axis, (margin, tolerance) in margins.items()}
    likely = [axis for axis, value in normalized.items() if value <= 1.0]
    if likely:
        return {"likely_axes": likely, "confidence": "likely", "sample_margins": {k: round(v[0], 3) for k, v in margins.items()}}
    nearest = min(normalized, key=normalized.get)
    return {"likely_axes": [nearest], "confidence": "uncertain", "sample_margins": {k: round(v[0], 3) for k, v in margins.items()}}
