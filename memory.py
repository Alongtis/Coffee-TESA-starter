"""What the agent remembers between rounds, saved as JSON so a restart does not forget."""
import json
from collections import defaultdict
from pathlib import Path

from appraisal import grade


class Memory:
    def __init__(self, path: str):
        self.path = Path(path)
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.sent: dict = data.get("sent", {})         # round_id -> offers we sent (with markup and buyer type)
        self.rounds: list = data.get("rounds", [])     # results of cleared rounds, oldest first

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"sent": self.sent, "rounds": self.rounds}, ensure_ascii=False, indent=1),
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
        self.rounds.append(results)
        self.save()
        return True

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
                mine.append(f"{s.get('buyer_type', '?')} x{s.get('markup', '?')} {o['status']}")
            by_type = defaultdict(lambda: [0, 0])
            for row in rnd["market_summary"]:
                by_type[row["buyer_type"]][0] += row["wanted_kg"]
                by_type[row["buyer_type"]][1] += row["bought_kg"]
            market = ", ".join(f"{t} bought {b * 100 // max(w, 1)}% of wanted" for t, (w, b) in by_type.items())
            lines.append(f"{rnd['round_id']}: my offers [{'; '.join(mine) or 'none'}]; market: {market}")
        return "\n".join(lines) or "none yet"
