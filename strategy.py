"""Decide this round's offers.  <- ไฟล์หลักที่ทีมควรแก้

Every round:
  1. candidates()  every (lot, RFQ) pair whose sample meets the RFQ spec
  2. llm_plan()    the LLM picks offers from those candidates: which, how many bags, what markup
  3. rule_plan()   used instead when the LLM is off, too slow, or answers nonsense
  4. guard()       every offer, from the LLM or the rules, is checked before it is sent

price_per_kg = markup × our TV estimate. A markup of 1.00 asks exactly what we think the lot is worth.
"""
import json
import math
import random
from dataclasses import dataclass, field

BAG = 60
DEFAULT_PACE = 1.3
# Start with the buyer hierarchy from the workshop; then adjust from actual outcomes.
BASE_MARKUP = {"SPECIALTY": 1.05, "COMMERCIAL": 1.00, "INSTANT": 0.95}
MIN_MARKUP, MAX_MARKUP = 0.85, 1.30   # strategy limits, independent of market formula


def bag_kg(ctx: "Context") -> int:
    return int(ctx.rules.get("bag_kg", BAG))


def calendar_for_round(state: dict, ctx: "Context") -> dict:
    """Return the full calendar entry that covers this round, with safe defaults."""
    number = state.get("round", {}).get("number")
    for entry in ctx.calendar:
        rounds = entry.get("rounds")
        if isinstance(rounds, int) and number == rounds:
            return entry
        if isinstance(rounds, list) and number in rounds:
            return entry
        if isinstance(rounds, str) and number is not None:
            for part in rounds.split(","):
                try:
                    if "-" in part:
                        start, end = (int(x.strip()) for x in part.split("-", 1))
                        if start <= number <= end:
                            return entry
                    elif int(part.strip()) == number:
                        return entry
                except ValueError:
                    continue
    rnd = state.get("round", {})
    return {"label": rnd.get("calendar", "unknown"), "demand_weight": 1.0, "specialty_share": None}


def current_markups(memory) -> dict[str, float]:
    """Nudge buyer-type prices using the last three rounds' cleared outcomes."""
    markups = dict(BASE_MARKUP)
    if not memory:
        return markups
    for rnd in memory.rounds[-3:]:
        sent_by_pair = {
            (o["rfq_id"], o["lot_id"]): o
            for o in memory.sent.get(rnd.get("round_id"), [])
        }
        deltas = {kind: [] for kind in markups}
        for offer in rnd.get("my_offers", []):
            sent = sent_by_pair.get((offer.get("rfq_id"), offer.get("lot_id")))
            if not sent or sent.get("buyer_type") not in deltas:
                continue
            status = offer.get("status")
            if status in ("ACCEPTED", "PARTIAL"):
                deltas[sent["buyer_type"]].append(0.01)
            elif status == "ABOVE_LIMIT":
                deltas[sent["buyer_type"]].append(-0.03)
            elif status == "OUTBID":
                deltas[sent["buyer_type"]].append(-0.015)
        for kind, adjustments in deltas.items():
            if adjustments:
                # Aggregate within a round so a large number of offers cannot cause a price swing.
                markups[kind] += sum(adjustments) / len(adjustments)
    return {kind: round(min(MAX_MARKUP, max(MIN_MARKUP, value)), 3)
            for kind, value in markups.items()}


@dataclass
class Context:
    rules: dict                       # GET /v1/calendar -> rules
    calendar: list                    # GET /v1/calendar -> calendar
    tv: dict                          # lot_id -> our TV estimate (baht/kg)
    rng: random.Random
    memory: object = None
    llm: object = None
    seconds_left: float = 999         # time left in this round when the state was read
    notes: list = field(default_factory=list)
    llm_response: dict | None = None
    llm_response_raw: str | None = None
    llm_plan_details: list[dict] = field(default_factory=list)
    rules_fill_details: list[dict] = field(default_factory=list)
    llm_guarded_kg: dict[tuple[str, str], int] = field(default_factory=dict)


def spec_ok(sample: dict, spec: dict) -> bool:
    # TODO: the sample is noisy. A lot just inside the limit may really be outside -> SPEC_MISMATCH.
    #       ลองเว้น margin เล็กน้อยสำหรับล็อตที่ค่าใกล้ขอบสเปก
    return (sample["defect_pct"] <= spec["max_defect_pct"] and sample["moisture_pct"] <= spec["max_moisture_pct"]
            and (spec.get("min_screen") is None or sample["screen"] >= spec["min_screen"]))


def rounds_left(state: dict, ctx: Context) -> int:
    rnd, total = state["round"], ctx.rules.get("timeline", {}).get("rounds", 1)
    return total if rnd["kind"] == "practice" else total - rnd["number"] + 1


def sell_budget(state: dict, ctx: Context) -> dict[str, int]:
    """How many kg of each lot to offer this round: spread sales over the rounds that are left."""
    bag = bag_kg(ctx)
    left = rounds_left(state, ctx)
    market_calendar = calendar_for_round(state, ctx)
    raw_demand_weight = market_calendar.get("demand_weight", 1.0)
    demand_weight = max(0.0, float(1.0 if raw_demand_weight is None else raw_demand_weight))
    pace = float(ctx.rules.get("round_capacity_factor", DEFAULT_PACE)) * demand_weight
    out = {}
    for lot in state["lots"]:
        avail = lot["kg_available"] // bag * bag
        share = avail if left <= 1 else math.ceil(avail / left * pace / bag) * bag
        out[lot["lot_id"]] = min(avail, max(bag, share)) if avail >= bag and pace > 0 else 0
    return out


def candidates(state: dict, ctx: Context) -> list[dict]:
    bag = bag_kg(ctx)
    budget = sell_budget(state, ctx)
    out = []
    for lot in state["lots"]:
        if budget[lot["lot_id"]] < bag:
            continue
        for rfq in state["rfqs"]:
            if spec_ok(lot["sample"], rfq["spec"]):
                out.append({"lot_id": lot["lot_id"], "rfq_id": rfq["rfq_id"], "buyer": rfq["buyer"],
                            "buyer_type": rfq["buyer_type"], "rfq_bags": rfq["bags"],
                            "max_bags": min(budget[lot["lot_id"]], rfq["kg"]) // bag,
                            "sample": lot["sample"], "spec": rfq["spec"],
                            "tv": ctx.tv[lot["lot_id"]]})
    # Shuffle (weighted by RFQ size): if 40 teams all chase the first RFQ, they all get OUTBID.
    keyed = [(ctx.rng.random() ** (1 / c["rfq_bags"]), c) for c in out]
    return [c for _, c in sorted(keyed, key=lambda x: -x[0])]


def rule_plan(state: dict, ctx: Context, cands: list[dict]) -> list[dict]:
    """No LLM: prioritize higher markups and TV, using the current adaptive prices."""
    bag = bag_kg(ctx)
    budget = sell_budget(state, ctx)
    rfq_left = {r["rfq_id"]: r["bags"] for r in state["rfqs"]}
    plan = []
    markups = current_markups(ctx.memory)
    for c in sorted(cands, key=lambda c: (-markups[c["buyer_type"]], -c["tv"])):
        bags = min(budget[c["lot_id"]] // bag, rfq_left[c["rfq_id"]])
        if bags > 0:
            plan.append({"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": bags, "markup": markups[c["buyer_type"]]})
            budget[c["lot_id"]] -= bags * bag
            rfq_left[c["rfq_id"]] -= bags
    return plan


SYSTEM = """You are the sales agent of a coffee farm selling green robusta beans in a sealed-offer market.
Each round buyers post RFQs. You choose which RFQs to offer your lots to, how many bags, and a markup.
price = markup x the lot's True Value estimate (tv). Buyers rank offers by price / true value, cheapest first,
and reject offers above their hidden ceiling. Unsold beans have the market-supplied salvage value at the end.
Buyer types pay differently: SPECIALTY the most, then COMMERCIAL, INSTANT the least. Ceilings are hidden;
learn them from the results (ABOVE_LIMIT means the price was too high for that buyer). Samples can differ from true lot quality;
use the sample and RFQ limits to avoid offers close to spec boundaries when possible.
Answer only JSON matching the schema."""

SCHEMA = {"type": "object", "required": ["offers", "reason"], "properties": {
    "offers": {"type": "array", "items": {"type": "object", "required": ["candidate", "bags", "markup"], "properties": {
        "candidate": {"type": "integer"}, "bags": {"type": "integer"}, "markup": {"type": "number"}}}},
    "reason": {"type": "string", "maxLength": 300}}}   # short: a long reason can run past max_tokens and cut the JSON


def llm_plan(state: dict, ctx: Context, cands: list[dict]) -> list[dict] | None:
    rnd = state["round"]
    bag = bag_kg(ctx)
    target = sum(sell_budget(state, ctx).values()) // bag
    if state["capacity_left_kg"] is not None:
        target = min(target, state["capacity_left_kg"] // bag)
    time_for_llm = ctx.seconds_left - 5          # keep 5 s to send the offers before the round closes
    if time_for_llm < 3:
        ctx.notes.append("no time for the LLM this round")
        return None
    table = [{"candidate": i, **{k: c[k] for k in ("lot_id", "buyer", "buyer_type", "rfq_bags", "max_bags", "tv", "sample", "spec")}}
             for i, c in enumerate(cands)]
    market_calendar = calendar_for_round(state, ctx)
    salvage_ratio = ctx.rules.get("salvage_ratio", 0.8)
    user = "\n".join([
        f"Round {rnd['round_id']} ({rnd['kind']}), market calendar: {json.dumps(market_calendar, ensure_ascii=False)}. Rounds left: {rounds_left(state, ctx)}.",
        f"Bag size: {bag} kg. Round capacity factor: {ctx.rules.get('round_capacity_factor', 'provided by market state')}.",
        f"Salvage ratio: {salvage_ratio} x TV for unsold beans.",
        f"You may offer at most {state['capacity_left_kg'] // bag if state['capacity_left_kg'] is not None else 'any'} bags in total this round.",
        f"Plan to offer about {target} bags in total this round, considering salvage ratio {salvage_ratio}.",
        f"Current suggested markups by buyer type: {json.dumps(current_markups(ctx.memory))}",
        "Recent results:", ctx.memory.summary() if ctx.memory else "none",
        "Candidates (lot x RFQ, already spec-checked):", json.dumps(table, ensure_ascii=False),
        "Return offers for the full feasible selling plan, up to the total bag target above. Do not leave a feasible bag unsold just to be conservative.",
        "Use each candidate at most once; bags must be a positive integer <= max_bags. Cover as much of the target as possible while respecting each lot, RFQ, and total round capacity.",
        'Return exactly one JSON object with both keys: {"offers": [...], "reason": "short explanation"}.',
    ])
    # TODO: give the LLM more to reason about — buyer personas' habits, the calendar ahead, price trends
    answer = ctx.llm.ask_json(SYSTEM, user, SCHEMA, timeout=time_for_llm)
    ctx.llm_response = answer
    ctx.llm_response_raw = getattr(ctx.llm, "last_response_text", None)
    if answer is None or not isinstance(answer.get("offers"), list):
        ctx.notes.append(f"llm failed: {ctx.llm.last_error}")
        return None
    plan = []
    for o in answer["offers"]:
        try:
            c = cands[int(o["candidate"])]
            entry = {"lot_id": c["lot_id"], "rfq_id": c["rfq_id"], "bags": int(o["bags"]), "markup": float(o["markup"])}
            plan.append(entry)
            ctx.llm_plan_details.append({**entry, "candidate": int(o["candidate"])})
        except (KeyError, IndexError, TypeError, ValueError):
            ctx.notes.append(f"llm bad offer ignored: {o}")
    ctx.notes.append(f"llm reason: {str(answer.get('reason'))[:200]}")
    return plan


def guard(state: dict, ctx: Context, plan: list[dict]) -> list[dict]:
    """Turn a plan into offers the market will accept. Never trust the LLM's numbers as they are."""
    bag = bag_kg(ctx)
    lots = {l["lot_id"]: l for l in state["lots"]}
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    room = state["capacity_left_kg"] if state["capacity_left_kg"] is not None else 10**9
    lot_left = {i: l["kg_available"] for i, l in lots.items()}
    rfq_left = {i: r["kg"] for i, r in rfqs.items()}
    offers, seen = [], set()
    for p in plan:
        lot, rfq = lots.get(p["lot_id"]), rfqs.get(p["rfq_id"])
        if lot is None or rfq is None or (p["lot_id"], p["rfq_id"]) in seen or not spec_ok(lot["sample"], rfq["spec"]):
            ctx.notes.append(f"guard dropped {p}")
            continue
        kg = min(p["bags"] * bag, lot_left[lot["lot_id"]], rfq_left[rfq["rfq_id"]], room) // bag * bag
        if kg < bag:
            continue
        salvage_floor = min(MAX_MARKUP, max(MIN_MARKUP, float(ctx.rules.get("salvage_ratio", 0.8)) + 0.01))
        markup = min(max(p["markup"], salvage_floor), MAX_MARKUP)
        offers.append({"rfq_id": rfq["rfq_id"], "lot_id": lot["lot_id"], "kg": kg,
                       "price_per_kg": round(markup * ctx.tv[lot["lot_id"]], 2)})
        seen.add((p["lot_id"], p["rfq_id"]))
        lot_left[lot["lot_id"]] -= kg
        rfq_left[rfq["rfq_id"]] -= kg
        room -= kg
    return offers


def merge_rule_fill(plan: list[dict], fill: list[dict]) -> list[dict]:
    """Use the rule plan to complete LLM quantities without duplicating lot/RFQ pairs."""
    merged = [dict(p) for p in plan]
    positions = {(p["lot_id"], p["rfq_id"]): i for i, p in enumerate(merged)}
    for p in fill:
        key = (p["lot_id"], p["rfq_id"])
        if key in positions:
            existing = merged[positions[key]]
            existing["bags"] = max(existing["bags"], p["bags"])
        else:
            positions[key] = len(merged)
            merged.append(dict(p))
    return merged


def decide(state: dict, ctx: Context) -> tuple[list[dict], str]:
    """Returns (offers, source) where source is "llm" or "rules"."""
    cands = candidates(state, ctx)
    if not cands:
        return [], "none"
    plan = llm_plan(state, ctx, cands) if ctx.llm else None
    if plan:
        # Rules complete omitted quantities and pairs; merge duplicate pairs so their
        # fallback quantity is not thrown away by guard().
        llm_offers = guard(state, Context(ctx.rules, ctx.calendar, ctx.tv, ctx.rng), plan)
        ctx.llm_guarded_kg = {(o["rfq_id"], o["lot_id"]): o["kg"] for o in llm_offers}
        fallback = rule_plan(state, ctx, cands)
        llm_bags = {(p["lot_id"], p["rfq_id"]): p["bags"] for p in plan}
        ctx.rules_fill_details = [
            {"lot_id": p["lot_id"], "rfq_id": p["rfq_id"],
             "added_bags": max(0, p["bags"] - llm_bags.get((p["lot_id"], p["rfq_id"]), 0))}
            for p in fallback
            if p["bags"] > llm_bags.get((p["lot_id"], p["rfq_id"]), 0)
        ]
        offers = guard(state, ctx, merge_rule_fill(plan, fallback))
        added_kg = max(0, sum(o["kg"] for o in offers) - sum(o["kg"] for o in llm_offers))
        return offers, "llm" if not added_kg else f"llm+rules (+{added_kg}kg fill)"
    return guard(state, ctx, rule_plan(state, ctx, cands)), "rules"
