"""Coffee market starter agent: poll the market, appraise once, send offers every round.

    python main.py              # play until the market closes
    python main.py --forever    # sandbox: keep playing game after game
"""
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import strategy
from appraisal import appraisal_items, estimate_lots, grade
from llm import LLM
from market import Market, MarketError
from memory import Memory

HERE = Path(__file__).parent
LOGS = HERE / "logs"


def load_env(path: Path) -> None:
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def log(event: str, **data) -> None:
    LOGS.mkdir(exist_ok=True)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "event": event, **data}
    with open(LOGS / "decisions.jsonl", "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")
    short = {k: v for k, v in data.items() if k not in ("offers", "notes", "llm_response", "llm_response_raw")}
    print(row["time"][11:], event, json.dumps(short, ensure_ascii=False)[:160], flush=True)


def append_report(text: str) -> None:
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / "round-reports.md", "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n---\n\n")


def write_offer_report(state: dict, offers: list[dict], source: str, ctx, rules: dict) -> None:
    rnd = state["round"]
    rfqs = {r["rfq_id"]: r for r in state["rfqs"]}
    lots = {l["lot_id"]: l for l in state["lots"]}
    lines = [f"## รอบ {rnd['round_id']} · {source}", "", "### LLM", ""]
    if ctx.llm_response is None:
        lines.append("ไม่มีคำตอบจาก LLM (ใช้ rules หรือ rules เติมแทน)")
        if ctx.llm_response_raw:
            lines.extend(["", "ข้อความดิบที่ parse/ตรวจไม่ผ่าน:", "```text", ctx.llm_response_raw, "```"])
    else:
        lines.extend(["เหตุผล: " + str(ctx.llm_response.get("reason", "ไม่มี")), "", "คำตอบ JSON:", "```json",
                      json.dumps(ctx.llm_response, ensure_ascii=False, indent=2), "```"])
        if ctx.llm_response_raw:
            lines.extend(["", "ข้อความดิบจาก gateway:", "```json", ctx.llm_response_raw, "```"])
    lines.extend(["", "### ทำไม rules จึงเติม", ""])
    if ctx.rules_fill_details:
        for fill in ctx.rules_fill_details:
            q = rfqs.get(fill["rfq_id"], {})
            lines.append(f"- {fill['rfq_id']} ({q.get('buyer_type', '?')}) / {fill['lot_id']}: เพิ่มตามแผน rules {fill['added_bags']} กระสอบ เพื่อเติมปริมาณที่ LLM เว้นไว้")
        lines.append("กฎเลือกตาม markup ผู้ซื้อก่อน แล้วใช้ TV สูงก่อน; guard ยังจำกัดตามล็อต, RFQ และ capacity")
    elif "rules" in source:
        lines.append("fallback: LLM ไม่มีแผนที่ใช้ได้ จึงใช้ rule_plan สร้างข้อเสนอ")
    else:
        lines.append("ไม่มีปริมาณที่ต้องเติม")
    if ctx.notes:
        lines.extend(["", "รายละเอียดการตัดสินใจ/guard:"])
        lines.extend([f"- {note}" for note in ctx.notes])
    lines.extend(["", "### ข้อเสนอและคุณภาพล็อต", "",
                  "| ที่มา | RFQ / ผู้ซื้อ | ล็อต | เกรดจากตัวอย่าง | defect % | moisture % | screen | สเปก RFQ (defect/moisture/screen) | กก. | บาท/กก. |",
                  "|---|---|---|---|---:|---:|---:|---|---:|---:|"])
    for offer in offers:
        lot, rfq = lots[offer["lot_id"]], rfqs[offer["rfq_id"]]
        sample, spec = lot["sample"], rfq["spec"]
        llm_kg = min(offer["kg"], ctx.llm_guarded_kg.get((offer["rfq_id"], offer["lot_id"]), 0))
        origin = "LLM + rules" if 0 < llm_kg < offer["kg"] else ("LLM" if llm_kg else "rules")
        screen_limit = "ไม่กำหนด" if spec["min_screen"] is None else f"≥{spec['min_screen']}"
        lines.append(f"| {origin} | {rfq['rfq_id']} / {rfq.get('buyer', '')} ({rfq['buyer_type']}) | {lot['lot_id']} | "
                     f"{grade(rules, sample['defect_pct'])[0]} | "
                     f"{sample['defect_pct']} | {sample['moisture_pct']} | {sample['screen']} | "
                     f"≤{spec['max_defect_pct']} / ≤{spec['max_moisture_pct']} / {screen_limit} | {offer['kg']} | {offer['price_per_kg']} |")
    append_report("\n".join(lines))


def write_results_report(results: dict, memory: Memory) -> None:
    sent = {(o["rfq_id"], o["lot_id"]): o for o in memory.sent.get(results["round_id"], [])}
    lines = [f"## ผลตลาดรอบ {results['round_id']}", "",
             "| RFQ / ผู้ซื้อ | ล็อต | ผล | ขายได้/ยื่น (กก.) | เกรดตัวอย่าง | defect % | moisture % | สเปก RFQ | สาเหตุเสี่ยงที่คาด |",
             "|---|---|---|---:|---|---:|---:|---|---|"]
    for offer in results.get("my_offers", []):
        saved = sent.get((offer.get("rfq_id"), offer.get("lot_id")), {})
        sample, spec = saved.get("sample", {}), saved.get("spec", {})
        rfq_label = f"{offer.get('rfq_id', '?')} / {saved.get('buyer', '?')} ({saved.get('buyer_type', '?')})"
        screen = "screen ไม่กำหนด" if spec.get("min_screen") is None else f"screen ≥{spec['min_screen']}"
        spec_text = (f"defect ≤{spec.get('max_defect_pct', '?')}%, moisture ≤{spec.get('max_moisture_pct', '?')}%, {screen}"
                     if spec else "ไม่พบข้อมูลสเปก")
        failure = next((f for f in memory.spec_failures.get(offer.get("lot_id"), [])
                        if f.get("round_id") == results.get("round_id")), None)
        axis_labels = {"defect": "เมล็ดเสีย", "moisture": "ความชื้น", "screen": "ขนาดเมล็ด"}
        suspected = (", ".join(axis_labels.get(axis, axis) for axis in failure.get("likely_axes", []))
                     + (" (ไม่แน่ชัด)" if failure.get("confidence") != "likely" else "")) if failure else "—"
        lines.append(f"| {rfq_label} | {offer.get('lot_id', '?')} | **{offer.get('status', '?')}** | "
                     f"{offer.get('filled_kg', 0)}/{offer.get('kg', 0)} | {saved.get('sample_grade', '?')} | "
                     f"{sample.get('defect_pct', '?')} | {sample.get('moisture_pct', '?')} | {spec_text} | {suspected} |")
        if offer.get("status") == "SPEC_MISMATCH":
            lines.append(f"\n**ตรวจล็อต {offer.get('lot_id')}**: ค่าตัวอย่างที่ใช้ตอนเสนอคือ defect {sample.get('defect_pct', '?')}%, moisture {sample.get('moisture_pct', '?')}%, screen {sample.get('screen', '?')}; ตารางด้านบนแสดงขีดจำกัด RFQ ที่ล็อตจริงไม่ผ่าน\n")
    lines.extend(["", "### ภาพรวม RFQ", "", "| RFQ | buyer type | ต้องการ (กก.) | ซื้อได้ (กก.) | ราคาเฉลี่ย |", "|---|---|---:|---:|---:|"])
    for row in results.get("market_summary", []):
        lines.append(f"| {row.get('rfq_id', '?')} | {row.get('buyer_type', '?')} | {row.get('wanted_kg', 0)} | {row.get('bought_kg', 0)} | {row.get('price_avg') or '—'} |")
    append_report("\n".join(lines))


def write_reveal_report(reveal: dict, memory: Memory, rules: dict) -> None:
    memory.record_reveal(reveal, rules)
    lots = {lot["lot_id"]: lot for lot in reveal.get("lots", [])}
    lines = ["## Sandbox reveal · sample เทียบคุณภาพจริง", "",
             "| ล็อต | เกรดจาก sample | เกรดจริง | สาเหตุเกรดเปลี่ยน | sample defect/moisture/screen | true defect/moisture/screen | TV จริง | TV ประเมิน |",
             "|---|---|---|---|---|---|---:|---:|"]
    for lot_id, lot in lots.items():
        sample, true = lot.get("sample", {}), lot.get("true", {})
        estimate = lot.get("your_estimate")
        estimate = estimate.get("est_value_per_kg") if isinstance(estimate, dict) else estimate
        sample_grade = grade(rules, sample["defect_pct"])[0]
        grade_reason = grade_change_reason(rules, sample["defect_pct"], true["defect_pct"])
        lines.append(f"| {lot_id} | {sample_grade} | {lot.get('true_grade', '?')} | "
                     f"{grade_reason} | {sample.get('defect_pct', '?')}% / {sample.get('moisture_pct', '?')}% / {sample.get('screen', '?')} | "
                     f"{true.get('defect_pct', '?')}% / {true.get('moisture_pct', '?')}% / {true.get('screen', '?')} | "
                     f"{lot.get('true_value_per_kg', '—')} | {estimate if estimate is not None else '—'} |")
    bad = []
    for rnd in memory.rounds:
        sent = {(o["rfq_id"], o["lot_id"]): o for o in memory.sent.get(rnd.get("round_id"), [])}
        for result in rnd.get("my_offers", []):
            if result.get("status") == "SPEC_MISMATCH":
                saved = sent.get((result.get("rfq_id"), result.get("lot_id")), {})
                lot = lots.get(result.get("lot_id"), {})
                true = lot.get("true", {})
                spec = saved.get("spec", {})
                screen_limit = "ไม่กำหนด" if spec.get("min_screen") is None else f"≥{spec['min_screen']}"
                bad.append(f"- {rnd['round_id']} {result.get('rfq_id')} / {result.get('lot_id')}: sample grade {saved.get('sample_grade', '?')} "
                           f"(defect {saved.get('sample', {}).get('defect_pct', '?')}%, moisture {saved.get('sample', {}).get('moisture_pct', '?')}%, screen {saved.get('sample', {}).get('screen', '?')}); "
                           f"true grade {lot.get('true_grade', '?')} (defect {true.get('defect_pct', '?')}%, moisture {true.get('moisture_pct', '?')}%, screen {true.get('screen', '?')}); "
                           f"RFQ limits defect ≤{spec.get('max_defect_pct', '?')}%, moisture ≤{spec.get('max_moisture_pct', '?')}%, screen {screen_limit}")
    lines.extend(["", "### รายการ SPEC_MISMATCH เทียบกับคุณภาพจริง", ""])
    lines.extend(bad or ["ไม่มีข้อเสนอที่จบด้วย SPEC_MISMATCH ในเกมนี้"])
    lines.extend(["", "### เทียบคุณภาพจริงกับทุกข้อเสนอ แยกตามรอบ", "",
                  "| รอบ | RFQ / ผู้ซื้อ | ล็อต | ผลตลาด | เกรด sample → จริง | เหตุผลเกรดเปลี่ยน | defect sample → จริง | moisture sample → จริง | screen sample → จริง | สเปกผ่านด้วยค่าจริง? |",
                  "|---|---|---|---|---|---|---|---|---|---|"])
    for rnd in memory.rounds:
        sent = {(o["rfq_id"], o["lot_id"]): o for o in memory.sent.get(rnd.get("round_id"), [])}
        for result in rnd.get("my_offers", []):
            saved = sent.get((result.get("rfq_id"), result.get("lot_id")), {})
            sample, true, spec = saved.get("sample", {}), saved.get("true_quality", {}), saved.get("spec", {})
            if not true:
                continue
            passes = (true.get("defect_pct", float("inf")) <= spec.get("max_defect_pct", float("-inf"))
                      and true.get("moisture_pct", float("inf")) <= spec.get("max_moisture_pct", float("-inf"))
                      and (spec.get("min_screen") is None or true.get("screen", float("-inf")) >= spec["min_screen"]))
            quality_status = "ผ่าน" if passes else "ไม่ผ่าน"
            lines.append(f"| {rnd.get('round_id', '?')} | {result.get('rfq_id', '?')} / {saved.get('buyer', '?')} ({saved.get('buyer_type', '?')}) | "
                         f"{result.get('lot_id', '?')} | {result.get('status', '?')} | {saved.get('sample_grade', '?')} → {saved.get('true_grade', '?')} | "
                         f"{saved.get('grade_change_reason', 'ไม่มีข้อมูล')} | "
                         f"{sample.get('defect_pct', '?')} → {true.get('defect_pct', '?')}% | "
                         f"{sample.get('moisture_pct', '?')} → {true.get('moisture_pct', '?')}% | "
                         f"{sample.get('screen', '?')} → {true.get('screen', '?')} | {quality_status} |")
    append_report("\n".join(lines))


def send_offers(api: Market, offers: list[dict]) -> list[dict]:
    """Send all offers at once; if the batch is rejected, send them one by one and skip the bad ones."""
    try:
        api.offer_batch(offers)
        return offers
    except MarketError as e:
        log("batch_rejected", error=str(e))
    sent = []
    for o in offers:
        try:
            api.offer(o)
            sent.append(o)
        except MarketError as e:
            log("offer_rejected", offer=o, error=str(e))
    return sent


def play(api: Market, llm: LLM | None, forever: bool) -> None:
    info = api.calendar()
    print(info["calendar"])
    rules = info["rules"]
    print("Rules:", json.dumps(rules, ensure_ascii=False, indent=2))
    memory, tv, acted, game = None, {}, set(), None
    rng = random.Random()
    while True:
        try:
            state = api.state()
        except Exception as e:                      # noqa: BLE001 — never die on a network error
            log("state_error", error=str(e))
            time.sleep(3)
            continue
        # the sandbox reuses lot ids every game, so the samples are what tell one game from the next
        lots_key = tuple(sorted((l["lot_id"], json.dumps(l["sample"], sort_keys=True)) for l in state["lots"]))
        if lots_key != game:                        # new game (or sandbox reset): start fresh
            game, acted, tv = lots_key, set(), estimate_lots(rules, state["lots"])
            tag = hashlib.sha1(repr(lots_key).encode()).hexdigest()[:8]
            memory = Memory(LOGS / f"memory-{state['market']}-{tag}.json")
            log("new_game", market=state["market"], lots=len(state["lots"]), tv=tv)
        phase = state["phase"]

        if phase == "APPRAISAL" and not state["appraisal_submitted"]:
            try:
                api.appraise(appraisal_items(rules, state["lots"], tv))
                log("appraisal_sent", tv=tv)
            except MarketError as e:
                log("appraisal_error", error=str(e))

        elif phase == "ROUND_OPEN" and not state["paused"] and state["round"]["round_id"] not in acted:
            rid = state["round"]["round_id"]
            acted.add(rid)
            if memory.record_results(state["last_results"]):
                write_results_report(state["last_results"], memory)
            if any(o["round_id"] == rid for o in state["my_open_offers"]):
                # restarted after sending this round: the market already holds our offers, and a new batch
                # would replace them (capacity_left_kg no longer counts them, so it would offer less)
                log("round_already_sent", round=rid, n=len(state["my_open_offers"]))
                continue
            ctx = strategy.Context(rules, info["calendar"], tv, rng, memory, llm, state["seconds_left"] or 999)
            t0 = time.time()
            try:
                offers, source = strategy.decide(state, ctx)
            except Exception as e:                  # noqa: BLE001 — a bug in strategy must not stop the agent
                log("strategy_crashed", error=repr(e))
                try:
                    fallback = strategy.rule_plan(state, ctx, strategy.candidates(state, ctx))
                    offers, source = strategy.guard(state, ctx, fallback), "rules"
                except Exception as fallback_error:  # noqa: BLE001 — skip this round if fallback also fails
                    log("strategy_fallback_crashed", error=repr(fallback_error))
                    offers, source = [], "none"
            sent = send_offers(api, offers) if offers else []
            memory.record_sent(rid, sent, state, tv, rules, ctx)
            write_offer_report(state, sent, source, ctx, rules)
            log("offers_sent", round=rid, source=source, n=len(sent), kg=sum(o["kg"] for o in sent),
                seconds=round(time.time() - t0, 1), offers=offers, notes=ctx.notes,
                llm_response=ctx.llm_response, llm_response_raw=ctx.llm_response_raw)

        elif phase == "CLOSED" and game is not None and "closed" not in acted:
            acted.add("closed")
            if memory.record_results(state["last_results"]):
                write_results_report(state["last_results"], memory)
            log("market_closed", standings=state["standings"])
            if state.get("market") == "sandbox":
                try:
                    reveal = api.reveal()
                    write_reveal_report(reveal, memory, rules)
                    log("sandbox_reveal_saved", lots=len(reveal.get("lots", [])))
                except Exception as e:  # noqa: BLE001 — reveal is diagnostic only
                    log("sandbox_reveal_error", error=str(e))
            if not forever:
                return

        # Read the clock from the server; never hard-code round times.
        left = state["seconds_left"] or 5
        time.sleep(min(5.0, max(0.5, left + 0.2)))


def main() -> None:
    load_env(HERE / ".env")
    required = ["MARKET_URL", "TEAM_KEY"]
    use_llm = os.environ.get("USE_LLM", "1") == "1"
    if use_llm:
        required.extend(["LLM_BASE_URL", "LLM_MODEL"])
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit("Missing required .env setting(s): " + ", ".join(missing))
    api = Market(os.environ["MARKET_URL"], os.environ["TEAM_KEY"])
    llm = None
    if use_llm:
        llm = LLM(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"], os.environ.get("LLM_API_KEY", ""),
                  timeout=float(os.environ.get("LLM_TIMEOUT", "60")), thinking=os.environ.get("LLM_THINKING") == "1")
    play(api, llm, "--forever" in sys.argv)


if __name__ == "__main__":
    main()
