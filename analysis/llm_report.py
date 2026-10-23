"""LLM 보고서 에이전트: 검증된 도구로 분기 결과를 조회해 담당자용 요약을 작성하고, 수치·판정을 자동 대조

역할 = 조회·요약·근거 연결·조사 항목 정리. 금지 = 확률 계산, 판정·권고 변경, 승인, 원인 단정.
대조 실패(숫자·분기 불일치, 판정/권고 불일치, 원인 단정 표현, 도구 이름 노출) 시 발행 보류.
규칙 기반 보고서(report.build)와 같은 기준으로 비교: 대조 통과율, 핵심 항목 포함률, 소요 시간, 토큰

실행 (시험): .venv/Scripts/python modelops/analysis/llm_report.py --rule r2.1 --model claude-haiku-5-5
API 키는 환경변수 ANTHROPIC_API_KEY 에서만 읽음
"""
import argparse
import json
import re
import time

import anthropic

import report
from candidates import LABEL
from config import RESULTS
from pipeline import RUNS

PROMPT_VERSION = "lr2"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}

SYSTEM = """당신은 신용평가모형 운영 담당자를 위한 분기 점검 보고서 초안 작성자입니다.
1. 먼저 다섯 도구를 모두 호출해 이번 분기 결과를 조회합니다.
2. 상태와 권고는 도구가 준 값을 그대로 옮깁니다. 바꾸거나 새로 판단하지 않습니다.
3. 숫자는 도구 결과에 있는 값만, 같은 자릿수로 씁니다 (비율은 소수 둘째 자리, 상각률은 % 소수 첫째 자리). 직접 계산한 숫자를 쓰지 않습니다.
4. 차이의 원인을 단정하지 않습니다. 가능한 요인은 '조사 항목'으로만 적습니다.
5. 담당자가 3분 안에 읽도록 문어체, 명사형 또는 '-함' 종결, 짧은 문장으로 씁니다. '-고,' '-며,' 뒤 쉼표, 문장 안 콜론과 화살표, 가운뎃점(·) 나열, '~부터 ~까지', 과장 수식어, 비유를 쓰지 않습니다.
6. 도구 이름, 변수의 영문 코드명은 본문에 쓰지 않습니다 (한국어 명칭 사용).
7. 근거(evidence)는 정확히 3개이며 제목은 '신규 대출 특성과 데이터 품질', '예측과 실제의 차이', '개선안 별도 검증' 순서입니다.
8. 조사 항목은 담당자 확인 항목 2~4개를 명사형으로 씁니다 (예: '결측 급증 변수와 발생 시점 확인'). 도구 결과에 없는 외부 사실(경기, 금리 추세 등)을 사실처럼 쓰지 않습니다."""

TOOLS = [
    {"name": "get_decision", "description": "분기 판정 결과: 상태, 권고, 판정 사유 코드, 판단 기준월",
     "input_schema": {"type": "object", "properties": {"quarter": {"type": "string"}},
                      "required": ["quarter"], "additionalProperties": False}},
    {"name": "get_outcome_checks", "description": "예측 대비 실제: 24개월 결과 확정 코호트(최근 2개)와 12개월 조기 결과 코호트(최근 2개)의 건수·실제·예측·실제/예측·보정 z·AUC, 등급별 실제/예측",
     "input_schema": {"type": "object", "properties": {"quarter": {"type": "string"}},
                      "required": ["quarter"], "additionalProperties": False}},
    {"name": "get_input_checks", "description": "신규 실행 대출 선행 점검: 건수, 점수 분포 변화(PSI), 직전 4분기 대비 변화가 큰 변수(한국어 명칭)",
     "input_schema": {"type": "object", "properties": {"quarter": {"type": "string"}},
                      "required": ["quarter"], "additionalProperties": False}},
    {"name": "get_candidate_results", "description": "개선 후보 비교 결과 (실험한 분기만): 선택 후보, 평가 코호트, 후보별 실제/예측·log loss 개선(%)·AUC·통과 여부",
     "input_schema": {"type": "object", "properties": {"quarter": {"type": "string"}},
                      "required": ["quarter"], "additionalProperties": False}},
    {"name": "get_method_notes", "description": "판정 규칙과 데이터 한계 요약 (보고서 유의 사항 작성용)",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string"}, "recommendation": {"type": "string"},
        "summary": {"type": "string", "description": "3~4문장 요약"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "text": {"type": "string"}},
            "required": ["title", "text"], "additionalProperties": False}},
        "investigate": {"type": "array", "items": {"type": "string"},
                        "description": "담당자 조사 항목 (원인 단정 금지)"},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["status", "recommendation", "summary", "evidence", "investigate", "caveats"],
    "additionalProperties": False,
}

SPOKEN = re.compile(r"(니다|어요|아요|해요|세요|죠|네요|군요|예요|에요)(?=[\s.,!?)]|$)")
CAUSAL = re.compile(r"때문(이다|임|에)|원인은(?! 무엇)(?![^.?]*\?)|로 인해|로 인한|탓")


def _r2(x):
    return None if x is None else round(float(x), 2)


def _p(x):
    return None if x is None else f"{float(x) * 100:.1f}%"


def tool_output(name, dec):
    if name == "get_decision":
        return {"quarter": dec["quarter"], "status": dec["status"],
                "recommendation": dec["recommendation"], "reasons": dec.get("reasons", []),
                "asof_month": dec.get("asof", "")[:7]}
    if name == "get_outcome_checks":
        def o(r, kind):
            return {"kind": kind, "cohort": r["cohort_q"], "n": r["n"], "actual": _p(r["actual"]),
                    "pred": _p(r["pred"]), "actual_over_pred": _r2(r["o_e_ratio"]),
                    "adj_z": round(r["z_adj"], 1), "auc": None if r.get("auc") is None else round(r["auc"], 3),
                    "grade_actual_over_pred": {g["grade"]: _r2(g["o_e_ratio"]) for g in r.get("grade", [])}}
        return {"lag24": [o(r, "24개월 확정") for r in dec.get("lag", [])],
                "early12": [o(r, "12개월 조기") for r in dec.get("early", [])]}
    if name == "get_input_checks":
        l = dec.get("lead") or {}
        if l.get("skipped") or not l.get("schema_ok"):
            return {"note": l.get("skipped") or "스키마 오류로 점검 불가"}
        return {"n": l["n"], "score_psi_vs_train": round(l["score_psi_ref"], 3),
                "score_psi_vs_last4q": round(l["score_psi_recent"], 3),
                "avg_pred": _p(l["score_mean"]), "avg_pred_train": _p(l["score_mean_ref"]),
                "top_changed_inputs": [{"name": report.FEAT_KO.get(t["feature"], t["feature"]),
                                        "psi": round(t["psi"], 2)} for t in l["top_recent"]]}
    if name == "get_candidate_results":
        c = dec.get("candidates")
        if not c:
            return {"note": "조치 검토 기준 미달로 개선 후보 실험 생략"}
        return {"chosen": LABEL.get(c["chosen"]) if c["chosen"] else None,
                "residual_error_flag": c.get("residual"), "validation": "1년 전 시점 데이터로 적합 후 이후 확정 코호트로 평가",
                "eval_cohorts": f"{c['sel_q'][0]}~{c['sel_q'][-1]}",
                "candidates": [{"name": LABEL[r["candidate"]], "actual_over_pred": _r2(r["o_e_ratio"]),
                                "logloss_gain_pct": round(r["rel_ll_gain"] * 100, 1),
                                "auc": round(r["auc"], 3), "passed": bool(r["passed"])}
                               for r in c["table"]]}
    if name == "get_method_notes":
        return {"rules": "실제/예측 10% 이상 차이(1.10 이상 또는 0.90 이하) + 과산포 보정 z 3 이상이 지속되면 조치 검토. 주의는 5%(1.05·0.95), 보정 z 2. 12개월 중간 신호는 18개월에 확인, 24개월 결과는 30개월에 확정",
                "limits": ["공개 대출 데이터 기반 과거 분기 재현이며 실제 금융회사 운영 아님",
                           "상각 시점은 마지막 납부월 기준 근사", "모형 변경은 담당자 승인 후 적용"]}
    raise ValueError(name)


def validate(out, dec, tool_texts):
    probs = []
    if out["status"] != dec["status"]:
        probs.append(f"상태 불일치: {out['status']}")
    if out["recommendation"] != dec["recommendation"]:
        probs.append(f"권고 불일치: {out['recommendation']}")
    text = " ".join([out["summary"]] + [e["title"] + " " + e["text"] for e in out["evidence"]]
                    + out["investigate"] + out["caveats"])
    # 숫자 대조: 정량 근거 JSON + 도구가 반환한 값(반올림 표기 포함)
    ref = {"dec": dec, "tools": [json.loads(t) for t in tool_texts]}
    v = report.verify(text, ref)
    if v["unmatched"]:
        probs.append("근거에 없는 숫자·분기: " + ", ".join(v["unmatched"][:8]))
    # 항목별 대조: 실제·예측·실제/예측 값이 해당 코호트의 같은 항목인지
    bad_f = report.verify_fields(text, dec)
    if bad_f:
        probs.append("항목 불일치: " + ", ".join(bad_f[:8]))
    if SPOKEN.search(text):
        probs.append("구어체 표현: " + SPOKEN.search(text).group(0))
    if any(x.rstrip().endswith("?") for x in out["investigate"]):
        probs.append("조사 항목 질문형 (명사형 종결 필요)")
    if CAUSAL.search(text):
        probs.append("원인 단정 표현: " + CAUSAL.search(text).group(0))
    leak = [t["name"] for t in TOOLS if t["name"] in text] + re.findall(r"\b(int_rate|fico|dti|grade_num)\b", text)
    if leak:
        probs.append("코드명 노출: " + ", ".join(leak))
    if len(out["evidence"]) != 3:
        probs.append(f"근거 개수 {len(out['evidence'])}")
    return probs, v["n_numbers"]


def coverage(text, dec):
    """핵심 항목 포함 여부: 24개월 코호트, 실제/예측, 조기 결과, 선택 후보"""
    items = {}
    if dec.get("lag"):
        l0 = dec["lag"][0]
        items["24개월 코호트"] = l0["cohort_q"] in text
        items["24개월 실제/예측"] = f"{l0['o_e_ratio']:.2f}" in text
    if dec.get("early"):
        items["조기 결과"] = f"{dec['early'][0]['o_e_ratio']:.2f}" in text
    c = dec.get("candidates")
    if c and c["chosen"]:
        items["선택 후보"] = LABEL[c["chosen"]] in text
    return items


def run_one(client, model, dec, effort):
    q = dec["quarter"]
    messages = [{"role": "user", "content": f"{q} 분기 점검 보고서 초안을 작성해 주세요. 먼저 다섯 도구를 모두 호출하세요."}]
    need = {t["name"] for t in TOOLS}
    calls, tool_texts, usage = [], [], {"in": 0, "out": 0}
    for _ in range(6):
        extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                 if model in FALLBACK_MODELS else {})
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            output_config={"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
            **extra)
        usage["in"] += resp.usage.input_tokens
        usage["out"] += resp.usage.output_tokens
        if resp.stop_reason in ("refusal", "max_tokens"):
            return None, calls, tool_texts, usage, resp.stop_reason
        uses = [b for b in resp.content if b.type == "tool_use"]
        if resp.stop_reason == "tool_use" and uses:
            messages.append({"role": "assistant", "content": resp.content})
            res = []
            for b in uses:
                try:
                    if b.input.get("quarter", q) != q:
                        raise ValueError("다른 분기는 조회할 수 없음")
                    s = json.dumps(tool_output(b.name, dec), ensure_ascii=False)
                    tool_texts.append(s); calls.append(b.name)
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": s})
                except Exception as e:
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": f"오류: {e}",
                                "is_error": True})
            messages.append({"role": "user", "content": res})
            continue
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if text is None:
            return None, calls, tool_texts, usage, f"응답 없음({resp.stop_reason})"
        missing = sorted(need - set(calls))
        if missing:
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": "아직 호출하지 않은 도구가 있음: " + ", ".join(missing)})
            continue
        return json.loads(text), calls, tool_texts, usage, None
    return None, calls, tool_texts, usage, "왕복 초과"


def revalidate(path):
    """저장된 응답을 API 재호출 없이 현재 검증 규칙으로 다시 검사"""
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    for r in recs:
        dec = json.loads((RUNS / r["rule_version"] / r["quarter"] / "decision.json").read_text(encoding="utf-8"))
        if r["output"]:
            tt = [json.dumps(tool_output(t["name"], dec), ensure_ascii=False) for t in TOOLS]
            r["problems"], r["n_numbers"] = validate(r["output"], dec, tt)
            r["published"] = not r["problems"]
            r["revalidated"] = True
        print(r["quarter"], r["published"], r["problems"])
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="r2.1")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--quarters", default="")
    ap.add_argument("--revalidate", action="store_true")
    a = ap.parse_args()
    if a.revalidate:
        return revalidate(RESULTS / f"m09_llm_report_{a.model}.jsonl")
    client = anthropic.Anthropic()
    decs = [json.loads((d / "decision.json").read_text(encoding="utf-8"))
            for d in sorted((RUNS / a.rule).iterdir()) if (d / "decision.json").exists()]
    if a.quarters:
        decs = [d for d in decs if d["quarter"] in a.quarters.split(",")]
    out_path = RESULTS / f"m09_llm_report_{a.model}.jsonl"
    rows = []
    for dec in decs:
        if dec.get("run_status") != "성공" or not dec.get("lag"):
            continue
        t0 = time.time()
        rule_txt = report.build(dec)
        rule_chk = report.verify(rule_txt, dec)
        t_rule = time.time() - t0
        t0 = time.time()
        try:
            out, calls, tool_texts, usage, err = run_one(client, a.model, dec, a.effort)
        except anthropic.APIError as e:
            out, calls, tool_texts, usage, err = None, [], [], {"in": 0, "out": 0}, f"API 오류: {type(e).__name__}"
        t_llm = time.time() - t0
        probs, n_num = validate(out, dec, tool_texts) if out else ([err], 0)
        llm_text = "" if not out else " ".join([out["summary"]] + [e["text"] for e in out["evidence"]])
        rec = {"quarter": dec["quarter"], "model": a.model, "prompt_version": PROMPT_VERSION,
               "rule_version": dec["rule_version"], "published": not probs, "problems": probs,
               "n_numbers": n_num, "seconds": round(t_llm, 1), "usage": usage, "calls": calls,
               "coverage": coverage(llm_text, dec), "rule_coverage": coverage(rule_txt, dec),
               "rule_verified": rule_chk["verified"], "rule_seconds": round(t_rule, 3), "output": out}
        rows.append(rec)
        print(dec["quarter"], "발행" if not probs else "보류", probs, rec["coverage"], usage, f"{t_llm:.0f}s",
              flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
