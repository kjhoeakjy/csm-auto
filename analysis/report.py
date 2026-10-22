"""분기 보고서 생성(규칙 기반)과 수치 대조

verify(text, decision): 보고서 본문의 모든 숫자·분기 표기가 decision.json 의 값에서 나온 것인지 확인.
대조 실패 시 발행 보류. LLM 보고서도 같은 함수로 검사.
"""
import re

from candidates import LABEL

Q_RE = re.compile(r"\b\d{4}Q[1-4]\b")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?")


def _walk(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _walk(v)
    else:
        yield o


def allowed_tokens(dec: dict):
    nums, qs = set(), set()
    for v in _walk(dec):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            x = float(v)
            for s in (f"{x:.0f}", f"{x:,.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}",
                      f"{x * 100:.1f}", f"{x * 100:.0f}", f"{abs(x):.1f}", f"{abs(x):.2f}",
                      f"{abs(x) * 100:.1f}"):
                nums.add(s.lstrip("+"))
        elif isinstance(v, str):
            qs.update(Q_RE.findall(v))
            nums.update(n.replace(",", "") for n in NUM_RE.findall(Q_RE.sub(" ", v)))
    nums |= {n.replace(",", "") for n in list(nums)}
    nums |= {q[:4] for q in qs}                                 # 분기에서 나온 연도 표기 (예: 2015년)
    nums |= {"1", "2", "3", "12", "18", "24", "30", "4"}      # 문서 고정 표현 (근거 번호, 개월 수)
    return nums, qs


def verify(text: str, dec: dict) -> dict:
    nums, qs = allowed_tokens(dec)
    bad_q = [q for q in Q_RE.findall(text) if q not in qs]
    body = Q_RE.sub(" ", text)
    toks = [t.replace(",", "").lstrip("+") for t in NUM_RE.findall(body)]
    bad_n = [t for t in toks if t not in nums and t.lstrip("-") not in nums]
    return {"verified": not bad_q and not bad_n, "n_numbers": len(toks) + len(Q_RE.findall(text)),
            "unmatched": bad_q + bad_n}


def _p(x):
    return f"{x * 100:.1f}%"


FEAT_KO = {
    "int_rate": "금리", "grade_num": "세부 등급", "fico": "FICO 점수", "dti": "부채상환비율(DTI)",
    "loan_amnt": "대출금액", "installment": "월 상환액", "term_m": "대출 기간", "emp_years": "근속 연수",
    "log_annual_inc": "연소득", "loan_to_income": "소득 대비 대출금액", "revol_util": "리볼빙 사용률",
    "log_revol_bal": "리볼빙 잔액", "inq_last_6mths": "최근 6개월 신용조회 수",
    "mths_since_last_record": "공공기록 후 경과월", "mths_since_last_delinq": "최근 연체 후 경과월",
    "credit_hist_m": "신용 이력 길이", "open_acc": "보유 계좌 수", "total_acc": "전체 계좌 수",
    "delinq_2yrs": "최근 2년 연체 수", "pub_rec": "공공기록 수", "pub_rec_bankruptcies": "파산 기록 수",
}


def card_evidence(dec: dict) -> list[dict]:
    """결과 카드용 근거 3개 (쉬운 표현). 수치는 dec 에서만 가져오고 verify 로 대조"""
    if dec.get("run_status") != "성공" or not dec.get("lag"):
        return []
    ev = []
    lead = dec.get("lead") or {}
    if lead.get("skipped"):
        ev.append({"title": "신규 대출 특성·데이터 품질", "text": f"{lead['skipped']}: 신규 대출 점검 유보"})
    elif not lead.get("schema_ok"):
        ev.append({"title": "신규 대출 특성·데이터 품질", "text": "데이터 형식 오류로 점검 불가 (판정 유보)"})
    else:
        t = lead["top_recent"][0]
        same = "학습 시점과 비슷함" if lead["score_psi_ref"] < 0.1 else "학습 시점과 달라짐"
        ev.append({"title": "신규 대출 특성·데이터 품질",
                   "short": f"점수 분포 {'안정' if lead['score_psi_ref'] < 0.1 else '변화'} (PSI {lead['score_psi_ref']:.3f}), 데이터 형식 정상",
                   "text": (f"이번 분기 신규 {lead['n']:,}건. 모형 점수 분포는 {same} "
                            f"(PSI {lead['score_psi_ref']:.3f}). 직전 1년 대비 가장 크게 바뀐 입력은 "
                            f"{FEAT_KO.get(t['feature'], t['feature'])} (PSI {t['psi']:.2f})")})
    l0 = dec["lag"][0]
    txt = (f"결과가 확정된 {l0['cohort_q']} 실행분의 실제 상각률 {_p(l0['actual'])}, "
           f"예측 {_p(l0['pred'])} (실제가 예측의 {l0['o_e_ratio']:.2f}배)")
    if dec.get("early"):
        e0 = dec["early"][0]
        txt += f". 더 최근 {e0['cohort_q']} 실행분의 12개월 중간 위험 신호도 기대의 {e0['o_e_ratio']:.2f}배"
    ev.append({"title": "예측과 실제의 차이", "text": txt,
               "short": f"실제 상각률이 예측의 {l0['o_e_ratio']:.2f}배 ({l0['cohort_q']} 실행분)"})
    c = dec.get("candidates")
    if c is None:
        ev.append({"title": "개선안 별도 검증", "text": "조치 검토 기준에 못 미쳐 개선안 실험 생략",
                   "short": "조치 기준 미달로 실험 생략"})
    else:
        t = {r["candidate"]: r for r in c["table"]}
        m = t["maintain"]
        if c["chosen"]:
            k = t[c["chosen"]]
            ev.append({"title": "개선안 별도 검증",
                       "text": (f"{LABEL[c['chosen']]}을 1년 전 데이터로 만들어 이후 확정된 "
                                f"{c['sel_q'][0]}~{c['sel_q'][-1]} 실행분에 적용: 실제/예측 "
                                f"{m['o_e_ratio']:.2f}배에서 {k['o_e_ratio']:.2f}배로, 예측 오차(log loss) "
                                f"{k['rel_ll_gain'] * 100:.1f}% 감소"),
                       "short": f"{LABEL[c['chosen']]}, 실제/예측 {m['o_e_ratio']:.2f}에서 {k['o_e_ratio']:.2f}로"})
        else:
            ev.append({"title": "개선안 별도 검증",
                       "text": (f"개선안 {len(t) - 1}개를 1년 전 데이터로 만들어 {c['sel_q'][0]}~"
                                f"{c['sel_q'][-1]} 실행분에 시험했으나 모두 사전 기준 미달: 변경 보류"),
                       "short": f"후보 {len(t) - 1}개 모두 기준 미달, 변경 보류"})
    return ev


def build(dec: dict) -> str:
    """정량 근거 JSON -> 담당자 보고서 (수치는 모두 dec 에서 옴)"""
    if dec.get("run_status") != "성공":
        return ""
    L = [f"# {dec['quarter']} 신용평가모형 점검 보고", "",
         f"- 상태: {dec['status']}", f"- 권고: {dec['recommendation']}", ""]
    L.append("## 판단 근거")
    for i, e in enumerate(dec.get("evidence", []), 1):
        L.append(f"{i}. {e['title']}: {e['text']}")
    L.append("")
    lag = dec.get("lag") or []
    if lag:
        l0 = lag[0]
        L.append("## 결과 확정 코호트 (24개월)")
        L.append(f"- {l0['cohort_q']} 실행 {l0['n']:,}건: 실제 {_p(l0['actual'])}, 예측 {_p(l0['pred'])}, "
                 f"차이 {l0['gap_pp']:+.1f}%p, 실제/예측 {l0['o_e_ratio']:.2f}, 보정 z {l0['z_adj']:.1f}, "
                 f"AUC {l0['auc']:.3f} ({l0['auc_lo']:.3f}~{l0['auc_hi']:.3f})")
        if len(lag) > 1:
            l1 = lag[1]
            L.append(f"- 직전 {l1['cohort_q']} 실행분: 실제/예측 {l1['o_e_ratio']:.2f}, 보정 z {l1['z_adj']:.1f}")
        if l0.get("grade"):
            g = ", ".join(f"{r['grade']} {r['o_e_ratio']:.2f}" for r in l0["grade"])
            L.append(f"- 등급별 실제/예측: {g}")
        L.append("")
    early = dec.get("early") or []
    if early:
        e0 = early[0]
        L.append("## 조기 결과 (12개월)")
        L.append(f"- {e0['cohort_q']} 실행 {e0['n']:,}건: 실제 {_p(e0['actual'])}, 기대 {_p(e0['pred'])}, "
                 f"실제/기대 {e0['o_e_ratio']:.2f}, 보정 z {e0['z_adj']:.1f}")
        L.append("")
    lead = dec.get("lead") or {}
    if lead.get("schema_ok") and not lead.get("skipped"):
        top = ", ".join(f"{t['feature']} {t['psi']:.2f}" for t in lead["top_recent"])
        L.append("## 신규 실행 대출 (선행 점검)")
        L.append(f"- {lead['n']:,}건, 평균 예측 {_p(lead['score_mean'])} (학습 코호트 {_p(lead['score_mean_ref'])})")
        L.append(f"- 직전 4분기 대비 변화가 큰 변수(PSI): {top}")
        L.append("")
    c = dec.get("candidates")
    if c:
        L.append("## 개선 후보 비교 (1년 전 시점 데이터로 적합, 이후 확정 코호트로 평가)")
        rows = sorted(c["table"], key=lambda r: r["logloss"])[:4]
        for r in rows:
            mark = "선택" if r["candidate"] == c["chosen"] else ("통과" if r["passed"] else "탈락")
            L.append(f"- {LABEL[r['candidate']]}: 실제/예측 {r['o_e_ratio']:.2f}, "
                     f"log loss 개선 {r['rel_ll_gain'] * 100:.1f}%, AUC {r['auc']:.3f} [{mark}]")
        L.append("")
    L.append("## 유의 사항")
    L.append("- 원인은 단정하지 않음. 심사 기준·상품 구성·경기 요인은 담당자 조사 항목")
    L.append("- 모형 변경·적용은 담당자 검토와 승인 후 진행")
    L.append("- 공개 대출 데이터 기반 과거 분기 재현. 상각 시점은 마지막 납부월 기준 근사")
    return "\n".join(L)
