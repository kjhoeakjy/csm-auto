"""정기 실행 (APScheduler). 운영 모드와 과거 재현 모드가 같은 처리 함수(pipeline.run_quarter)를 호출

  live   : 매 분기 첫 달 15일 02:00 에 직전 분기 처리 (CronTrigger). 실제 운영 서버는 없으므로 등록만 시연
  replay : 가상 시계를 1초에 1분기씩 진행하며 2015Q4~2018Q4 처리 (무인 배치)

실행: .venv/Scripts/python modelops/analysis/scheduler.py --mode replay
"""
import argparse
import logging

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from pipeline import Context, run_quarter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("modelops")


def replay(start="2015Q4", end="2018Q4", with_retrain=True):
    ctx = Context.load()
    quarters = list(pd.period_range(start, end, freq="Q"))
    sched = BlockingScheduler()

    def tick():
        if not quarters:
            sched.shutdown(wait=False)
            return
        q = quarters.pop(0)
        r = run_quarter(ctx, q, with_retrain=with_retrain)
        log.info("%s %s | %s | %s | 후보 실험 %s%s", q, r["run_status"], r["status"],
                 r["recommendation"], r.get("candidate_experiment"),
                 " (이미 처리됨, 재생성 안 함)" if r.get("skipped") else "")

    sched.add_job(tick, "interval", seconds=1, max_instances=1, coalesce=True)
    sched.start()


def live():
    sched = BlockingScheduler()

    def job():
        # 실행마다 최신 적재 자료를 다시 읽음 (원천 갱신 반영). 신규 원천 적재와 점수 산출은 m01, m03 단계
        ctx = Context.load()
        q = pd.Timestamp.now().to_period("Q") - 1
        run_quarter(ctx, q)

    sched.add_job(job, CronTrigger(month="1,4,7,10", day=15, hour=2), id="quarterly_check")
    log.info("등록된 작업: %s", [str(j.trigger) for j in sched.get_jobs()])
    sched.start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    a = ap.parse_args()
    replay() if a.mode == "replay" else live()
