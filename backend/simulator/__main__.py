"""模拟器命令行。

  python -m simulator run [--backfill-days 7] [--speed 1]   # 回填（库为空时）并实时运行
  python -m simulator run --reset                           # 清空运行数据，重新回填
  python -m simulator inject storm                          # 全区暴雨
  python -m simulator inject blower_fault --plant P01       # 指定厂站的场景
  python -m simulator inject pipe_leak --plant P05 --duration 120 --param effective=0.5
  python -m simulator clear [--plant P03] [--kind storm]    # 结束场景
  python -m simulator speed 60                              # 演示时钟 60 倍速
  python -m simulator scenarios                             # 列出可用场景
"""

import argparse
import json
import logging
import sys
import uuid

import redis

from . import runner
from .scenarios import DEFAULT_DURATION_MIN, KINDS


def _send(cmd: dict) -> None:
    r = redis.Redis.from_url(runner.REDIS_URL, decode_responses=True)
    sub = r.pubsub(ignore_subscribe_messages=True)
    sub.subscribe(runner.CH_EVENTS)
    sub.get_message(timeout=1)  # 消费订阅确认，确保发布命令前订阅已生效
    cmd["id"] = uuid.uuid4().hex
    if r.publish(runner.CH_CONTROL, json.dumps(cmd, ensure_ascii=False)) == 0:
        sys.exit("模拟器未运行（没有订阅者）")
    reply = None
    while (msg := sub.get_message(timeout=10)) is not None:
        data = json.loads(msg["data"])
        if data.get("id") == cmd["id"]:
            reply = data
            break
    if reply is None:
        sys.exit("模拟器未在 10 秒内响应")
    print(reply["message"])
    if not reply["ok"]:
        sys.exit(1)


def _param(s: str) -> tuple[str, float | str]:
    k, _, v = s.partition("=")
    try:
        return k, float(v)
    except ValueError:
        return k, v


def main() -> None:
    p = argparse.ArgumentParser(prog="simulator", description="汉水智控 数据模拟器")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="回填历史并实时运行")
    pr.add_argument("--backfill-days", type=int, default=7)
    pr.add_argument("--backfill-step", type=float, default=60.0, help="回填步长（秒）")
    pr.add_argument("--speed", type=float, default=1.0)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--reset", action="store_true", help="清空运行数据后重新回填")

    pi = sub.add_parser("inject", help="注入场景")
    pi.add_argument("kind", choices=list(KINDS))
    pi.add_argument("--plant", help="厂站编码，如 P03（storm 可不填）")
    pi.add_argument("--duration", type=float, help="持续分钟数（仿真时间）")
    pi.add_argument("--param", action="append", default=[], type=_param, help="场景参数，如 peak_mm_h=20")

    pc = sub.add_parser("clear", help="结束场景")
    pc.add_argument("--plant")
    pc.add_argument("--kind", choices=list(KINDS))

    ps = sub.add_parser("speed", help="设置演示时钟倍速")
    ps.add_argument("value", type=float)

    sub.add_parser("scenarios", help="列出可用场景")

    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.cmd == "run":
        runner.run(a.backfill_days, a.backfill_step, a.speed, a.seed, a.reset)
    elif a.cmd == "inject":
        _send({"action": "inject", "kind": a.kind, "plant": a.plant, "duration_min": a.duration,
               "params": dict(a.param)})
    elif a.cmd == "clear":
        _send({"action": "clear", "plant": a.plant, "kind": a.kind})
    elif a.cmd == "speed":
        _send({"action": "speed", "value": a.value})
    elif a.cmd == "scenarios":
        for k, desc in KINDS.items():
            print(f"{k:14s} 默认 {DEFAULT_DURATION_MIN[k]:>4d} 分钟  {desc}")


if __name__ == "__main__":
    main()
