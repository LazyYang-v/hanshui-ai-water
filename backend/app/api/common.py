from datetime import datetime, timedelta

from fastapi import HTTPException, Query

from app.deps import LOCAL_TZ, Now
from app.domain.metrics import Metric
from app.services.queries import BUCKET_NAMES

MAX_SPAN = timedelta(days=31)


class TimeRange:
    """查询区间：默认最近 24 小时（以演示时钟为准）；不带时区的时间按北京时间解释。"""

    def __init__(
        self,
        now: Now,
        start: datetime | None = Query(None, description="起始时间，默认 end 前 24 小时"),
        end: datetime | None = Query(None, description="结束时间，默认演示时钟当前时间"),
        bucket: str = Query("auto", description=f"聚合粒度：auto / {' / '.join(BUCKET_NAMES)}"),
    ):
        end = aware(end) if end else now
        start = aware(start) if start else end - timedelta(hours=24)
        if start >= end:
            raise HTTPException(422, "start 必须早于 end")
        if end - start > MAX_SPAN:
            raise HTTPException(422, "查询区间不能超过 31 天")
        self.start, self.end, self.bucket = start, end, bucket


def aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=LOCAL_TZ)


def parse_metrics(raw: str | None, catalog: dict[str, Metric], default: list[str]) -> list[Metric]:
    keys = [k.strip() for k in raw.split(",") if k.strip()] if raw else default
    unknown = [k for k in keys if k not in catalog]
    if unknown:
        raise HTTPException(422, f"未知指标：{', '.join(unknown)}。可用：{', '.join(catalog)}")
    if not keys:
        raise HTTPException(422, "至少指定一个指标")
    return [catalog[k] for k in dict.fromkeys(keys)]
