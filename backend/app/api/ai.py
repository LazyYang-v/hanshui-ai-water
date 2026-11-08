from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.ai.text2sql import Text2SQL
from app.analytics.statistics import StatisticsQuery
from app.deps import Now, ReleasedNow

router = APIRouter(prefix="/api/ai", tags=["AI"])


class Text2SQLRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(min_length=2, max_length=300, examples=["昨天各厂处理了多少水？"])


def _text2sql(request: Request) -> Text2SQL:
    t2s: Text2SQL | None = getattr(request.app.state, "text2sql", None)
    if t2s is None or not t2s.gateway.available:
        raise HTTPException(503, "大模型未配置（在 .env 中设置 DEEPSEEK_API_KEY 等）")
    return t2s


@router.post("/text2sql", summary="自然语言查数：生成 SQL → 安全校验 → 只读执行 → 出错自修复")
async def text2sql(body: Text2SQLRequest, request: Request, now: Now) -> dict:
    return (await _text2sql(request).ask(body.question.strip(), now)).to_dict()


@router.post("/statistics", summary="固定只读统计：异常时长、原始与均值极值、排名及数据覆盖")
async def statistics(body: StatisticsQuery, request: Request, now: ReleasedNow) -> dict:
    service = getattr(request.app.state, "statistics", None)
    if service is None:
        raise HTTPException(503, "只读统计服务暂不可用")
    try:
        return await service.query(body, now)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/gateway", summary="LLM 网关状态：各厂商调用量、P50/P95 延迟、token、熔断")
async def gateway(request: Request) -> dict:
    gw = getattr(request.app.state, "gateway", None)
    if gw is None:
        return {"available": False}
    return {"available": gw.available, **gw.summary()}
