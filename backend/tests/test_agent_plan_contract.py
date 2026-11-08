"""规划格式容错仅限可答路径的空理由，不能补造统计参数或拒答理由。"""

import json

import pytest
from pydantic import ValidationError

from app.ai.agent.models import Decision
from app.ai.agent.prompts import PLAN_SYSTEM
from tests.test_agent import decision


@pytest.mark.parametrize("action", ["tools", "answer"])
def test_missing_empty_reason_defaults_only_for_non_refusal(action):
    value = decision([("sql_query", "黄官镇指定日氨氮")] if action == "tools" else (), action=action)
    del value["reason"]
    assert Decision.model_validate_json(json.dumps(value)).reason == ""


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_refusal_still_requires_an_explicit_nonblank_reason(reason):
    value = decision(action="refuse", intent="unsupported")
    if reason is None:
        del value["reason"]
    else:
        value["reason"] = reason
    with pytest.raises(ValidationError):
        Decision.model_validate_json(json.dumps(value))


def test_missing_statistics_parameters_cannot_execute():
    value = decision([("plant_statistics", "黄官镇指定日氨氮统计")])
    with pytest.raises(ValidationError, match="必须提供parameters"):
        Decision.model_validate_json(json.dumps(value))


def test_prompt_complete_example_validates_without_tool_parameter_ambiguity():
    example = next(line for line in PLAN_SYSTEM.splitlines() if line.startswith('{"intent"'))
    plan = Decision.model_validate_json(example)
    assert [call.tool for call in plan.calls] == ["plant_statistics", "sql_query", "rag_search"]
    assert plan.calls[0].parameters is not None
    assert all(call.parameters is None for call in plan.calls[1:])
