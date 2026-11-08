"""评测判分逻辑的单元测试。"""

from evals.text2sql.score import match, norm


def test_extra_columns_and_column_order():
    gold = [["南郑", 100.0], ["协税", 5.0]]
    pred = [[5.0, "协税", "P02"], [100.0, "南郑", "P01"]]
    assert match(gold, pred)


def test_row_count_must_match():
    assert not match([["a"]], [["a"], ["b"]])
    assert match([], [])


def test_numeric_tolerance():
    assert match([[0.12345]], [[0.12]])        # 绝对误差 ≤ 0.01
    assert match([[69483.4]], [[69483]])       # 相对误差 ≤ 1%
    assert not match([[100.0]], [[102.0]])
    assert match([[7]], [[7.0]])


def test_ordered_flag():
    gold = [["a", 1.0], ["b", 2.0]]
    rev = [["b", 2.0], ["a", 1.0]]
    assert match(gold, rev, ordered=False)
    assert not match(gold, rev, ordered=True)


def test_rows_must_align_not_just_columns():
    gold = [["a", 1.0], ["b", 2.0]]
    crossed = [["a", 2.0], ["b", 1.0]]   # 每列的取值集合相同，但行对不上
    assert not match(gold, crossed)


def test_datetime_normalization():
    assert norm("2026-11-03T00:00:00+08:00") == "2026-11-03"
    assert norm("2026-11-03") == "2026-11-03"
    assert norm("2026-11-03T01:00:00+00:00") == "2026-11-03T09:00:00"
    assert match([["2026-11-03T00:00:00+08:00", 5.0]], [["2026-11-03", 5.0]])
    # 必须明确允许只答时刻，普通日期题不能丢掉日期。
    assert not match([["2026-10-31T14:01:00+08:00"]], [["14:01"]])
    assert match([["2026-10-31T14:01:00+08:00"]], [["14:01"]], time_only=True)
    assert not match([["2026-10-31T14:01:00+08:00"]], [["14:02"]], time_only=True)


def test_duplicate_values_need_distinct_columns():
    gold = [[1.0, 1.0]]
    assert not match(gold, [[1.0]])
    assert match(gold, [[1.0, 1.0]])


def test_only_explicit_refusal_scores_as_correct():
    from app.ai.text2sql.service import Attempt, Text2SQLResult
    from evals.text2sql.run import judge
    question = {"id": "U", "q": "下周天气", "sql": None}
    malformed = Text2SQLResult("下周天气", ok=False, attempts=[Attempt(None, "JSON 格式错误")])
    assert not judge(question, malformed, {})["correct"]
    refusal = Text2SQLResult("下周天气", ok=False, refused=True, attempts=[Attempt(None)])
    assert judge(question, refusal, {})["correct"]
