"""对运行中的 API 做真实问答验收（会调用已配置的 Flash）。

    uv run python -m evals.rag.smoke

只检查预期回答/拒答、实际模型、引用链接和摘录，不当作事实准确率评测。
"""

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from evals.http import alogin

CASES = [
    ("风机跳停、DO下降后，应该怎样排查氨氮升高？", False),
    ("参考案例的节电30.4%是这个Demo实测的成绩吗？", False),
    ("黄官镇污水处理站站长的手机号码是多少？", True),
    ("今天黄官镇出水氨氮的实时读数是多少？", True),
]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / 'results' / 'api-smoke.json')
    args = parser.parse_args()
    path = args.output
    if path.exists():
        parser.error("结果已存在，请用--output指定新的文件，不能覆盖历史验收")
    async with httpx.AsyncClient(base_url=args.url, timeout=90, trust_env=False) as client:
        await alogin(client)
        gateway = (await client.get('/api/ai/gateway')).raise_for_status().json()
        providers = gateway.get('providers', {})
        assert set(providers) == {'deepseek'} and providers['deepseek']['model'] == 'deepseek-flash'
        health = await client.get('/api/health')
        health.raise_for_status()
        catalog_response = await client.get('/api/ai/knowledge')
        catalog_response.raise_for_status()
        catalog = catalog_response.json()
        assert catalog['dense_available'], "验收要求真实向量模型可用"

        async def one(question, expected_refusal):
            response = await client.post('/api/ai/rag', json={'question': question})
            response.raise_for_status()
            body = response.json()
            valid_citations = True
            for citation in body['citations']:
                original = await client.get(citation['url'])
                original.raise_for_status()
                valid_citations &= ''.join(citation['quote'].split()) in ''.join(original.json()['text'].split())
            cached_refusal = (body['model'] is None and body['retrieval']['cache_hit']
                              and not body['generation_calls'] and body['refused'])
            passed = ((body['model'] == 'deepseek-flash' or cached_refusal) and body['refused'] == expected_refusal
                      and body['ok'] == (not expected_refusal) and valid_citations
                      and (expected_refusal or bool(body['citations'])))
            print(json.dumps({"question": question, "passed": passed, "answer": body['answer'],
                              "reason": body['reason'], "error": body['error'],
                              "validation_errors": body['validation_errors'], "ms": body['total_ms']},
                             ensure_ascii=False), flush=True)
            return {"passed": passed, "expected_refusal": expected_refusal,
                    "citations_verified": valid_citations, "result": body}

        results = await asyncio.gather(*(one(question, refused) for question, refused in CASES))
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"knowledge": catalog, "cases": results}, ensure_ascii=False, indent=2))
    if not all(r['passed'] for r in results):
        raise SystemExit(f"验收未全部通过，详情：{path}")
    print(f"4/4 API 场景通过；引用原文链接与摘录均核验通过。记录：{path}")


if __name__ == '__main__':
    asyncio.run(main())
