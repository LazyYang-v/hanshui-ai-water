"""下载本地 BGE 模型并构建向量缓存：uv run python -m app.ai.rag.build。"""

import json

from app.ai.rag.retrieval import load_index
from app.config import settings


def main():
    index = load_index(settings.rag_corpus_dir, settings.rag_model_cache, settings.rag_vector_cache,
                       allow_download=True)
    print(json.dumps(index.status(), ensure_ascii=False, indent=2))
    if not index.status()["dense_available"]:
        raise SystemExit("构建失败：只生成了 BM25 检索器，不能作为向量构建成功")


if __name__ == "__main__":
    main()
