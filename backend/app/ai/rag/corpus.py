"""受版本管理的 Markdown 知识库：按标题/段落切分，保留来源与行号。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

DEFAULT_CORPUS = Path(__file__).resolve().parents[3] / "knowledge"
SECTION = re.compile(r"^##\s+(.+?)(?:\s+\{#([a-z0-9-]+)\})?\s*$")
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    title: str
    heading: str
    text: str
    source_type: str
    source: str
    source_url: str | None
    source_page: int | None
    line_start: int
    line_end: int

    def to_dict(self) -> dict:
        return {**asdict(self), "url": f"/api/ai/knowledge/chunks/{self.id}"}

    @property
    def search_text(self) -> str:
        return f"{self.title}\n{self.heading}\n{self.text}"


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    source_type: str
    source: str
    source_url: str | None
    content: str
    chunks: tuple[Chunk, ...]

    def summary(self) -> dict:
        return {"id": self.id, "title": self.title, "source_type": self.source_type,
                "source": self.source, "source_url": self.source_url, "chunks": len(self.chunks),
                "url": f"/api/ai/knowledge/{self.id}"}


def load_document(path: Path, max_chars: int = 500, overlap: int = 60) -> Document:
    """保留标题边界，长段按字符窗口拆分；每个窗口均能回溯到原文行号。"""
    if max_chars < 100 or not 0 <= overlap < max_chars:
        raise ValueError("切分参数非法")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path.name} 缺少来源 frontmatter")
    try:
        end = lines.index("---", 1)
        meta = yaml.safe_load("\n".join(lines[1:end]))
        doc_id, title, source_type, source = (meta[k] for k in ("id", "title", "source_type", "source"))
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"{path.name} 的来源元数据不完整") from exc
    if not all(isinstance(v, str) and v.strip() for v in (doc_id, title, source_type, source)):
        raise ValueError(f"{path.name} 的来源字段必须为非空字符串")
    if not SAFE_ID.fullmatch(doc_id) or source_type not in {"demo", "reference"}:
        raise ValueError(f"{path.name} 的 id 或来源类型非法")
    source_url = meta.get("source_url")
    if source_url and not source_url.startswith("https://"):
        raise ValueError("来源链接必须是 HTTPS")

    chunks = []
    headings = [(i, SECTION.fullmatch(line)) for i, line in enumerate(lines[end + 1:], end + 1)]
    headings = [(i, match) for i, match in headings if match]
    if not headings:
        raise ValueError(f"{path.name} 没有二级标题")
    anchors = set()
    for pos, (i, match) in enumerate(headings):
        heading, anchor = match.groups()
        if not anchor or anchor in anchors:
            raise ValueError(f"{path.name} 的二级标题须有唯一 ASCII anchor")
        anchors.add(anchor)
        stop = headings[pos + 1][0] if pos + 1 < len(headings) else len(lines)
        body = "\n".join(lines[i + 1:stop])
        part, start = 1, 0
        while start < len(body):
            finish = min(start + max_chars, len(body))
            if finish < len(body):
                # 优先选靠后的段落/句子边界，防止短窗口和无意义碎片。
                boundary = max(body.rfind("\n\n", start + max_chars // 2, finish),
                               body.rfind("。", start + max_chars // 2, finish))
                if boundary >= 0:
                    finish = boundary + 1
            segment = body[start:finish]
            text = segment.strip()
            if text:
                left = start + len(segment) - len(segment.lstrip())
                right = finish - (len(segment) - len(segment.rstrip()))
                line_start = i + 2 + body[:left].count("\n")
                line_end = i + 2 + body[:right].count("\n")
                page = (meta.get("section_pages") or {}).get(anchor)
                chunks.append(Chunk(f"{doc_id}:{anchor}:{part}", doc_id, title, heading, text,
                                    source_type, source, source_url, page, line_start, line_end))
                part += 1
            if finish == len(body):
                break
            start = max(start + 1, finish - overlap)
    if not chunks:
        raise ValueError(f"{path.name} 没有可索引正文")
    return Document(doc_id, title, source_type, source, source_url, raw, tuple(chunks))


def load_corpus(directory: Path = DEFAULT_CORPUS) -> list[Document]:
    docs = [load_document(path) for path in sorted(directory.glob("*.md"))]
    if not docs:
        raise ValueError(f"知识库为空：{directory}")
    if len({d.id for d in docs}) != len(docs):
        raise ValueError("知识库文档 id 重复")
    return docs


def corpus_fingerprint(documents: list[Document]) -> str:
    raw = "\n".join(d.content for d in documents)
    return hashlib.sha256(raw.encode()).hexdigest()
