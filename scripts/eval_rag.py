"""RAGAS로 지식 QA(RAG) 품질 평가. 실행: uv run python scripts/eval_rag.py

지표: faithfulness, answer_relevancy, context_precision.
- faithfulness의 컨텍스트는 **에이전트가 실제로 도구 호출로 본 문서**(ToolMessage 원문)를 쓴다.
  스크립트가 질문으로 직접 검색한 결과와 다를 수 있기 때문
  (에이전트는 질의를 바꿔 여러 번 검색할 수 있음).
- context_precision은 리트리버 자체의 품질이므로 원 질문으로 검색한 결과를 쓴다.
결과: eval/results/<날짜>.md(최신), eval/results/history.md(실행 이력 1행 추가),
eval/results/<타임스탬프>.json(gitignore).
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.messages import HumanMessage, ToolMessage
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from rent_agent.agents.knowledge_agent import CONTEXT_SEPARATOR, build_knowledge_agent
from rent_agent.agents.llm import configure_tracing
from rent_agent.config import get_settings
from rent_agent.rag.retriever import get_retriever

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def agent_contexts(messages) -> list[str]:
    """에이전트가 search_real_estate_knowledge 로 실제로 받은 문단들(중복 제거, 순서 유지)."""
    seen: dict[str, None] = {}
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "search_real_estate_knowledge":
            for chunk in str(m.content).split(CONTEXT_SEPARATOR):
                if chunk.strip():
                    seen.setdefault(chunk.strip(), None)
    return list(seen)


async def main() -> None:
    settings = get_settings()
    configure_tracing(settings)
    retriever = get_retriever(settings)
    agent = build_knowledge_agent(settings, retriever)

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    # 기본 max_tokens=1024는 한국어 법령 청크의 NLI 판정 JSON이 잘려
    # IncompleteOutputException (실측)
    judge = llm_factory(settings.openai_model, client=client, max_tokens=4096)
    emb = RagasOpenAIEmbeddings(client=client, model=settings.openai_embedding_model)
    faith = Faithfulness(llm=judge)
    relev = AnswerRelevancy(llm=judge, embeddings=emb)
    prec = ContextPrecision(llm=judge)

    rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for row in rows:
        q = row["question"]
        result = agent.invoke({"messages": [HumanMessage(q)]})
        answer = result["messages"][-1].content
        retriever_contexts = [d.page_content for d in retriever.invoke(q)]
        seen_contexts = agent_contexts(result["messages"]) or retriever_contexts
        f = await faith.ascore(user_input=q, response=answer, retrieved_contexts=seen_contexts)
        r = await relev.ascore(user_input=q, response=answer)
        p = await prec.ascore(
            user_input=q, reference=row["reference"], retrieved_contexts=retriever_contexts
        )
        results.append(
            {
                **row,
                "answer": answer,
                "agent_contexts": seen_contexts,
                "faithfulness": f.value,
                "answer_relevancy": r.value,
                "context_precision": p.value,
            }
        )
        print(f"[{len(results)}/{len(rows)}] F={f.value:.2f} R={r.value:.2f} P={p.value:.2f}  {q}")

    summary = {k: round(mean(x[k] for x in results), 3) for k in METRICS}
    now = datetime.now()
    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{now:%Y-%m-%d_%H%M}.json").write_text(
        json.dumps({"summary": summary, "rows": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    config_line = (
        f"모델: {settings.openai_model} / 임베딩: {settings.openai_embedding_model} "
        f"/ k={settings.retriever_k} / n={len(rows)} "
        f"/ faithfulness 컨텍스트=에이전트 ToolMessage"
    )
    md = [
        f"# RAG 평가 결과 {now:%Y-%m-%d}",
        "",
        config_line,
        "",
        "| 지표 | 평균 |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| 질문 | F | R | P |",
        "|---|---|---|---|",
        *[
            f"| {x['question']} | {x['faithfulness']:.2f} | "
            f"{x['answer_relevancy']:.2f} | {x['context_precision']:.2f} |"
            for x in results
        ],
        "",
    ]
    (out_dir / f"{now:%Y-%m-%d}.md").write_text("\n".join(md), encoding="utf-8")

    history = out_dir / "history.md"
    if not history.exists():
        history.write_text(
            "# RAG 평가 실행 이력\n\n| 실행 시각 | F | R | P | 설정 |\n|---|---|---|---|---|\n",
            encoding="utf-8",
        )
    with history.open("a", encoding="utf-8") as fh:
        fh.write(
            f"| {now:%Y-%m-%d %H:%M} | {summary['faithfulness']} | {summary['answer_relevancy']} "
            f"| {summary['context_precision']} | {config_line} |\n"
        )
    print("\n요약:", summary)


if __name__ == "__main__":
    asyncio.run(main())
