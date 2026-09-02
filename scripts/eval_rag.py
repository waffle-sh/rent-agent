"""RAGAS로 지식 QA(RAG) 품질 평가. 실행: uv run python scripts/eval_rag.py

지표 3개와 선택 근거:
- faithfulness: 답이 검색된 문서에 근거하는가. 법령 QA에서 환각은 곧 오답이므로 최우선 지표.
- answer_relevancy: 질문에 맞게 답했는가.
- context_precision: 리트리버가 정답 관련 문서를 상위에 올렸는가(검색 품질).

결과는 eval/results/<날짜>.json(gitignore) + <날짜>.md(커밋)로 저장한다.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from statistics import mean

from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness

from rent_agent.agents.knowledge_agent import build_knowledge_agent
from rent_agent.agents.llm import configure_tracing
from rent_agent.config import get_settings
from rent_agent.rag.retriever import get_retriever

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("faithfulness", "answer_relevancy", "context_precision")


async def main() -> None:
    settings = get_settings()
    configure_tracing(settings)
    retriever = get_retriever(settings)
    agent = build_knowledge_agent(settings, retriever)

    # 평가자(judge) LLM은 피평가 에이전트와 같은 모델을 쓴다. 별도 모델을 쓰면
    # 점수 차이가 "RAG 품질"인지 "judge 성향"인지 구분되지 않는다.
    # max_tokens: ragas 기본값 1024로는 한국어 법령 청크에 대한 NLI verdict JSON이 잘려
    # instructor가 IncompleteOutputException을 던진다(ragas/llms/base.py 권고대로 4096으로 올림).
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    judge = llm_factory(settings.openai_model, client=client, max_tokens=4096)
    emb = RagasOpenAIEmbeddings(client=client, model=settings.openai_embedding_model)
    faith = Faithfulness(llm=judge)
    relev = AnswerRelevancy(llm=judge, embeddings=emb)
    prec = ContextPrecision(llm=judge)

    dataset = (ROOT / "eval" / "dataset.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in dataset.splitlines() if line.strip()]

    results = []
    for row in rows:
        question = row["question"]
        # 에이전트가 실제로 보는 것과 같은 컨텍스트를 평가에 넣기 위해 같은 리트리버를 쓴다.
        contexts = [d.page_content for d in retriever.invoke(question)]
        answer = agent.invoke({"messages": [HumanMessage(question)]})["messages"][-1].content
        f = await faith.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
        r = await relev.ascore(user_input=question, response=answer)
        p = await prec.ascore(
            user_input=question, reference=row["reference"], retrieved_contexts=contexts
        )
        results.append(
            {
                **row,
                "answer": answer,
                "faithfulness": f.value,
                "answer_relevancy": r.value,
                "context_precision": p.value,
            }
        )
        print(
            f"[{len(results)}/{len(rows)}] F={f.value:.2f} R={r.value:.2f} "
            f"P={p.value:.2f}  {question}"
        )

    summary = {k: round(mean(x[k] for x in results), 3) for k in METRICS}
    stamp = datetime.now().strftime("%Y-%m-%d")
    out_dir = ROOT / "eval" / "results"
    (out_dir / f"{stamp}.json").write_text(
        json.dumps({"summary": summary, "rows": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        f"# RAG 평가 결과 {stamp}",
        "",
        f"모델: {settings.openai_model} / 임베딩: {settings.openai_embedding_model} "
        f"/ k={settings.retriever_k}",
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
    ]
    (out_dir / f"{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n요약:", summary)


if __name__ == "__main__":
    asyncio.run(main())
