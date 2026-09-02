"""에이전트 실행 흐름 재현. 사용: uv run python scripts/demo_trace.py [--question ...]

README "에이전트 동작 예시"를 직접 재현하기 위한 스크립트다. 실제 OpenAI를 호출하므로
수십 초 걸리고 토큰 비용이 발생한다. `.env`의 실제 키를 쓰기 위해 get_settings() 대신
Settings(_env_file=...)로 읽는다(테스트 conftest가 .env를 차단하는 것과 같은 이유).

트레이스 형식은 app/streamlit_app.py의 `_trace`와 같다. streamlit을 import하지 않으려고
같은 6줄 루프를 여기서 다시 구현한다.
"""

import argparse

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rent_agent.agents import supervisor as supervisor_module
from rent_agent.agents.llm import configure_tracing
from rent_agent.agents.supervisor import build_graph
from rent_agent.config import PROJECT_ROOT, Settings

# tests/agents/test_graph_integration.py::test_jeonse_diagnosis_calls_risk_and_report와 동일
DEFAULT_QUESTION = (
    "서울 강남구 까치마을 39.6㎡ 전세를 보려고 합니다. 보증금 4억 5천, 매매 시세 6억, "
    "근저당 채권최고액 1억 2천, 자기자금 2억, 연소득 4천만원입니다. 괜찮은 계약인가요?"
)


def trace_lines(result) -> list[str]:
    """이번 턴의 에이전트 → 도구 핸드오프 순서. 마지막 두 줄은 결정적 후처리의 흔적."""
    messages = result["messages"]
    lines: list[str] = []
    for m in supervisor_module._current_turn(messages):
        name = getattr(m, "name", None)
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"🧭 {name or 'supervisor'} → {tc['name']}")
        elif isinstance(m, ToolMessage) and name and not name.startswith("transfer_"):
            lines.append(f"🔧 {name} 결과 수신")
    if messages:
        last = messages[-1]
        lines.append(f"📝 최종 답변 작성: {getattr(last, 'name', None) or 'supervisor'}")
        forwarded = (getattr(last, "response_metadata", None) or {}).get("forwarded_from")
        if forwarded:
            lines.append(f"📎 {forwarded} 원문 그대로 전달 (supervisor 재작성 대체)")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="에이전트에 보낼 질문")
    args = parser.parse_args()

    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    configure_tracing(settings)
    graph = build_graph(settings)

    print(f"질문: {args.question}\n")
    result = graph.invoke({"messages": [HumanMessage(args.question)]})

    print("--- 실행 흐름 ---")
    print("\n".join(trace_lines(result)))
    print("\n--- 최종 답변 ---")
    print(result["messages"][-1].content)
