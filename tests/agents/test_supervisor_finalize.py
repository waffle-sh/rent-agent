from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rent_agent.agents.supervisor import preserve_worker_answer

REPORT = "## 종합 판정: 위험\n...\n본 리포트는 참고 정보이며 법률·금융 자문이 아닙니다."


def _flow(final_supervisor_text: str) -> list:
    return [
        HumanMessage("보증금 4.5억 시세 6억 진단해줘", id="h1"),
        AIMessage(
            "",
            name="supervisor",
            id="s1",
            tool_calls=[{"name": "transfer_to_report_agent", "args": {}, "id": "c1"}],
        ),
        ToolMessage("transferred", tool_call_id="c1", name="transfer_to_report_agent", id="t1"),
        AIMessage(REPORT, name="report_agent", id="r1"),
        AIMessage(
            "",
            name="report_agent",
            id="r2",
            tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "c2"}],
        ),
        ToolMessage("back", tool_call_id="c2", name="transfer_back_to_supervisor", id="t2"),
        AIMessage(final_supervisor_text, name="supervisor", id="s2"),
    ]


def test_replaces_supervisor_paraphrase_with_report_verbatim():
    out = preserve_worker_answer({"messages": _flow("요약하면 위험합니다. 근저당이 있어요.")})
    [replacement] = out["messages"]
    assert replacement.id == "s2" and replacement.name == "supervisor"
    assert replacement.content == REPORT
    assert replacement.response_metadata["forwarded_from"] == "report_agent"


def test_noop_when_supervisor_already_verbatim():
    assert preserve_worker_answer({"messages": _flow(REPORT + "\n")}) == {}


def test_noop_when_no_worker_answer_in_this_turn():
    msgs = [
        HumanMessage("안녕", id="h1"),
        AIMessage("보증금과 매매 시세를 알려주세요.", name="supervisor", id="s1"),
    ]
    assert preserve_worker_answer({"messages": msgs}) == {}


def test_uses_only_messages_after_last_human_turn():
    msgs = _flow("...") + [
        HumanMessage("고마워", id="h2"),
        AIMessage("도움이 되었다니 다행입니다.", name="supervisor", id="s3"),
    ]
    assert preserve_worker_answer({"messages": msgs}) == {}


def test_knowledge_answer_is_also_preserved():
    msgs = [
        HumanMessage("대항력은 언제 생기나요", id="h1"),
        AIMessage(
            "전입신고 다음 날 0시부터입니다.\n근거: 주택임대차보호법 제3조 https://law.go.kr",
            name="knowledge_agent",
            id="k1",
        ),
        AIMessage("다음 날부터 생깁니다.", name="supervisor", id="s1"),
    ]
    [rep] = preserve_worker_answer({"messages": msgs})["messages"]
    assert "https://law.go.kr" in rep.content
