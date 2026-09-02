from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rent_agent.agents.supervisor import needs_report, preserve_worker_answer

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


def _risk_only_flow() -> list:
    return [
        HumanMessage("보증금 4.5억 시세 6억 진단해줘", id="h1"),
        AIMessage(
            "",
            name="supervisor",
            id="s1",
            tool_calls=[{"name": "transfer_to_risk_agent", "args": {}, "id": "c1"}],
        ),
        ToolMessage("transferred", tool_call_id="c1", name="transfer_to_risk_agent", id="t1"),
        AIMessage(
            "",
            name="risk_agent",
            id="r1",
            tool_calls=[{"name": "assess_jeonse_risk", "args": {}, "id": "c2"}],
        ),
        ToolMessage('{"level": "위험"}', tool_call_id="c2", name="assess_jeonse_risk", id="t2"),
        AIMessage("전세가율 75%로 위험입니다.", name="risk_agent", id="r2"),
        AIMessage(
            "",
            name="risk_agent",
            id="r3",
            tool_calls=[{"name": "transfer_back_to_supervisor", "args": {}, "id": "c3"}],
        ),
        ToolMessage("back", tool_call_id="c3", name="transfer_back_to_supervisor", id="t3"),
        AIMessage("위험합니다. 조심하세요.", name="supervisor", id="s2"),
    ]


def test_needs_report_when_risk_tool_ran_without_report():
    assert needs_report({"messages": _risk_only_flow()}) == "report"


def test_no_report_needed_when_report_exists():
    assert needs_report({"messages": _flow("요약")}) == "preserve"


def test_no_report_needed_for_knowledge_only_turn():
    msgs = [
        HumanMessage("대항력?", id="h1"),
        AIMessage("다음 날 0시.", name="knowledge_agent", id="k1"),
        AIMessage("다음 날 0시.", name="supervisor", id="s1"),
    ]
    assert needs_report({"messages": msgs}) == "preserve"


def test_needs_report_ignores_previous_turns():
    msgs = _risk_only_flow() + [
        HumanMessage("고마워", id="h2"),
        AIMessage("네.", name="supervisor", id="s3"),
    ]
    assert needs_report({"messages": msgs}) == "preserve"


def test_no_report_forced_when_risk_tool_returned_input_error():
    msgs = _risk_only_flow()
    msgs[4] = ToolMessage(
        "입력 오류: market_price: 0보다 커야 합니다.",
        tool_call_id="c2",
        name="assess_jeonse_risk",
        id="t2",
    )
    msgs[-1] = AIMessage("매매 시세를 알려주시면 진단해 드릴게요.", name="supervisor", id="s2")
    assert needs_report({"messages": msgs}) == "preserve"


def test_knowledge_answer_not_forced_when_multiple_workers_answered():
    msgs = [
        HumanMessage("강남구 까치마을 시세랑 보증보험 조건 알려줘", id="h1"),
        AIMessage("까치마을 신규 전세 중위값 4.8억, 3건.", name="market_agent", id="m1"),
        AIMessage("HUG 보증은 전세가율 90% 이하.\n근거: HUG", name="knowledge_agent", id="k1"),
        AIMessage(
            "시세는 4.8억(신규 3건)이고, HUG 보증은 전세가율 90% 이하여야 합니다.\n근거: HUG",
            name="supervisor",
            id="s1",
        ),
    ]
    assert preserve_worker_answer({"messages": msgs}) == {}


def test_replacement_keeps_supervisor_metadata():
    msgs = _flow("요약")
    msgs[-1] = AIMessage(
        "요약",
        name="supervisor",
        id="s2",
        usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )
    [rep] = preserve_worker_answer({"messages": msgs})["messages"]
    assert rep.usage_metadata["total_tokens"] == 12 and rep.content == REPORT


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
