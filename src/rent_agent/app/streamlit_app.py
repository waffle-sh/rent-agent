"""Streamlit UI: (1) 지식 Q&A 채팅, (2) 전세 진단 폼. 둘 다 같은 Supervisor 그래프를 호출한다."""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from rent_agent.agents import supervisor
from rent_agent.agents.llm import configure_tracing
from rent_agent.config import get_settings

st.set_page_config(
    page_title="rent-agent · 전세 리스크 상담", page_icon="🏠", layout="wide"
)

# UI 라벨 → market_agent 도구의 housing_type 값
HOUSING_LABELS = {
    "아파트": "apartment",
    "연립·다세대(빌라)": "multi_house",
    "오피스텔": "officetel",
}


@st.cache_resource
def _graph():
    settings = get_settings()
    configure_tracing(settings)  # .env의 LangSmith 설정을 프로세스 환경으로
    # `supervisor.build_graph`(모듈 속성)로 호출한다. Streamlit은 이 파일을 모듈이 아닌
    # 스크립트로 exec하므로 테스트가 이 파일의 전역을 바꿔 끼울 수 없다.
    # 모듈 속성으로 호출하면 호출 시점에 해석되어 monkeypatch가 통한다 (tests/app 참고).
    return supervisor.build_graph(settings, checkpointer=InMemorySaver())


def _run(prompt: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return _graph().invoke({"messages": [HumanMessage(prompt)]}, config=config)


def _trace(result) -> list[str]:
    """에이전트 호출 순서를 사람이 읽는 형태로. 포트폴리오 데모용 관측 정보."""
    lines: list[str] = []
    for m in result["messages"]:
        name = getattr(m, "name", None)
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"🧭 {name or 'supervisor'} → {tc['name']}")
        elif isinstance(m, ToolMessage) and name and not name.startswith("transfer_"):
            lines.append(f"🔧 {name} 결과 수신")
    return lines


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []

st.title("🏠 rent-agent — 사회초년생 전세 상담")
st.caption(
    "법령·제도 질문과 전세 매물 위험 진단을 멀티에이전트가 처리합니다. "
    "참고 정보이며 법률·금융 자문이 아닙니다."
)

tab_chat, tab_diag = st.tabs(["💬 지식 Q&A", "🔎 전세 진단"])

with tab_chat:
    for role, text in st.session_state.chat:
        st.chat_message(role).markdown(text)
    if q := st.chat_input("예: 전입신고하면 대항력은 언제부터 생기나요?"):
        st.session_state.chat.append(("user", q))
        st.chat_message("user").markdown(q)
        with st.chat_message("assistant"), st.spinner("에이전트가 답변을 준비 중..."):
            result = _run(q)
            answer = result["messages"][-1].content
            st.markdown(answer)
            with st.expander("에이전트 실행 흐름"):
                st.write("\n".join(_trace(result)) or "(추적 정보 없음)")
        st.session_state.chat.append(("assistant", answer))

with tab_diag:
    with st.form("diag"):
        c0, c1, c2, c3 = st.columns(4)
        housing_label = c0.selectbox("주거 유형", list(HOUSING_LABELS))
        region_text = c1.text_input("지역 (구/시)", "강남구")
        apt = c2.text_input("건물/단지명", "")
        area = c3.number_input("전용면적 (㎡)", min_value=0.0, value=0.0, step=0.1)
        c4, c5, c6 = st.columns(3)
        deposit = c4.number_input("전세 보증금 (만원)", min_value=0, value=30000, step=500)
        price = c5.number_input("매매 시세 (만원)", min_value=0, value=50000, step=500)
        liens = c6.number_input(
            "선순위 근저당 채권최고액 (만원)", min_value=0, value=0, step=500
        )
        c7, c8, c9 = st.columns(3)
        senior_dep = c7.number_input(
            "선순위 임차보증금 (만원, 다가구)", min_value=0, value=0, step=500
        )
        capital = c8.number_input("자기자금 (만원)", min_value=0, value=0, step=500)
        income = c9.number_input("연소득 (만원, 선택)", min_value=0, value=0, step=100)
        rate = st.slider("전세대출 예상 금리 (%)", 0.0, 10.0, 3.5, 0.1)
        submitted = st.form_submit_button("진단하기", type="primary")

    if submitted:
        if deposit == 0 or price == 0:
            st.error("보증금과 매매 시세는 필수입니다.")
        else:
            parts = [
                f"{region_text} {apt} {f'{area}㎡' if area else ''} {housing_label} "
                f"전세 계약을 검토 중입니다. "
                f"주거 유형 코드는 {HOUSING_LABELS[housing_label]}입니다."
            ]
            parts.append(
                f"보증금 {deposit}만원, 매매 시세 {price}만원, "
                f"선순위 근저당 채권최고액 {liens}만원, "
                f"선순위 임차보증금 {senior_dep}만원, 자기자금 {capital}만원, "
                f"{'연소득 ' + str(income) + '만원, ' if income else ''}예상 금리 {rate}%."
            )
            if apt:
                parts.append("같은 건물(단지)의 최근 전세 시세와도 비교해 주세요.")
            parts.append("이 계약이 적절한지 판단하고 리포트를 작성해 주세요.")
            with st.spinner("시세 조회 · 위험 계산 · 리포트 작성 중..."):
                result = _run(" ".join(parts))
            st.markdown(result["messages"][-1].content)
            with st.expander("에이전트 실행 흐름"):
                st.write("\n".join(_trace(result)) or "(추적 정보 없음)")
