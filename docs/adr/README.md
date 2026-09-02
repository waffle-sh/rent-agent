# 설계 결정 기록 (ADR)

기술·아키텍처 결정마다 상황/결정/근거/대안/결과를 기록한다. 프로젝트 규칙("기술·아키텍처·구현 방식 결정에 모두 설명 가능한 근거가 있어야 함")을 만족시키는 문서 집합이다. 수치는 모두 2026-09-02 실측이며, 실측이 초기 결정을 뒤집은 경우 원래 선택·관측·변경을 함께 남긴다.

| 번호 | 제목 | 상태 |
|---|---|---|
| [0001](0001-llm-openai.md) | LLM/임베딩 모델로 OpenAI 채택 | 채택 |
| [0002](0002-vector-store-chroma.md) | 벡터 스토어 Chroma(로컬 persist)와 검색 전략 | 채택 |
| [0003](0003-multi-agent-supervisor.md) | 멀티에이전트 구조로 LangGraph Supervisor 패턴 + 결정적 후처리 노드 | 채택 |
| [0004](0004-jeonse-risk-rules.md) | 전세 위험 판단 규칙 (LLM 비의존 순수 함수) | 채택 |
| [0005](0005-ragas-langchain-community-pin.md) | RAGAS 호환을 위한 `langchain-community==0.3.31` 고정 | 채택 |
| [0006](0006-uv-python312-streamlit.md) | uv 패키지 관리 / Python 3.12 / Streamlit UI | 채택 |

공통 템플릿: `## 상황` → `## 결정` → `## 근거` → `## 검토한 대안` → `## 결과/트레이드오프`.
