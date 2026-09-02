# ADR-0003: 멀티에이전트 구조로 LangGraph Supervisor 패턴 + 결정적 후처리 노드

- 상태: 채택
- 날짜: 2026-09-02

## 상황

서비스가 처리해야 하는 요청은 성격이 뚜렷하게 다른 네 갈래다.

1. **지식 QA** — "대항력은 언제 생기나요?" → Chroma RAG 검색 후 조문 인용
2. **시세 조회** — 국토부 아파트/연립다세대/오피스텔 전월세 실거래가 API 호출 후 통계
3. **위험 판단** — 보증금·시세·근저당·자산을 받아 전세가율/부담률/경매 회수액 계산
4. **리포트** — 위 결과를 종합해 헤더·수치·면책 문구를 갖춘 최종 답 작성

이걸 하나의 ReAct 에이전트에 몰아넣으면 시스템 프롬프트가 네 역할의 지시를 모두 담아야 하고, 도구 목록도 RAG + 3개 API + 위험 계산으로 커진다. 라우팅 품질과 개별 역할의 테스트 가능성이 함께 떨어진다.

## 결정

**`langgraph-supervisor`의 `create_supervisor` 패턴 + 4개 워커 에이전트**를 쓴다. 그리고 컴파일된 supervisor 팀을 **외부 `StateGraph`의 서브그래프 노드로 감싸고, 두 개의 결정적 후처리 노드를 붙인다** ([`agents/supervisor.py`](../../src/rent_agent/agents/supervisor.py)).

```
START → team(supervisor + 4 워커) → [needs_report?] → report → preserve_worker_answer → END
                                          └────── preserve ──────┘
```

핵심 결정 세 가지:

- **위험 판단은 LLM이 아니라 순수 Python 함수**다 ([`domain/risk.py`](../../src/rent_agent/domain/risk.py)). LLM은 계산 결과를 설명만 한다.
- **`output_mode="full_history"`**.
- **결과의 완결성은 프롬프트가 아니라 그래프가 보장한다** — `ensure_report`(`needs_report` 조건부 엣지 + `report` 노드), `preserve_worker_answer`.

## 근거

### 역할 분리

- **프롬프트 단순화**: 워커마다 자기 역할의 지시만 갖는다. 모든 시스템 프롬프트는 [`agents/prompts.py`](../../src/rent_agent/agents/prompts.py) 한 곳에 모아 diff로 프롬프트 변경 이력을 추적한다. RAGAS 2·3차 개선이 프롬프트 한 파일 수정으로 끝난 것이 이 구조의 이득이다(ADR-0002 지표 표).
- **개별 테스트**: 워커의 도구는 각각 유닛 테스트된다(`tests/agents/test_risk_tool.py`, `test_market_tool.py`). 그래프 전체를 도는 테스트는 `@integration`으로 분리한다.
- **트레이스 가독성**: LangSmith에서 "supervisor → risk_agent → report_agent" 흐름이 그대로 보인다. 단일 에이전트의 긴 도구 호출 열보다 디버깅이 쉽다.
- **핸드오프 도구 자동 생성**: `create_supervisor`가 워커별 `transfer_to_*` 도구를 만들어 준다. 직접 라우팅 엣지를 쓰면 워커 추가마다 그래프를 수정해야 한다.

### 위험 판단을 순수 함수로 둔 이유

LLM에 계산을 맡기면 같은 입력이 같은 판정을 낸다는 보장이 없다. 전세 위험도는 사용자가 **금전적 결정에 쓰는 수치**이므로 재현성이 요구 조건이다. `domain/`은 LLM·HTTP 의존이 전혀 없어(계획의 책임 분리 원칙) 위험 로직 전체가 외부 호출 없이 유닛 테스트된다. 판정 기준과 그 출처는 ADR-0004에 있다.

### `output_mode="full_history"`

`last_message` 모드는 워커의 마지막 메시지만 supervisor에 되돌린다. 그러면 report_agent가 **도구 결과의 원본 수치**(전세가율, 예상 회수액, 부족액)를 보지 못하고 요약문만 보게 되어, 리포트가 수치를 재진술하다 틀릴 여지가 생긴다. `full_history`는 도구 메시지를 포함한 전체 이력을 넘겨 리포트가 원본을 인용하게 한다. 부수 효과로 UI에서 에이전트 호출 흐름을 그대로 보여줄 수 있다.

### 두 개의 결정적 후처리 노드 — 2026-09-02 통합 테스트 실측

**① `ensure_report`: supervisor가 report_agent를 건너뛰었다 (2회 중 1회)**

- **관측**: risk_agent의 결과를 받은 뒤 supervisor가 report_agent에 위임하지 않고 자기가 직접 답한 사례. 헤더·면책 문구가 없는 짧은 답이 나왔다.
- **조치**: `needs_report` 조건부 엣지 — 이번 턴에 `assess_jeonse_risk` 도구가 **유효한** 결과를 냈는데 report_agent의 답이 없으면 그래프가 `report` 노드를 직접 실행한다.
- **예외 처리**: 도구가 `"입력 오류"` 접두어를 돌려준 경우(입력 추출 실패 → supervisor가 되묻는 것이 맞음)는 리포트를 강제하지 않는다.

**② `preserve_worker_answer`: supervisor가 리포트를 재작성해 형식을 유실했다 (3회 중 1회)**

- **관측**: report/knowledge 에이전트의 답을 supervisor가 자기 말로 다시 써서 `## 종합 판정` 헤더와 면책 문구가 사라진 사례.
- **먼저 시도한 것**: `langgraph-supervisor`가 제공하는 `forward_message` 도구. **모델이 이 도구를 호출하지 않았다**(실측). 도구가 있어도 호출은 모델 판단이라 확률적이다.
- **조치**: 마지막 사용자 턴 이후 워커의 최종 답이 있으면 supervisor의 마지막 메시지 `content`를 **원문으로 교체**한다. 같은 메시지 id로 돌려주어 `add_messages` 리듀서가 교체 처리하고, `usage_metadata` 등은 유지한다. `response_metadata`에 `forwarded_from`을 남겨 무엇이 교체됐는지 추적한다.
- **교체 범위 제한**: report_agent 답은 항상 원문 우선(종합 리포트가 곧 최종 답). knowledge_agent 답은 **이 턴에 답한 워커가 그것 하나일 때만** 교체한다 — 시세+지식처럼 여러 워커가 답한 턴에서는 supervisor의 종합이 정당하고, 지식 답만으로 덮으면 시세 결과가 사라진다.
- **멀티턴 안전장치**: 두 노드 모두 `_current_turn()`으로 마지막 `HumanMessage` 이후 구간만 본다. 이전 턴의 리포트를 이번 턴 답으로 끌어오지 않는다. 핸드오프용 `tool_calls`가 달린 `AIMessage`는 워커의 "답"이 아니므로 `_last_worker_answer()`에서 제외한다.

**원칙**: 프롬프트 지시는 확률적이다. 결과의 **완결성·충실성은 LLM 판단에 맡기지 않고 그래프가 보장**한다.

### 외부 그래프 합성

컴파일된 supervisor 팀을 외부 `StateGraph(MessagesState)`의 `team` 노드로 넣었다.

- **체크포인터는 외부 그래프에만 붙인다** (`outer.compile(checkpointer=...)`). 내부 팀은 체크포인터 없이 컴파일해 상태 저장 지점을 하나로 유지한다.
- 내부 에이전트의 `remaining_steps` 필드는 `MessagesState`에 노출되지 않음을 확인했다 — 외부 상태 스키마를 `MessagesState`로 두어도 충돌하지 않는다.
- **의존성 리스크 대비**: `langgraph-supervisor` 0.0.31은 내부에서 deprecated `create_react_agent`를 사용한다. 0.0.x 버전대이므로 향후 호환 파손 가능성이 있다. 외부 그래프 설계 덕분에 `team` 노드만 직접 작성한 라우터로 교체할 수 있다 — 후처리 노드와 그래프 배선은 그대로 남는다.

## 검토한 대안

| 대안 | 장점 | 배제 이유 |
|---|---|---|
| **단일 ReAct 에이전트** | 구조 단순, 호출 수 최소 | 도구 5종 이상 + 네 역할 지시를 한 프롬프트에 담아야 한다. 도구가 늘수록 라우팅 품질이 떨어지고 역할별 테스트가 불가능하다. |
| **Swarm (피어 간 핸드오프)** | 중앙 병목 없음, 유연 | 에이전트끼리 서로 넘기므로 흐름 예측이 어렵다. "위험 판단 후 반드시 리포트"처럼 **보장해야 하는 순서**가 있는 이 서비스와 맞지 않는다. |
| **프롬프트 강화로 ①② 해결** | 코드 추가 없음 | 확률적이다. ①은 2회 중 1회, ②는 3회 중 1회 실패했고 프롬프트는 이 실패 모드를 없애지 못한다. |
| **`forward_message` 도구로 ② 해결** | 라이브러리 제공 기능 | **모델이 호출하지 않았다**(실측). 호출 여부가 다시 모델 판단이라 문제가 한 단계 미뤄질 뿐이다. |
| **위험 판단도 LLM에게** | 코드 감소, 유연한 입력 처리 | 재현성 상실. 금전 판단 수치를 확률적으로 생성하는 것은 허용할 수 없다. |

## 결과/트레이드오프

- **얻은 것**: 역할별 프롬프트·테스트 분리, 읽히는 트레이스, 그리고 **형식·완결성의 결정적 보장**(프롬프트 재현율에 의존하지 않음).
- **잃은 것 — 지연과 비용**: supervisor 라우팅 호출이 매 위임마다 추가되고, `full_history`가 대화 전체를 워커에 전달하므로 토큰이 늘어난다. 단일 에이전트 대비 호출 수·지연·비용이 모두 상승한다. `gpt-4.1-mini` 선택(ADR-0001)이 이 비용을 상쇄한다.
- **추가 복잡도**: 후처리 노드 두 개는 메시지 이름(`AIMessage.name`)에 의존한다. `make_report_node`는 `create_agent(name=...)`가 붙이는 이름을 방어적으로 한 번 더 보장한다. 워커를 추가할 때 `VERBATIM_AGENTS`를 갱신할지 판단해야 한다.
- **버전 고정 리스크**: `langgraph-supervisor>=0.0.31,<0.1`로 상한을 걸어 두었다([`pyproject.toml`](../../pyproject.toml)). 0.1 진입 시 `team` 노드 교체를 검토한다.
