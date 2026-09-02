# ADR-0005: RAGAS 호환을 위한 `langchain-community==0.3.31` 고정

- 상태: 채택
- 날짜: 2026-09-02

## 상황

RAG 품질을 주장하려면 측정이 있어야 한다(프로젝트 규칙: 모든 결정에 설명 가능한 근거). 공인 지표를 쓰기 위해 **RAGAS 0.4.3**을 평가 의존성으로 넣었다 — `faithfulness`(환각), `answer_relevancy`, `context_precision`(검색).

그런데 `ragas==0.4.3`을 설치하고 import하면 다음이 터진다.

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

`ragas`가 `langchain_community.chat_models.vertexai`를 **하드 import**하는데, 이 모듈은 `langchain-community` **0.4.x에서 제거**되었다. 본 프로젝트는 langchain 1.3.x 계열을 쓰므로 의존성 해석이 `langchain-community` 0.4.x를 끌어오면서 충돌한다. Vertex AI는 이 프로젝트에서 전혀 쓰지 않는데도 import 시점에 실패한다.

## 결정

**`langchain-community==0.3.31`로 정확히 고정한다** ([`pyproject.toml`](../../pyproject.toml)).

```toml
# ragas 0.4.3이 langchain_community.chat_models.vertexai를 import함 → 0.4.x와 충돌. ADR-0005
"langchain-community==0.3.31",
```

`langchain 1.3.18`과 함께 정상 동작함을 **2026-09-02 실측 확인**했다(설치 검증: `langchain 1.3.18` / `langchain-core 1.6.1` / `langchain-community 0.3.31` / `ragas 0.4.3`, 5회의 RAGAS 평가 실행 완료).

## 근거

- **가장 좁은 조치**: 문제의 원인은 단 하나의 제거된 모듈이다. `langchain` 본체를 내리거나 ragas를 버리는 것보다, 그 모듈이 아직 존재하는 마지막 마이너 버전으로 community만 고정하는 것이 영향 범위가 가장 작다.
- **런타임 영향 없음**: 프로젝트 코드는 `langchain-community`를 **직접 import하지 않는다.** 실제 사용하는 것은 `langchain-core`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`, `langgraph`이며 이들은 모두 최신 계열이다. community는 ragas가 import 통과하기 위해서만 필요하다.
- **정확 고정(`==`)의 이유**: 범위(`>=0.3.31,<0.4`)로 두면 0.3.x 안에서 다시 움직일 수 있고, 이 핀의 목적은 "ragas가 import에 성공하는, 검증된 정확한 조합"을 잠그는 것이다. `uv.lock`과 함께 재현성을 보장한다.
- **핀 해제 조건이 명확하다**: ragas 상위 버전이 vertexai 하드 import를 제거하면(지연 import 또는 optional extra로 전환) 핀을 풀고 community를 최신으로 올린다. 그때까지 `pyproject.toml` 주석이 이 ADR을 가리킨다.

### 부수 발견: RAGAS 판정자 `max_tokens=4096`

같은 평가 셋업에서 두 번째 문제가 실측되었다. RAGAS 기본 `max_tokens=1024`로는 **한국어 법령 청크에 대한 NLI 판정 JSON이 잘려** `IncompleteOutputException`이 발생한다. 한국어 법령 문장은 같은 의미당 토큰 수가 많고, `faithfulness`는 답을 진술 단위로 쪼개 각각을 컨텍스트와 대조한 결과를 JSON으로 출력하므로 출력이 길다.

→ 판정자 LLM을 `max_tokens=4096`으로 만든다 ([`scripts/eval_rag.py`](../../scripts/eval_rag.py)).

```python
# 기본 max_tokens=1024는 한국어 법령 청크의 NLI 판정 JSON이 잘려
# IncompleteOutputException (실측)
judge = llm_factory(settings.openai_model, client=client, max_tokens=4096)
```

이 값이 없으면 평가가 완주하지 못한다 — ADR-0002의 1→6차 지표 표는 이 설정 위에서 얻은 것이다.

## 검토한 대안

| 대안 | 장점 | 배제 이유 |
|---|---|---|
| **ragas를 쓰지 않고 자체 LLM-judge 구현** | 의존성 충돌 없음, 프롬프트를 완전히 통제 → 재현성 상승 | 공인 지표라는 신뢰를 잃는다. 자체 판정 기준으로 낸 0.9는 포트폴리오에서 근거로 쓰기 어렵다. 또한 `faithfulness`/`context_precision`의 검증된 프롬프트를 직접 재현하는 비용이 크다. |
| **평가 전용 별도 venv** | 운영 의존성을 오염시키지 않음 | 운영 복잡도 상승 — `uv sync` 한 번으로 끝나는 재현성을 잃고, 두 lock 파일과 두 실행 경로를 관리해야 한다. CI 설정도 갈라진다. 얻는 것은 "직접 쓰지도 않는 패키지의 버전이 낮다"는 것뿐이다. |
| **`langchain-community` 자체를 제거하고 ragas를 monkeypatch** | 최신 community 사용 가능 | 라이브러리 내부 import를 런타임에 속이는 방식은 ragas 버전이 바뀔 때마다 깨진다. 핀 한 줄보다 유지 비용이 크다. |
| **langchain을 0.3.x 계열로 내린다** | 충돌 자연 해소 | langgraph 1.2.x / `langgraph-supervisor`가 요구하는 최신 계열을 포기해야 한다. 평가 도구를 위해 본체를 내리는 것은 우선순위가 뒤집힌 결정이다. |

## 결과/트레이드오프

- **얻은 것**: `uv sync` 한 번으로 운영과 평가가 모두 되는 단일 환경. RAGAS 6회 실행으로 청킹·k·프롬프트·평가 컨텍스트 개선을 실측 근거와 함께 진행할 수 있었다(ADR-0002).
- **잃은 것**: `langchain-community`가 구버전에 묶인다. 향후 community의 어떤 통합을 직접 쓰려 하면 이 핀이 걸림돌이 된다.
- **위험 관리**: 프로젝트 코드가 community를 직접 import하지 않으므로 현재 위험은 사실상 0이다. 이 성질이 유지되는지는 `grep -r langchain_community src/`로 확인할 수 있다 — 히트가 생기는 순간 이 ADR을 재검토해야 한다.
- **핀 해제**: ragas가 vertexai 하드 import를 제거한 버전을 내면 `pyproject.toml`의 `==0.3.31`을 풀고 `uv lock --upgrade`로 검증한다.
