# ADR-0006: uv 패키지 관리 / Python 3.12 / Streamlit UI

- 상태: 채택
- 날짜: 2026-09-02

## 상황

이직용 포트폴리오다. 평가자가 레포를 클론해서 **명령 몇 줄로 같은 결과를 재현할 수 있어야** 하고, 코드가 보여줘야 할 것은 멀티에이전트·RAG 설계이지 인프라 배관이 아니다. 동시에 langchain 1.3.x / langgraph 1.2.x / chromadb 1.5.x / ragas 0.4.3처럼 서로 의존 범위가 겹치는 패키지들을 흔들리지 않게 잠가야 한다(ADR-0005의 핀이 그 예다).

## 결정

- **패키지·가상환경 관리: `uv`** — `pyproject.toml` + `uv.lock`, 개발 의존성은 `[dependency-groups] dev`.
- **Python 3.12** — `requires-python = ">=3.12,<3.13"` ([`pyproject.toml`](../../pyproject.toml)), [`.python-version`](../../.python-version)에 `3.12`, ruff `target-version = "py312"`.
- **UI: Streamlit** — `src/rent_agent/app/streamlit_app.py`.

실행 경로는 세 줄이다.

```bash
uv sync
uv run python scripts/ingest.py
uv run streamlit run src/rent_agent/app/streamlit_app.py
```

## 근거

### uv

- **lock 재현성**: `uv.lock`이 전체 의존성 트리를 해석된 버전으로 고정한다. ADR-0005의 `langchain-community==0.3.31`처럼 미묘한 조합이 성립해야 하는 프로젝트에서, "내 머신에서는 됐다"를 없애는 것이 필수다.
- **속도**: 의존성 트리가 크다(langchain 계열 5개 + langgraph 3개 + chromadb + ragas + streamlit). 재설치·재해석이 빨라야 실험 반복이 가능하다. RAGAS를 5회 돌리며 핀을 조정한 작업이 이 속도 위에서 이루어졌다.
- **`uv run`이 실행 경로를 통일한다**: 가상환경 활성화 단계가 없어진다. README·CI·평가 스크립트가 모두 `uv run ...` 한 형태를 쓰므로 문서와 실제 실행이 어긋나지 않는다.
- **표준 준수**: 별도 포맷이 아니라 PEP 621 `pyproject.toml`을 쓴다. uv를 버려도 `pip install -e .`로 돌아갈 수 있다.

### Python 3.12

- **라이브러리 호환 최광범위**: chromadb는 C 확장(및 native 빌드 의존)을 포함하고, `langchain`/`ragas` 생태계도 3.12에 대한 휠이 안정적으로 제공된다. 3.13은 일부 C 확장 패키지의 사전 빌드 휠이 아직 갖춰지지 않아 소스 빌드로 떨어질 위험이 있다.
- **상한을 명시(`<3.13`)한 이유**: 상한이 없으면 3.13이 설치된 머신에서 `uv sync`가 3.13을 골라 위 위험에 노출된다. `.python-version`으로 인터프리터까지 고정해 클론 즉시 같은 버전이 잡히게 했다.
- **3.11 이하로 내리지 않은 이유**: `StrEnum`(3.11+)과 `X | None` 타입 표기를 쓴다. `domain/models.py`의 `Region`/`RiskLevel`이 `StrEnum` 기반이다.

### Streamlit

- **파이썬 단일 스택**: 프런트엔드 빌드 체인(Node, 번들러, 타입 정의)이 없다. 데모 화면 하나를 위해 두 번째 언어·두 번째 의존성 트리를 도입하지 않는다.
- **데모 속도**: 스트리밍 출력, 채팅 UI, 폼 입력이 기본 제공된다. 이 서비스가 필요한 UI는 "채팅 + 매물 정보 입력 폼 + 판정 리포트 표시"가 전부다.
- **LangGraph와 자연스럽게 맞물린다**: 그래프 스트림을 그대로 화면에 흘리고, `output_mode="full_history"`(ADR-0003)로 얻은 에이전트 호출 흐름을 그대로 보여줄 수 있다.
- **포트폴리오 초점 유지**: 평가자가 봐야 하는 것은 `agents/`·`domain/`·`rag/`다. UI 코드가 저장소에서 차지하는 비중이 작을수록 좋다.

## 검토한 대안

| 대안 | 장점 | 배제 이유 |
|---|---|---|
| **pip + requirements.txt** | 가장 익숙, 추가 도구 없음 | 전이 의존성 lock이 없다(`pip freeze`는 플랫폼·설치 순서에 의존). ADR-0005 같은 조합을 재현 보장할 수 없다. |
| **Poetry** | 성숙한 lock, 넓은 사용층 | 의존성 해석이 느려 반복 실험 비용이 크다. uv가 같은 표준(`pyproject.toml`)을 쓰면서 더 빠르다. |
| **conda** | 네이티브 의존성 처리 강점 | 환경 파일과 채널 관리가 추가된다. 순수 파이썬 휠로 해결되는 이 프로젝트에 과하다. |
| **Python 3.13** | 최신 성능 개선 | 일부 C 확장 사전 빌드 휠 미지원 — chromadb 등에서 소스 빌드/실패 위험. 이 프로젝트가 3.13에서 얻는 이득은 없다. |
| **FastAPI + React** | 실서비스형 구조, 프런트 자유도 | 포트폴리오 범위 대비 과하다. 백엔드 라우트·CORS·상태 관리·빌드 파이프라인이 추가되지만 보여줄 핵심(에이전트 설계)은 하나도 늘지 않는다. |
| **Gradio** | Streamlit과 유사한 장점, ML 데모 표준 | 멀티 위젯 폼(보증금·시세·근저당·지역·자산 등 9개 필드)의 레이아웃 제어가 Streamlit보다 불편하다. 채팅과 폼을 한 화면에 두는 요구에는 Streamlit이 맞다. |

## 결과/트레이드오프

- **얻은 것**: `uv sync` 한 줄 재현, 인터프리터·의존성 완전 고정, 프런트엔드 스택 0.
- **잃은 것 — uv**: 평가자가 uv를 설치해야 한다(단일 바이너리). `pyproject.toml`이 표준이므로 pip 폴백이 가능하다는 점이 완화책이다.
- **잃은 것 — Python 3.12 상한**: 3.13 전용 기능을 쓸 수 없고, 3.13에서만 제공되는 휠이 생기면 상한을 올려야 한다. 재검토 트리거는 "chromadb·ragas가 3.13 휠을 제공"이다.
- **잃은 것 — Streamlit**: 스크립트 재실행 모델이라 상태를 `st.session_state`와 LangGraph 체크포인터로 관리해야 하고(ADR-0003의 외부 그래프 체크포인터가 이를 받는다), 동시 사용자 규모 확장이나 세밀한 UI 커스터마이즈에는 맞지 않는다. 실서비스로 갈 때는 FastAPI 백엔드로 그래프를 분리하고 UI를 교체하는 것이 경로다 — `agents/supervisor.py`의 `build_graph(settings, checkpointer)`가 이미 UI와 분리된 진입점이므로 그래프 코드는 그대로 재사용된다.
- **품질 게이트**: ruff(`line-length = 100`, `select = ["E","F","I","B","UP"]`) + pytest를 GitHub Actions(`.github/workflows/ci.yml`, 별도 태스크에서 추가 예정 — 2026-09-02 기준 미작성)에서 돌린다. 로컬 게이트는 `uv run ruff check .` / `uv run pytest`다. 외부 API가 필요한 테스트는 `@integration` 마커로 CI에서 제외한다(`addopts = "-m 'not integration'"`).
