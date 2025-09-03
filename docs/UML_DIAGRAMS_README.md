# NEXTEP.AI UML 다이어그램 문서

본 문서는 NEXTEP.AI 커리어 예측 및 조언 시스템의 UML 다이어그램들을 포함하고 있습니다.

## 📋 다이어그램 목록

### 1. Use Case Diagram
**파일명**: `usecase_diagram.puml`
**설명**: 시스템의 주요 기능과 사용자(비회원, 회원, 관리자) 간의 관계를 보여주는 다이어그램

**주요 기능들**:
- 인증 및 사용자 관리 (회원가입, 로그인, 프로필 관리)
- 커리어 예측 (개인정보 입력, 예측 실행, 시나리오 비교)
- AI 조언 시스템 (개인화 조언, 채팅 상담, 스트리밍 응답)
- 정보 제공 (노동시장 동향, 학습 추천)
- 시스템 관리 (ML 모델 관리, RAG 데이터 관리)

### 2. Sequence Diagrams

#### 2.1 커리어 예측 플로우
**파일명**: `sequence_prediction_flow.puml`
**설명**: 사용자의 예측 요청부터 결과 표시까지의 전체 프로세스

**주요 플로우**:
1. 예측 페이지 접근 (회원/비회원 분기)
2. 예측 실행 (XGBoost 모델 사용)
3. 결과 시각화 및 표시
4. 동적 직업군 변경 (AJAX)

#### 2.2 AI 조언 및 채팅 시스템
**파일명**: `sequence_ai_advice_flow.puml`
**설명**: AI 조언 생성부터 실시간 채팅까지의 복합적인 AI 서비스 플로우

**주요 플로우**:
1. 초기 AI 조언 생성 (RAG + LLM)
2. 사용자 추가 질문 처리
3. 스트리밍 응답 (실시간)
4. 대화기록 초기화

#### 2.3 사용자 인증 시스템
**파일명**: `sequence_authentication_flow.puml`
**설명**: 회원가입, 로그인, 프로필 관리 등 인증 관련 전체 플로우

**주요 플로우**:
1. 회원가입 (검증, 암호화, DB 저장)
2. 로그인 (인증, 세션 생성)
3. 로그아웃 (세션 해제)
4. 프로필 관리 (업데이트)

#### 2.4 RAG 시스템 동작
**파일명**: `sequence_rag_system_flow.puml`
**설명**: RAG(Retrieval-Augmented Generation) 시스템의 초기화부터 질의 처리까지

**주요 플로우**:
1. RAG 시스템 초기화
2. PDF 문서 처리 및 임베딩
3. 기존 데이터 마이그레이션
4. 사용자 질의 처리 (검색 + 생성)

### 3. 시스템 아키텍처 개요
**파일명**: `system_architecture_overview.puml`
**설명**: 전체 시스템의 계층별 구조와 컴포넌트 간 관계

**주요 계층**:
- 프론트엔드 레이어 (Browser, JavaScript, Charts)
- 백엔드 애플리케이션 레이어 (Flask, Routes)
- 비즈니스 로직 레이어 (Services, Utils)
- AI/ML 서비스 레이어 (LLM, RAG, ML)
- 데이터 레이어 (DB, Cache, Session)
- 외부 서비스 (Ollama, XGBoost, PDF 문서)

## 🛠 사용 방법

### PlantUML로 다이어그램 생성하기

1. **PlantUML 설치**
   ```bash
   # Java 필요
   wget http://sourceforge.net/projects/plantuml/files/plantuml.jar/download
   ```

2. **다이어그램 생성**
   ```bash
   # PNG 형식으로 생성
   java -jar plantuml.jar usecase_diagram.puml
   java -jar plantuml.jar sequence_*.puml
   java -jar plantuml.jar system_architecture_overview.puml
   
   # SVG 형식으로 생성
   java -jar plantuml.jar -tsvg usecase_diagram.puml
   ```

3. **온라인 도구 사용**
   - [PlantUML Online](http://www.plantuml.com/plantuml/uml/)
   - 파일 내용을 복사해서 온라인에서 바로 확인 가능

### VS Code에서 사용하기

1. **PlantUML 확장 설치**
   - 확장: "PlantUML" by jebbs

2. **미리보기**
   - `.puml` 파일 열기
   - `Ctrl+Shift+P` → "PlantUML: Preview Current Diagram"

## 🎯 다이어그램 특징

### Use Case Diagram
- **액터 구분**: 비회원, 회원, 관리자별 색상 구분
- **패키지 구조**: 기능별 그룹화로 가독성 향상
- **관계 표현**: include, extend 관계로 기능 의존성 표현

### Sequence Diagrams
- **상세한 플로우**: 실제 코드 구조를 반영한 정확한 시퀀스
- **조건부 처리**: alt/else로 다양한 시나리오 표현
- **루프 처리**: 반복적인 작업(스트리밍, 데이터 처리) 표현
- **에러 처리**: 예외 상황과 대응 방법 포함

### 시스템 아키텍처
- **계층형 구조**: 명확한 레이어 분리
- **의존성 표현**: 컴포넌트 간 관계와 데이터 흐름
- **색상 코딩**: 계층별 시각적 구분

## 📊 주요 기술 스택 (다이어그램 반영)

### 백엔드
- **Flask**: 웹 애플리케이션 프레임워크
- **SQLAlchemy**: ORM (Oracle DB 연결)
- **LangChain**: AI/LLM 통합 프레임워크

### AI/ML
- **Ollama**: LLM 서비스 (exaone3.5:7.8b)
- **XGBoost**: 예측 모델 (소득/만족도)
- **ChromaDB**: 벡터 데이터베이스 (RAG)

### 프론트엔드
- **JavaScript**: 동적 UI 처리
- **Chart.js**: 데이터 시각화
- **Bootstrap**: UI 프레임워크

## 📈 업데이트 이력

- **2024-09-03**: 초기 UML 다이어그램 세트 작성
  - Use Case Diagram 완성
  - 4개 주요 Sequence Diagram 완성
  - 시스템 아키텍처 개요 완성

## 🔍 참고사항

1. **실제 코드 반영**: 모든 다이어그램은 현재 코드베이스를 분석하여 작성됨
2. **기술 정확성**: 실제 사용되는 기술 스택과 라이브러리를 정확히 반영
3. **플로우 검증**: 각 시퀀스는 실제 사용자 시나리오를 기반으로 검증됨
4. **확장 가능성**: 향후 기능 추가 시 다이어그램 업데이트 용이

---

이 다이어그램들은 NEXTEP.AI 시스템의 전체적인 구조와 동작 방식을 이해하는 데 도움이 됩니다. 
시스템 개발, 유지보수, 그리고 새로운 팀원의 온보딩 시 참고 자료로 활용하세요.