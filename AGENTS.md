# AGENTS.md

> 이 파일은 AI 코딩 에이전트(Claude Code, Cursor, Copilot 등)가 프로젝트를 이해하고 일관된 방식으로 작업할 수 있도록 컨텍스트와 규칙을 제공합니다.
> 표준 파일명은 `AGENTS.md`(대문자), 레포지토리 루트에 위치시킵니다.

---

## 1. 프로젝트 개요

- **과업명**: AI를 활용한 자금세탁탐지 방법론 (Anti-Money Laundering, AML)
- **파트너사**: 금융결제원 (KFTC) AX혁신전략실 — Team IRIS
- **멘토**: Seonkyu Lim (sklim@kftc.or.kr)
- **주관**: 모두의연구소 기업 프로젝트 (2026)
- **배경**: BIS Innovation Hub × Bank of England 공동 수행 **Project Hertha**(2024.2–2025.6) 챌린지에 금융결제원 Team IRIS가 참여한 경험을 기반으로, **결제시스템 운영자 관점에서 네트워크 분석 기반 AML 탐지** 방법론을 심화 연구.
- **핵심 목표**:
  1. 거래 데이터를 그래프로 구성하고 **Graph Feature(중심성 등)**가 탐지 성능에 기여하는 정도 정량화
  2. **초대규모 그래프**(수백만 계좌 × 수억 거래)에서 GNN 실용성 검토 — 샘플링, 파티셔닝, 미니배치, 임베딩 피처화 전략 비교
  3. 가능하면 **Low-Homophily**(의심 계좌가 정상 계좌와 주로 연결) 이슈와 **AML Agent 벤치마크** 연구 방향까지 연계

---

## 2. 레퍼런스 경로

> 에이전트는 작업 전 아래 문서들을 먼저 참조할 것. 경로/URL은 실제 값으로 교체.

### 2.1 코드베이스
- **메인 레포지토리**: `/Users/jkyu/Desktop/projects/GraphAML` 또는 `https://github.com/<org>/<repo>`
- **브랜치 전략**: `main` (배포) / `develop` (통합) / `feature/*` / `exp/*` (실험)
- **하위 모듈**:
  - `./data/` — 데이터 다운로드·전처리·그래프 빌드
  - `./features/` — 중심성·모티프·커뮤니티 등 Graph Feature 추출
  - `./models/` — 베이스라인(XGBoost, LightGBM) + GNN (GCN, GAT, GraphSAGE, GIN, PNA 등)
  - `./scaling/` — 샘플링(GraphSAINT, Cluster-GCN), 파티셔닝, 미니배치
  - `./evaluation/` — 극단 불균형 환경 평가(AUPRC 중심)
  - `./notebooks/` — EDA, 시각화

### 2.2 논문 · 참고자료

| 구분 | 제목 / 출처 | 요약 |
|------|------------|------|
| 리뷰 핵심 | **Project Hertha Report (BIS, 2025)** — <https://www.bis.org/publ/othp96.htm> | BIS-BoE 공동 AML 네트워크 분석 검증. 협업 기반 탐지가 특히 복잡·다층 패턴에서 정확도 크게 향상. 10개 typology 정의(Gather-scatter, Scatter-gather, Stack, Fan-out, Fan-in, Simple cycle 등) |
| 대표 GNN 논문 | **Cheng et al., "Anti-money laundering by group-aware deep graph learning"** — IEEE TKDE 35.12 (2023): 12444–12457 (GAGNN) | 유니온페이 데이터, 의심 계좌 간 커뮤니티 패턴 탐지. 그룹 인지(group-aware) 구조. |
| GNN 계보 | **FlowScope** (AAAI '20, 중신은행 참여) | 다단계 대규모 자금 흐름 추적. 경로 기반 탐지. |
| GNN 계보 | **MonLAD** (WSDM '22, Tencent) | 실시간 스트림 + 잔액 변동 기반 중개 계좌(Agent) 탐지. |
| GNN 계보 | **GRANDE** (ICDM '22, Ant Group) | 방향성·엣지 단위 특징 동시 학습. |
| GNN 계보 | **GFP** (ICAIF '24, IBM Research) | 스머핑 등 그래프 패턴 피처 생성 → 다운스트림 탐지 성능 개선. |
| 데이터 논문 | **Altman et al., "Realistic Synthetic Financial Transactions for AML Models"** (NeurIPS '23) | IBM AMLworld 데이터셋 생성 방법론. 6개 변형(HI/LI × S/M/L). |
| GNN 기초 | **Hamilton et al., "Inductive Representation Learning on Large Graphs"** (GraphSAGE, NeurIPS '17) | 대규모 그래프 미니배치·이웃 샘플링의 표준. |
| GNN 기초 | **Chiang et al., "Cluster-GCN"** (KDD '19) | 메모리 효율적 대규모 GCN 학습. |
| Nasdaq | **Global Financial Crime Report (Nasdaq Verafin, 2024)** | 2023년 약 3.1조 달러 불법자금 유통. 유형별 규모 추정. |

> 경로 규칙: 논문 PDF는 `./docs/papers/<firstAuthor><Year>_<shortTitle>.pdf`, 요약 노트는 `./docs/papers/notes/` 아래 동일 파일명 `.md`로.

### 2.3 회의 스크립트
- **킥오프 멘토링 (2026-04-30)**: `./docs/meetings/2026-04-21_kickoff.md`
- **주간 멘토링**: `./docs/meetings/weekly/YYYY-MM-DD_topic.md`
- **의사결정 로그**: `./docs/meetings/decisions.md` (누가 / 언제 / 무엇을 / 왜 / 대안)

### 2.4 기획안
- **프로젝트 기획안**: `./docs/plan/project-plan.md`
- **실험 트랙 로드맵**: `./docs/plan/roadmap.md`
  - Track A: Graph Feature 기여도 분석
  - Track B: 초대규모 GNN 확장성
  - Track C (stretch): Low-Homophily 대응, AML Agent 벤치마크 리뷰
- **완료기준(DoD)**: `./docs/plan/definition-of-done.md`

### 2.5 기업설명회 / 멘토링 PDF
- **KFTC 멘토링 자료 (2026-04-21)**: `./docs/rfp/2026-04-21_KFTC_AML_mentoring.pdf`
- **발췌 노트**: `./docs/rfp/kftc-mentoring-notes.md`

### 2.6 노션 페이지 (URL만, MCP 연동 X)
- **프로젝트 허브**: `<노션 URL>`
- **리서치 노트**: `<노션 URL>`
- **주간 리포트**: `<노션 URL>`

> 에이전트는 노션에 **직접 읽기/쓰기 액세스가 없다**. URL은 참조용으로만. 필요한 내용은 로컬 `./docs/notion-sync/` 에 마크다운으로 동기화한 사본을 둘 것.

---

## 3. 프로젝트 방향 (모두의연구소)

### Track A — Graph Feature 활용을 통한 탐지 모델 성능 개선
자금세탁은 개별 거래보다 **계좌 간 연결 구조·자금 흐름 패턴**에서 드러남. 거래 데이터를 그래프로 구성하고 다음 피처들을 추출하여 탐지 성능에 얼마나 기여하는지 ablation:
- **중심성 지표**: Degree / Closeness / Betweenness / Eigenvector / PageRank
- **지역 구조**: Clustering Coefficient, Triangles, k-core, Motif 카운트
- **커뮤니티**: Louvain, Leiden, Label Propagation
- **시간적 특징**: 시간창(window) 내 in/out 거래 빈도·금액 분포, burst 감지
- **비교군**: Graph Feature 없음 (raw transaction features only) vs. 있음 (raw + graph features) 동일 분류기(XGBoost/LightGBM)에서 비교

### Track B — 초대규모 데이터에서 GNN 활용성 검토
- **현실 제약**: 수백만 계좌 × 수억 거래 → Full-batch GNN 학습 불가
- **접근 방법 비교**:
  - Neighbor Sampling (GraphSAGE)
  - Cluster-GCN
  - GraphSAINT (subgraph sampling)
  - Mini-batch full-neighbor
  - Historical embedding (GNNAutoScale 등)
- **대안 전략**: GNN을 탐지 모델로 직접 쓰지 않고, **노드 임베딩만 생성 → 피처화 → 경량 ML 다운스트림** (XGBoost 등) 조합
- **측정 항목**: 학습 시간, 피크 메모리, AUPRC, 배포 inference 지연

### Track C (선택) — Low-Homophily & AML Agent
- **Low-Homophily**: 의심(S)-정상(B) 연결이 많아 GNN의 이웃 집계 과정에서 의심 신호가 희석. **Behavioral k-NN view** 등 행동 기반 유사도 그래프 보강 실험.
- **AML Agent 벤치마크**: Function Calling 기반 multi-step 분석(거래 조회 → 패턴 탐지 → 위험 평가 → STR 작성) 워크플로 리뷰.

---

## 4. 대상 데이터셋

### 기본: IBM Transactions for AML (AMLworld)
- **출처**: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
- **구성**: 6 datasets = {HI, LI} × {Small, Medium, Large}
  - HI / LI: Higher / Lower illicit ratio
  - Small ≈ 5–7M, Medium ≈ 31–32M, Large ≈ 176–180M transactions
- **라벨**: 거래 단위 `Is Laundering` + 패턴 유형(Scheme)
- **용도**: Track A/B 공통 벤치마크. 확장 실험용 엣지 케이스 샘플은 별도 폴더.

### 보조 (참고용, 라이선스 확인 필수)
- Elliptic (Bitcoin transactions, `~2%` anomaly)
- DGraph-Fin (Social fraud, `<0.1%`)
- IEEE-CIS (Card transactions, `3.5%`)
- YelpChi / Amazon (Review fraud) — 금융 도메인 아님, 주의

> Project Hertha 원본 합성 데이터(3억8천만 건, 자금세탁 0.011%)는 **비공개**. 인용·비교 목적으로만 언급.

---

## 5. 기술 스택 · 컨벤션

- **언어**: Python 3.11+
- **데이터**: Polars(대용량 ETL), Pandas(소규모 EDA), DuckDB(로컬 SQL)
- **그래프 엔진**: NetworkX(소규모), **igraph / graph-tool**(대규모 중심성 계산), cuGraph(GPU 옵션)
- **GNN 프레임워크**: **PyTorch Geometric (PyG)** 기본, DGL 병행 검토
- **ML 베이스라인**: XGBoost, LightGBM, CatBoost
- **실험 추적**: MLflow 또는 Weights & Biases
- **저장소**: Parquet(테이블), `.pt`/`.bin`(그래프 텐서), `.npz`(희소 인접행렬)
- **코드 스타일**: `ruff` + `black`, 타입 힌트 필수, docstring Google style
- **난수 시드 고정**: `seed=42` 기본, 실험은 최소 4-seed 평균±std 보고 (Hertha Team IRIS 표기 방식 준수)

---

## 6. 유의 사항

### 6.1 데이터 · 법적 경계
- **실제 금융 거래 데이터 직접 접근 금지**. 본 프로젝트는 **합성 데이터 또는 공개 벤치마크만** 사용. KFTC 내부 데이터는 **언급 금지, 추론 시도 금지**.
- 한국 법령(특정금융거래정보법, 금융실명법, 개인정보보호법)과 **FATF 권고사항** 맥락 인지. 프로덕션 아이디어 제안 시 법규 충돌 여부 체크.
- **STR / CTR / CDD / EDD / PEP** 등 용어는 정의부터 정확히 — 발신 실수가 오개념으로 굳음.

### 6.2 극단적 클래스 불균형
- 자금세탁 거래 비율 = **0.01% 수준**. `Accuracy`는 **쓸모없음**. 99.99% 찍어도 "전부 정상"으로 도달 가능.
- **권장 지표**: AUPRC (Average Precision) 최우선, F1 (positive class), Recall@K, Precision@K
- **보조 지표**: AUROC는 보고하되 의사결정 기준 삼지 말 것 — 불균형에서 낙관적으로 편향.
- **Threshold 튜닝**: validation set에서 별도 조정, test set에 leak 금지.

### 6.3 그래프 구성 시 주의
- **방향성**: 송금 → 수취 (directed). 양방향 무시하면 cycle/fan-in/fan-out 구분 불가.
- **Multi-edge**: 동일 계좌쌍 간 거래 여러 건. 단순화하면 빈도·burst 정보 소실. MultiDiGraph 유지 권장.
- **노드 타입**: Bank Account / Cryptocurrency Exchange / Cash / Foreign — **heterogeneous graph**로 다루면 Hertha typology 구분이 자연스러움.
- **시간 정보**: 반드시 유지. 정적 스냅샷은 **temporal leakage** 위험.
- **Self-loop**: 자행이체는 케이스별 판단.

### 6.4 데이터 누수(Leakage) 방지
- **Train/Val/Test 분할은 시간 기준**. 랜덤 분할은 미래 정보로 과거 예측하는 leak 유발.
- 그래프 피처 계산 시 **각 split 내부 데이터만** 사용 (test edge 포함하여 중심성 계산하면 누수).
- 동일 의심 커뮤니티의 여러 계좌가 다른 split에 섞이는 **account-level leakage**도 체크.

### 6.5 Low-Homophily 특성
- 자금세탁 그래프는 S(suspicious)-B(benign) 연결 비율이 높음 (예: S-S 15% : S-B 85%).
- 표준 GCN/GAT는 homophily 가정 → 성능 저하. **Heterophily-aware 모델**(H2GCN, GPR-GNN, FAGCN) 또는 **Behavioral k-NN view** 같은 구조 보강 고려.

### 6.6 초대규모 그래프 운영
- Full-batch GCN은 **수백만 노드부터 OOM** (A100 80GB에서도).
- 이웃 샘플링 시 fan-out, layer 수, batch size 튜닝이 성능에 크게 영향. 과도한 샘플링은 구조 신호 손실.
- **CPU 메모리**가 종종 GPU보다 먼저 터짐 — 엣지 리스트 int32 캐스팅, Parquet partitioning.
- Preprocessing(피처 계산, 이웃 인덱싱)이 학습보다 오래 걸리는 경우가 흔함. **캐시 & 재사용** 설계.

### 6.7 평가 · 재현성
- 최소 **4 seed 평균 ± std** 보고. 단일 seed 결과로 우위 주장 금지.
- Hyperparameter tuning은 validation에서만. test는 최종 보고 1회.
- 실험 결과는 MLflow/W&B에 자동 기록. 수기 스프레드시트 금지.

### 6.8 에이전트 작업 규칙
- **대용량 데이터 로드 전 샘플링 전략 먼저**. 180M rows를 무지성 `pd.read_csv`로 열면 즉사.
- **실행 시간 오래 걸리는 코드는 `--dry-run`, `--sample`, `--limit` 플래그 먼저 제공** 후 사용자 승인.
- **기존 파일 임의 리팩토링 금지**. 실험 브랜치에서만 수정.
- 데이터 파일(CSV/Parquet/그래프 텐서)은 **Git LFS 또는 별도 스토리지**, 레포에 직접 커밋 금지.
- API 키·토큰·내부 URL은 `.env` + `.gitignore`. 절대 커밋 금지.
- 노션·회의록 등 외부 레퍼런스를 **직접 수정하려 하지 말 것**. 로컬 사본에서만.
- **법률·규제 해석 질문은 단정하지 말 것** — "법률 자문 아님, 확인 필요" 플래그 동반.

---

## 7. 빠른 시작

```bash
# 환경 세팅
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
# PyG는 CUDA 버전에 맞춰 별도 설치
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# IBM AML 데이터 다운로드 (Small 먼저)
python -m data.fetch_aml --variant HI-Small

# 그래프 빌드 (directed multigraph, timestamp 유지)
python -m data.build_graph --input data/raw/HI-Small.parquet --out data/graphs/hi-small.pt

# Graph Feature 추출
python -m features.extract_centrality --graph data/graphs/hi-small.pt --out data/features/hi-small_centrality.parquet

# 베이스라인 (XGBoost, raw features only)
python -m models.train_xgb --features raw --data HI-Small --seed 42

# 베이스라인 + Graph Features (Track A 비교)
python -m models.train_xgb --features raw+graph --data HI-Small --seed 42

# GraphSAGE (Track B 샘플링 비교)
python -m models.train_gnn --model sage --sampler neighbor --data HI-Small --seed 42
```

---

## 8. 연락처 · 오너십

- **프로젝트 리드**: 유진국 / jkyupsyche@gmail.com>
- **KFTC 멘토**: Seonkyu Lim (sklim@kftc.or.kr, +82 10-8603-3390)
- **모두의연구소 담당**: 박기웅 (교육퍼실리테이터팀)

---

*Last updated: 2026-04-28*
