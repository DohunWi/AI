# Resort F&B 7-Day Forecasting (Lookback 28) — Engineering README
Task : 곤지암 리조트 데이터 분석 및 식음업장 수요예측


## Motivation
<업장별 단기 수요예측으로 재고/인력 최적화. 담하/미라시아 가중 高>

## 개요
- 목표: 업장/메뉴 단위 단기 수요예측(7일)
- 입력: `TEST_00.csv` ~ `TEST_09.csv` (각 28일 구간)
- 출력: 각 파일 구간의 **다음 7일** 예측값
- 환경: VS Code + Jupyter Notebook, macOS(M2)

## 데이터
- 학습: `data/train/train.csv`  /  약 18개월 구간의 시계열 업장_메뉴 매출 데이터
- 평가 입력: `data/test/TEST_00~09.csv` / 28일 구간의 매출 데이터
- 주요 컬럼: `영업일자`, `영업장명_메뉴명`, `매출수량`
- 파생/도메인 특성: `weekday`, `month`, `season`, `is_holiday(양·음력 통합)`, `rolling_mean_7`, `delta`
- 전처리: IQR 클리핑, 메뉴별 스케일링, 예측 하한 **min=1** 클리핑

## Method
- **LSTM**(MultiEmbedding): weekday/season 임베딩, 28→7, MSE 학습
- **Cluster-Based**: 메뉴 특성 → KMeans 군집 → RF 분류(0 매출) + LGBM(글로벌) + XGB(클러스터별)
- **Ensemble**: 전역 예측값 0.3/0.7 가중 평균

## Reproducibility
- Seed: <42>
- Env: Python 3.10.18, PyTorch <2.8.0>, MPS: on(if available)

## How to Run
- **One-liner**: `run_ensemble_pipeline()` 
