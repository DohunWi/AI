# Resort F&B 7-Day Forecasting (Lookback 28) / Dacon — Engineering README
Task : 곤지암 리조트 데이터 분석 및 식음업장 수요예측

## 주최 / 주관
- 주최 : LG AI 연구원
- 주관 : 데이콘
- 참여 : 한경닷컴

## Motivation
<업장별 단기 수요예측으로 재고/인력 최적화. 담하/미라시아 가중 高>

## 개요
- 목표: 업장/메뉴 단위 단기 수요예측(7일)
- 입력: `TEST_00.csv` ~ `TEST_09.csv` (각 28일 구간)
- 출력: 각 파일 구간의 **다음 7일** 예측값
- 환경: VS Code + Jupyter Notebook, macOS(M2)

## Project Structure
.
├── __pycache__
│   ├── baseline_lstm.cpython-310.pyc
│   ├── clipping.cpython-310.pyc
│   ├── cluster_based_model.cpython-310.pyc
│   ├── complete_advanced_model.cpython-310.pyc
│   ├── ens_model.cpython-310.pyc
│   ├── hotfix.cpython-310.pyc
│   ├── launch_detector.cpython-310.pyc
│   ├── menu_clipping.cpython-310.pyc
│   ├── menu_detector.cpython-310.pyc
│   └── weight_processor.cpython-310.pyc
├── data
│   ├── test
│   └── train
├── debug.log
├── EP/
├── helper/
├── LG AI certification.pdf
├── models/
├── outputs
│   └── Submission
├── README.md
└── src
    ├── baseline_cluster.ipynb
    ├── baseline_lstm.py
    ├── Baseline.ipynb
    ├── cluster_based_model.py
    ├── ens_model.py
    └── ensemble_model.ipynb

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install pandas numpy scikit-learn matplotlib tqdm korean_lunar_calendar
pip install torch xgboost lightgbm
```

## Data
- 학습: `data/train/train.csv`  /  약 18개월 구간의 시계열 업장_메뉴 매출 데이터
- 평가 입력: `data/test/TEST_00~09.csv` / 28일 구간의 매출 데이터
- 주요 컬럼: `영업일자`, `영업장명_메뉴명`, `매출수량`
- 파생/도메인 특성: `weekday`, `month`, `season`, `is_holiday(양·음력 통합)`, `rolling_mean_7`, `delta`
- 전처리: IQR 클리핑, 메뉴별 스케일링, 예측 하한 **min=1** 클리핑

## Method
- **LSTM**(MultiEmbedding): weekday/season 임베딩, 28→7, MSE 학습
- **Cluster-Based**: 메뉴 특성 → KMeans 군집 → RF 분류(0 매출) + LGBM(글로벌) + XGB(클러스터별)
- **Ensemble**: 기본 전역 가중 **LSTM 0.3 / Cluster 0.7**,

## Reproducibility
- Seed: <42>
- Env: Python 3.10.18, PyTorch <2.8.0>, MPS: on(if available)

## How to Run
- **One-liner**: `run_ensemble_pipeline()` 

## 실행 방법
```bash 
git clone https://github.com/DohunWi/AI.git
cd AI/src
python3 ens_model.py
```

## 예측 평가 산식
- 식음업장 별 가중치가 있는 SMAPE 

## score
- public : 0.48093 , 40/819
- private : 0.48347 , 54/817
