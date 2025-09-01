# Resort F&B 7-Day Forecasting (Lookback 28) — Engineering README
Task : 곤지암 리조트 데이터 분석 및 식음업장 수요예측


## Motivation
<업장(메뉴) 수요 변동성, 연휴/주말 영향, 시계열 특성 고려, 고가중치 업장(담하/미라시아)에서의 비즈니스 임팩트>

## 개요
- 목표: 업장/메뉴 단위 단기 수요예측(7일)
- 입력: `TEST_00.csv` ~ `TEST_09.csv` (각 28일 구간)
- 출력: 각 파일 구간의 **다음 7일** 예측값
- 환경: VS Code + Jupyter Notebook, macOS(M2)

## 데이터
- 학습: `data/train/train.csv`
- 평가 입력: `data/test/TEST_00~09.csv`
- 주요 컬럼: `영업일자`, `영업장명_메뉴명`, `매출수량`
- 파생/도메인 특성: `weekday`, `month`, `season`, `is_holiday(양·음력 통합)`, `rolling_mean_7`, `delta`
- 전처리: IQR 클리핑, 메뉴별 스케일링, 예측 하한 **min=1** 클리핑

## Method
- Models: LSTM(Multi-Output, 임베딩), tree-based clustering-model(RF, LightGBM, XGBoost)
- Ensemble: 두 개 모델의 예측값을 3:7 가중평균
- Loss/Metric: MSE + **Weighted SMAPE(평가, 실제=0일 제외)**
- Feature Eng.: weekday/month/season, is_holiday(양·음력), rolling_mean_7, delta, momentum/volatility(옵션), clustering-features
- Preprocess: IQR clipping, per-menu MinMaxScaler, **min=1** clip


