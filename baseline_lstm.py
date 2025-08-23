# =========================
#version00
# =========================

import os
import random
import glob
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from korean_lunar_calendar import KoreanLunarCalendar


plt.rcParams['font.family'] = 'AppleGothic'  # macOS

#Fixed Random Seed  & Setting Hyperparameter
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 32, 50
DEVICE = torch.device("cpu")
MONTH_SCALE = 12

MIN_SEQUENCE_COUNT = 10

# 키: '영업장명_메뉴명', 값: 'YYYY-MM-DD' (ISO 문자열 또는 datetime)
DISCONTINUED = {
    '담하_꼬막_비빔밥': '2024-04-01',
    '담하_들깨_양지탕': '2024-04-01',
    # ...
}

def get_lunar_to_solar(years, lunar_month, lunar_day, span=1):
    calendar = KoreanLunarCalendar()
    dates = []
    for year in years:
        for offset in range(-span, span+1):
            try:
                calendar.setLunar(year, lunar_month, lunar_day + offset, False)
                dates.append(calendar.SolarIsoFormat())
            except:
                pass  # 예외 처리: 음력 마지막날 초과
    return dates
# 예시: 2023 ~ 2025
years = [2023, 2024, 2025]
lunar_solar_dates = []
lunar_solar_dates += get_lunar_to_solar(years, 1, 1, span=1)   # 설날 ±1
lunar_solar_dates += get_lunar_to_solar(years, 8, 15, span=1)  # 추석 ±1

solar_md_holidays = [
    (1, 1),   # 신정
    (3, 1),   # 삼일절
    (5, 5),   # 어린이날
    (6, 6),   # 현충일
    (8, 15),  # 광복절
    (10, 3),  # 개천절
    (10, 9),  # 한글날
    (12, 25), # 크리스마스
]

def generate_combined_holiday_list(df, solar_md_list, lunar_solar_list):
    df = df.copy()
    df['영업일자'] = pd.to_datetime(df['영업일자'])

    # 양력 기반 holiday 판별
    df['is_solar_holiday'] = df['영업일자'].apply(
        lambda x: (x.month, x.day) in solar_md_list
    )

    # 음력 변환된 holiday 포함
    lunar_set = set(pd.to_datetime(lunar_solar_list))
    df['is_lunar_holiday'] = df['영업일자'].isin(lunar_set)

    # 최종 통합
    df['is_holiday'] = (df['is_solar_holiday'] | df['is_lunar_holiday']).astype(int)
    df = df.drop(columns=['is_solar_holiday', 'is_lunar_holiday'])
    return df

#Define Model
class MultiEmbeddingLSTM(nn.Module):
    def __init__(
        self,
        input_dim,                  # 수치 feature 수
        hidden_dim=64,
        num_layers=2,
        output_dim=7,
        num_weekdays=7,
        weekday_embed_dim=3,
        num_seasons=4,
        season_embed_dim=2,
        dropout=0.3
    ):
        super().__init__()

        # 임베딩
        self.weekday_embedding = nn.Embedding(num_weekdays, weekday_embed_dim)
        self.season_embedding = nn.Embedding(num_seasons, season_embed_dim)

        # LayerNorm for embedding
        self.weekday_norm = nn.LayerNorm(weekday_embed_dim)
        self.season_norm = nn.LayerNorm(season_embed_dim)

        # Dropout after embedding
        self.embedding_dropout = nn.Dropout(dropout)

        total_input_dim = input_dim + weekday_embed_dim + season_embed_dim

        # LSTM
        self.lstm = nn.LSTM(total_input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # LSTM output dropout
        self.post_lstm_dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, weekday_ids, season_ids):
        """
        x: (B, T, input_dim)
        weekday_ids: (B, T)
        season_ids: (B, T)
        """
        weekday_embed = self.weekday_embedding(weekday_ids)      # (B, T, D2)
        season_embed = self.season_embedding(season_ids)         # (B, T, D3)

        # 정규화
        weekday_embed = self.weekday_norm(weekday_embed)
        season_embed = self.season_norm(season_embed)

        x_concat = torch.cat([x, weekday_embed, season_embed], dim=-1)  # (B, T, total_dim)
        x_concat = self.embedding_dropout(x_concat)

        out, _ = self.lstm(x_concat)
        out = self.post_lstm_dropout(out)

        return self.fc(out[:, -1, :])
    
def clip_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return np.clip(series, None, upper)
    # return series

def compute_iqr_lower_bounds(train_df):
    lower_bounds = {}
    for menu, group in train_df.groupby('영업장명_메뉴명'):
        q1 = group['매출수량'].quantile(0.25)
        q3 = group['매출수량'].quantile(0.75)
        iqr = q3 - q1
        lower = max(q1 - 1.5 * iqr, 0)
        menu_key = menu[0] if isinstance(menu, tuple) else menu
        lower_bounds[menu_key] = lower
    return lower_bounds

#Train
def train_lstm(train_df, use_validation=True, dropout=0.3):
    trained_models = {}

    for store_menu, group in tqdm(train_df.groupby(['영업장명_메뉴명']), desc ='Training LSTM'):
        #시계열 정렬 , preprocessing
        key = store_menu[0] if isinstance(store_menu, tuple) else store_menu
        store_train = group.sort_values('영업일자').copy()
        store_train['영업일자'] = pd.to_datetime(store_train['영업일자'])
        store_train['weekday'] = store_train['영업일자'].dt.dayofweek
        store_train['weekday'] = store_train['weekday'].astype(int)
        store_train['month'] = store_train['영업일자'].dt.month

        store_train['season'] = store_train['month'].map({
            12:0,1:0,2:0, #winter
            3:1,4:1,5:1, #spring
            6:2,7:2,8:2, #summer
            9:3,10:3,11:3 #fall
        })  
        store_train['season'] = store_train['season'].astype(int)

        store_train['month'] /= MONTH_SCALE
        

        #data clipping
        store_train['clipped_SQ'] = clip_iqr(store_train['매출수량'])
        store_train['delta'] = store_train['clipped_SQ'].diff().fillna(0)
        
        #rolling_mean_7
        store_train['rolling_mean_7'] = store_train['clipped_SQ'].rolling(window=7, min_periods=1).mean()

        #train/validation split을 안전하게 하기 위해 여유분 데이터를 확보하는 목적 -> MIN_SEQUENCE_COUNT
        if len(store_train) < LOOKBACK + PREDICT + MIN_SEQUENCE_COUNT:
            continue

        #features = ['clipped_SQ','weekday','month','season', 'is_holiday', 'rolling_mean_7','delta_scaled', 'store_id']
        features = ['clipped_SQ', 'rolling_mean_7', 'delta_scaled', 'month', 'is_holiday']
        scaler = MinMaxScaler()

        ##정규화, 요일정보 
        store_train[['clipped_SQ','rolling_mean_7']] = scaler.fit_transform(store_train[['clipped_SQ', 'rolling_mean_7']])
        scaler_delta = MinMaxScaler()
        store_train[['delta_scaled']] = scaler_delta.fit_transform(store_train[['delta']])


        train_vals = store_train[features].values  # shape: (N, 8)

        # 시퀀스 생성
        X, y, weekday_seqs, season_seqs = [], [], [], []

        for i in range(len(store_train) - LOOKBACK - PREDICT + 1):
            X.append(store_train[features].values[i:i+LOOKBACK])
            y.append(store_train['clipped_SQ'].values[i+LOOKBACK:i+LOOKBACK+PREDICT])
            weekday_seqs.append(store_train['weekday'].values[i:i+LOOKBACK])
            season_seqs.append(store_train['season'].values[i:i+LOOKBACK])

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        weekday_seqs = torch.tensor(weekday_seqs).long()
        season_seqs = torch.tensor(season_seqs).long()

        if use_validation:
            # Train/Validation split (기존 방식 유지)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            weekday_train, weekday_val = weekday_seqs[:split_idx], weekday_seqs[split_idx:]
            season_train, season_val = season_seqs[:split_idx], season_seqs[split_idx:]
        else:
            # 전체 데이터 학습에 사용
            X_train = X
            y_train = y
            weekday_train = weekday_seqs
            season_train = season_seqs

        X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
        weekday_train, season_train = weekday_train.to(DEVICE), season_train.to(DEVICE)

        if use_validation:
            X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
            weekday_val, season_val = weekday_val.to(DEVICE), season_val.to(DEVICE)

        model = MultiEmbeddingLSTM(input_dim=len(features), output_dim=PREDICT, dropout=dropout).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        #loss 저장 리스트
        train_losses = []
        val_losses = []

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            idx = torch.randperm(len(X_train))

            for i in range(0, len(X_train), BATCH_SIZE):
                batch_idx = idx[i:i+BATCH_SIZE]
                X_batch = X_train[batch_idx]
                y_batch = y_train[batch_idx]
                weekday_batch = weekday_train[batch_idx]
                season_batch = season_train[batch_idx]

                output = model(X_batch, weekday_batch, season_batch)
                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / (len(X_train) // BATCH_SIZE + 1)
            train_losses.append(avg_loss)

            if use_validation:
                model.eval()
                with torch.no_grad():
                    val_output = model(X_val, weekday_val, season_val)
                    val_loss = criterion(val_output, y_val).item()
                    val_losses.append(val_loss)

        visualize_loss(train_losses, val_losses if use_validation else None, key, save=True)

        trained_models[store_menu] = {
            'model': model.eval(),
            'scaler': scaler,
            'scaler_delta': scaler_delta,
            'last_sequence': {
                'X': train_vals[-LOOKBACK:],  # 수치 입력
                'weekday': store_train['weekday'].values[-LOOKBACK:],  # 임베딩용
                'season': store_train['season'].values[-LOOKBACK:]
            }
        }

    return trained_models

def visualize_loss(train_losses, val_losses, store_menu, save=False, out_dir="./loss_plots"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    if val_losses is not None:
        plt.plot(val_losses, label='Validation Loss')
    plt.title(f"[{store_menu}] Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        safe_name = re.sub(r'[^\w\-_.]', '_', store_menu)
        path = os.path.join(out_dir, f"{safe_name}.png")
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

#Prediction
def predict_lstm(test_df, trained_models, test_prefix: str, *,
    discontinued: dict[str, str | pd.Timestamp] | None = None,  # <- 추가
    rule: str = 'after',     # 'after' : 단종일자 이후(>) 0, 'on_or_after' : 단종일자 당일 포함(>=) 0
    grace_days: int = 0
    ):            # 유예일 (단종일자 + grace_days 이후부터 0)
    results = []

    cutoff_map = None
    if discontinued is not None:
        def _to_ts(v):
            return v if isinstance(v, pd.Timestamp) else pd.to_datetime(v)
        cutoff_map = {k: _to_ts(v) for k, v in discontinued.items()}
        if grace_days != 0:
            for k in cutoff_map:
                cutoff_map[k] = cutoff_map[k] + pd.Timedelta(days=grace_days)

    for store_menu, store_test in test_df.groupby(['영업장명_메뉴명']):
        key = store_menu
        if key not in trained_models:
            continue

        model = trained_models[key]['model']
        scaler = trained_models[key]['scaler']
        scaler_delta = trained_models[key]['scaler_delta']
        

        store_test_sorted = store_test.sort_values('영업일자').copy()
        store_test_sorted['영업일자'] = pd.to_datetime(store_test_sorted['영업일자'])
        store_test_sorted['weekday'] = store_test_sorted['영업일자'].dt.dayofweek.astype(int)
        store_test_sorted['month'] = store_test_sorted['영업일자'].dt.month
        store_test_sorted['season'] = store_test_sorted['month'].map({
            12:0,1:0,2:0,
            3:1,4:1,5:1,
            6:2,7:2,8:2,
            9:3,10:3,11:3
        }).astype(int)
        store_test_sorted['clipped_SQ'] = clip_iqr(store_test_sorted['매출수량'])

        #month 정규화
        store_test_sorted['month'] /= MONTH_SCALE

        # holiday feature 추가
        store_test_sorted = generate_combined_holiday_list(store_test_sorted, solar_md_holidays, lunar_solar_dates)

        # rolling_mean_7 계산
        store_test_sorted['rolling_mean_7'] = store_test_sorted['clipped_SQ'].rolling(window=7, min_periods=1).mean()

        #delta
        store_test_sorted['delta'] = store_test_sorted['clipped_SQ'].diff().fillna(0)
        store_test_sorted[['delta_scaled']] = scaler_delta.transform(store_test_sorted[['delta']])


        # feature_cols = ['매출수량','month','is_holiday','rolling_mean_7','delta_scaled']
        # recent_vals = store_test_sorted[feature_cols].values[-LOOKBACK:]

        # 최근 LOOKBACK 개 추출
        if len(store_test_sorted) < LOOKBACK:
            last_seq = trained_models[key]['last_sequence']
            x_input = torch.tensor([last_seq['X']]).float().to(DEVICE)
            weekday_seq = torch.tensor([last_seq['weekday']]).long().to(DEVICE)
            season_seq = torch.tensor([last_seq['season']]).long().to(DEVICE)
            last_obs_date = pd.to_datetime(last_seq['영업일자'].max())
        else:
            recent_vals = store_test_sorted.iloc[-LOOKBACK:].copy()

            # clipped_SQ 생성 및 정규화
            recent_vals['clipped_SQ'] = clip_iqr(recent_vals['매출수량'])
            recent_vals['rolling_mean_7'] = recent_vals['clipped_SQ'].rolling(window=7, min_periods=1).mean()
            recent_vals['delta'] = recent_vals['clipped_SQ'].diff().fillna(0)
            recent_vals[['clipped_SQ', 'rolling_mean_7']] = scaler.transform(recent_vals[['clipped_SQ', 'rolling_mean_7']])
            recent_vals[['delta_scaled']] = scaler_delta.transform(recent_vals[['delta']])

            feature_cols = ['clipped_SQ', 'rolling_mean_7', 'delta_scaled', 'month', 'is_holiday']
            x_input = torch.tensor([recent_vals[feature_cols].values]).float().to(DEVICE)
            weekday_seq = torch.tensor([recent_vals['weekday'].values]).long().to(DEVICE)
            season_seq = torch.tensor([recent_vals['season'].values]).long().to(DEVICE)
            last_obs_date = pd.to_datetime(recent_vals['영업일자'].max())


        # 예측
        with torch.no_grad():
            pred_scaled = model(x_input, weekday_seq, season_seq).squeeze().cpu().numpy()


        # 역정규화 및 하한 클리핑
        restored = []
        for i in range(PREDICT):
            val = pred_scaled[i]
            dummy_input = np.zeros((1, 2))
            dummy_input[0, 0] = val
            restored_val = scaler.inverse_transform(dummy_input)[0][0]
            #restored.append(max(restored_val, lower_bound))
            restored.append(max(restored_val, 0))  # 또는 lower_bound

        # ----- (B) 단종 처리: 예측 구간 실제 날짜와 비교 -----
        # 예측 구간의 "실제 달력 날짜" 생성 (last_obs_date 다음날부터 PREDICT일)
        horizon_dates = pd.date_range(start=last_obs_date + pd.Timedelta(days=1),
                                      periods=PREDICT, freq='D')

        if cutoff_map is not None:
            cutoff = cutoff_map.get(key, None)
            if cutoff is not None:
                if rule == 'on_or_after':
                    zero_mask = horizon_dates >= cutoff
                else:  # 'after'
                    zero_mask = horizon_dates > cutoff
                vals_real = np.where(zero_mask, 0.0, vals_real)


        # 예측일자: TEST_00+1일 ~ TEST_00+7일
        # 결과 저장
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
        for d, val in zip(pred_dates, restored):
            results.append({
                '영업일자': d,
                '영업장명_메뉴명': store_menu[0],
                '매출수량': val
            })
    

    return pd.DataFrame(results)

def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    # (영업일자, 메뉴) → 매출수량 딕셔너리로 변환
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))

    final_df = sample_submission.copy()

    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:  # 메뉴명들
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)

    return final_df
