
# ===============================================================
# Store-wise Global N-HiTS (v6) — Complete, Commented Implementation
# Author: ChatGPT (GPT-5 Thinking)
# Target env: VSCode + Jupyter (macOS M2; MPS if available)
#
# What this file provides
# ----------------------
# 1) "업장 단위" 글로벌 모델 (한 업장 내 모든 메뉴를 하나의 모델이 공동 학습)
#    - 메뉴 임베딩(item embedding)으로 메뉴별 고유성 반영
#    - 업장 컨텍스트(store_rm7 = 업장 총매출 7일 이동평균)로 메뉴 간 상관성 주입
# 2) 훈련/추론 함수:
#    - train_nhits_storewise(train_df, use_validation=True, ...)
#    - predict_nhits_storewise(test_df, trained, test_prefix, ...)
# 3) 제출 변환 유틸:
#    - convert_to_submission_format(pred_df, sample_submission)
#
# 입력 데이터 형식(예시 컬럼)
# -------------------------
# - '영업일자': 날짜(문자열/Datetime)
# - '영업장명_메뉴명': "업장명_메뉴명" 문자열
# - '매출수량': 실수(학습 시)
#
# 출력
# ----
# - train: 업장별 패키지 dict (모델, 스케일러 등)
# - predict: ['영업일자','영업장명_메뉴명','매출수량'] DataFrame
#
# 사용 예
# -------
# >>> import pandas as pd
# >>> train = pd.read_csv('train.csv')
# >>> trained = train_nhits_storewise(train, use_validation=True)
# >>> test = pd.read_csv('TEST_00.csv')
# >>> pred_df = predict_nhits_storewise(test, trained, test_prefix='TEST_00')
# >>> # 제출 포맷 변환:
# >>> sample = pd.read_csv('sample_submission.csv')
# >>> final = convert_to_submission_format(pred_df, sample)
# >>> final.to_csv('submission_TEST_00.csv', index=False)
#
# Notes
# -----
# - MPS 가속 사용 가능 (Apple Silicon)
# - 손실: Weighted Huber + sMAPE (alpha 가중)
# - 조기 종료(EarlyStopping) & ReduceLROnPlateau
# - 학습 Loss 그래프는 ./loss_plots_store 에 저장
# ===============================================================

import os, re, glob, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# (Optional) 음력 휴일 변환이 필요한 경우
try:
    from korean_lunar_calendar import KoreanLunarCalendar
    _LUNAR_OK = True
except Exception:
    _LUNAR_OK = False


# -------------------------
# 하이퍼파라미터/전역
# -------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

LOOKBACK, PREDICT, BATCH_SIZE, EPOCHS = 28, 7, 64, 40
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

PATIENCE = 10
MIN_SEQUENCE_COUNT = 10

# ---- 휴일 세팅(양력 + 선택적 음력) ----
years_default = [2023, 2024, 2025]

# 고정 양력 공휴일(월,일)
solar_md_holidays = [
    (1, 1), (3, 1), (5, 5), (6, 6), (8, 15), (10, 3), (10, 9), (12, 25)
]

# 음력 주요 명절(설/추석 ±1일) → 양력 변환
def get_lunar_to_solar(years, lunar_month, lunar_day, span=1):
    if not _LUNAR_OK:
        return []
    cal = KoreanLunarCalendar()
    dates = []
    for y in years:
        for offset in range(-span, span+1):
            try:
                cal.setLunar(y, lunar_month, lunar_day + offset, False)
                dates.append(cal.SolarIsoFormat())
            except Exception:
                pass
    return dates

# (필요 시) 음력 변환
lunar_solar_dates = []
lunar_solar_dates += get_lunar_to_solar(years_default, 1, 1, span=1)   # 설 ±1
lunar_solar_dates += get_lunar_to_solar(years_default, 8, 15, span=1)  # 추석 ±1

# ---- 글로벌 휴일 캐시 ----
HOLIDAYS_INT = None   # numpy int64 (days since epoch)

# -------------------------
# 캘린더/피처 유틸
# -------------------------
def generate_combined_holiday_list(df, solar_md_list, lunar_solar_list):
    """양/음력 휴일 병합 → is_holiday=0/1"""
    out = df.copy()
    out['영업일자'] = pd.to_datetime(out['영업일자'])
    out['is_solar_holiday'] = out['영업일자'].apply(lambda x: (x.month, x.day) in solar_md_list)
    lunar_set = set(pd.to_datetime(lunar_solar_list)) if len(lunar_solar_list) else set()
    out['is_lunar_holiday'] = out['영업일자'].isin(lunar_set)
    out['is_holiday'] = (out['is_solar_holiday'] | out['is_lunar_holiday']).astype(int)
    return out.drop(columns=['is_solar_holiday', 'is_lunar_holiday'])

def add_month_idx_features(df: pd.DataFrame, date_col: str = '영업일자') -> pd.DataFrame:
    """기본 달력 피처: weekday/month/season + 연간 주기(sin/cos)"""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    m = out[date_col].dt.month
    doy = out[date_col].dt.dayofyear.astype(np.int16)  # 1..365
    out['doy_sin'] = np.sin(2*np.pi*doy/365).astype('float32')
    out['doy_cos'] = np.cos(2*np.pi*doy/365).astype('float32')
    out['month_idx'] = (m - 1).astype(int)  # 0~11
    out['season'] = m.map({12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}).astype(int)
    out['weekday'] = out[date_col].dt.dayofweek.astype(int)
    return out

def clip_df(series, q=0.9995):
    """상위 극단값 클립 (스케일 안정화)"""
    s = pd.Series(series, copy=False).astype('float32')
    ub = float(np.nanquantile(s, q))
    return s.clip(upper=ub)

def add_holiday_proximity(df: pd.DataFrame, date_col='영업일자', holiday_col='is_holiday', out_col='holiday_prox', K=10) -> pd.DataFrame:
    """휴일 근접도(0~1): 가까울수록 1에 가까움"""
    df = df.copy()
    if date_col not in df or holiday_col not in df:
        return df
    tmp = df[[date_col, holiday_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)
    mask = tmp[holiday_col].astype(bool)
    s_h = tmp[date_col].where(mask)
    prev_h, next_h = s_h.ffill(), s_h.bfill()
    dist_prev = (tmp[date_col] - prev_h).dt.days.astype('float32').fillna(K+1)
    dist_next = (next_h - tmp[date_col]).dt.days.astype('float32').fillna(K+1)
    dist_h = np.minimum(dist_prev, dist_next).clip(0, K).astype('float32')
    out = ((K - dist_h) / K).astype('float32')
    df[out_col] = out.values
    return df

def add_ts_stats(df: pd.DataFrame, target_col="clipped_SQ", date_col="영업일자",
                 lags=(7,14,28), roll_windows=(7,14), ewm_spans=(7,), eps=1e-3) -> pd.DataFrame:
    """지연/이동평균/변동성 등 기본 시계열 통계"""
    df = df.sort_values(date_col).copy()
    x = df[target_col].astype('float32')
    # lag
    for k in lags:
        df[f"lag_{k}"] = x.shift(k).astype('float32')
    # rolling
    for w in roll_windows:
        df[f"roll_mean_{w}"] = x.rolling(w, min_periods=1).mean().astype('float32')
        df[f"roll_std_{w}"]  = x.rolling(w, min_periods=1).std().fillna(0).astype('float32')
    # ewm
    for s in ewm_spans:
        df[f"ewm_mean_{s}"] = x.ewm(span=s, adjust=False).mean().astype('float32')
    # momentum/rel_level/vol
    for k in (7,):
        df[f"momentum_{k}"] = ((x - x.shift(k)) / (np.abs(x.shift(k)) + eps)).astype('float32')
    for w in (7,14):
        m = df[f"roll_mean_{w}"]; s = df[f"roll_std_{w}"]
        df[f"rel_level_{w}"] = (x / (m + eps)).astype('float32')
        df[f"vol_{w}"]       = (s / (m + eps)).astype('float32')
    return df

# --------- 필수 피처 세트 (store_rm7 포함) ---------
FEATURES = [
    'clipped_SQ',        # scaled
    'rolling_mean_7',    # scaled
    'delta_scaled',      # scaled
    'holiday_prox',
    'is_holiday',
    'lag_7', 'lag_14', 'lag_28',
    'rel_level_7',
    'vol_14',
    'momentum_7',
    'ewm_mean_7',
    'doy_sin', 'doy_cos',
    'store_rm7',         # NEW: 업장 총매출 7일 이동평균(스케일된 값)
]
FEAT = {name: i for i, name in enumerate(FEATURES)}

def build_features(df: pd.DataFrame,
                   scaler_y: MinMaxScaler,
                   scaler_rm: MinMaxScaler,
                   scaler_delta: MinMaxScaler,
                   fit: bool = False,
                   date_col: str = '영업일자') -> pd.DataFrame:
    """메뉴별 개별 스케일러로 타깃/롤링/델타 정규화 + 시계열 통계"""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = add_month_idx_features(out, date_col)
    out = generate_combined_holiday_list(out, solar_md_holidays, lunar_solar_dates)
    out = add_holiday_proximity(out, date_col, 'is_holiday', 'holiday_prox', K=10)

    # target clip & deltas
    if '매출수량' in out.columns:
        out['clipped_SQ'] = clip_df(out['매출수량'])
    else:
        # 추론 시 이미 존재할 수 있음
        out['clipped_SQ'] = out.get('clipped_SQ', 0.0)

    out['delta']          = out['clipped_SQ'].diff().fillna(0)
    out['rolling_mean_7'] = out['clipped_SQ'].rolling(window=7, min_periods=1).mean()

    # scaling (메뉴별 개별 스케일러)
    if fit:
        scaler_y.fit(out[['clipped_SQ']])
        scaler_rm.fit(out[['rolling_mean_7']])
        scaler_delta.fit(out[['delta']])

    out[['clipped_SQ']]     = scaler_y.transform(out[['clipped_SQ']])
    out[['rolling_mean_7']] = scaler_rm.transform(out[['rolling_mean_7']])
    out[['delta_scaled']]   = scaler_delta.transform(out[['delta']])

    # 시계열 통계 (과거 기반)
    out = add_ts_stats(out, target_col="clipped_SQ", date_col=date_col,
                       lags=(7,14,28), roll_windows=(7,14), ewm_spans=(7,))

    # 결측/타입 정리
    use_cols = [c for c in FEATURES if c in out.columns]
    for c in use_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype('float32')

    out['is_holiday'] = out['is_holiday'].astype('float32')
    return out

# -------------------------
# 미래 달력(휴일 근접) 유틸
# -------------------------
def build_holidays_int(years):
    """휴일 목록을 'epoch days(int64)'로 만들어 고속 근접도 계산에 사용"""
    dates = []
    for y in years:
        for (m, d) in solar_md_holidays:
            try:
                dates.append(pd.Timestamp(year=y, month=m, day=d))
            except Exception:
                pass
    lunar_list = pd.to_datetime(lunar_solar_dates, errors='coerce')
    holidays = pd.to_datetime(pd.Index(dates).append(pd.Index(lunar_list)), errors='coerce').dropna().unique()
    holidays = pd.DatetimeIndex(holidays).tz_localize(None)
    ns = holidays.to_numpy(dtype='datetime64[ns]')
    days = (ns - np.datetime64('1970-01-01', 'ns')) // np.timedelta64(1, 'D')
    return days.astype('int64')

def holiday_prox_searchsorted(dates_pd_index, holidays_int, K=10):
    """날짜 배열과 휴일-정수배열로 근접도(0~1) 계산"""
    d_int = pd.to_datetime(dates_pd_index).values.astype('datetime64[D]').astype('int64')
    idx = np.searchsorted(holidays_int, d_int, side='left')
    M = holidays_int.size
    prev_idx = np.clip(idx - 1, 0, max(M - 1, 0))
    next_idx = np.clip(idx,       0, max(M - 1, 0))
    dist_prev = d_int - holidays_int[prev_idx]
    dist_next = holidays_int[next_idx] - d_int
    dist_prev = np.where(idx > 0, dist_prev, K + 1)
    dist_next = np.where(idx < M, dist_next, K + 1)
    mind = np.minimum(dist_prev, dist_next)
    mind = np.clip(mind, 0, K)
    prox = (K - mind) / float(K)
    return prox.astype(np.float32)

def make_future_calendar_fast(last_dates_np, horizon, holidays_int, K=10):
    """각 시퀀스의 마지막 날짜로부터 H-step 미래 달력(wd, month, prox) 생성"""
    S = len(last_dates_np)
    base = pd.to_datetime(last_dates_np).values.astype('datetime64[D]')
    fut = base[:, None] + np.arange(1, horizon + 1, dtype='timedelta64[D]')[None, :]
    fut_pd = pd.to_datetime(fut.reshape(-1))
    wd = fut_pd.dayofweek.values.reshape(S, horizon).astype(np.int64)
    mm = (fut_pd.month.values.reshape(S, horizon) - 1).astype(np.int64)  # 0~11
    prox_flat = holiday_prox_searchsorted(fut_pd, holidays_int, K=K)      # (S*H,)
    px = prox_flat.reshape(S, horizon)
    return wd, mm, px

# -------------------------
# 손실/조기 종료/시각화
# -------------------------
def estimate_delta_from_y(y_real, clip_min: float = 10.0, clip_max: float = 80.0) -> float:
    """Huber delta 추정 (데이터 중위수/MAD 기반; 안정화)"""
    y = np.asarray(y_real, dtype=float); y = y[np.isfinite(y)]
    if y.size == 0:
        return float((clip_min + clip_max) / 2.0)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    mad = max(mad, 1e-6)
    delta = 1.35 * mad
    return float(np.clip(delta, clip_min, clip_max))

def weighted_huber_smape(yhat, y, w=None, *, delta=0.05, eps=1e-3, alpha=0.5):
    """Huber(alpha) + sMAPE(1-alpha) 혼합 손실. eps는 sMAPE 분모 안정화 용도"""
    hub = F.huber_loss(yhat, y, delta=delta, reduction='none')
    den = (yhat.abs() + y.abs()).clamp_min(float(eps)) if isinstance(eps, (float,int)) else torch.maximum(yhat.abs()+y.abs(), eps)
    smp = 2.0 * (yhat - y).abs() / den
    loss = alpha * hub + (1 - alpha) * smp
    if w is not None:
        loss = loss * w
    return loss.mean()

class EarlyStopping:
    """최고 성능 가중치 복원 옵션이 있는 표준 조기 종료"""
    def __init__(self, patience=10, min_delta=0.0, mode='min',
                 restore_best_weights=True, min_epochs=0, relative=True, smooth_beta=0.0):
        self.patience = patience; self.min_delta = min_delta; self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs; self.relative = relative; self.smooth_beta = smooth_beta
        self.best = None; self.best_state = None; self.wait = 0; self.stop = False; self.ema = None

        # 추가: 베스트 epoch 기록
        self.best_epoch = -1

    def _improved(self, current):
        if self.best is None: return True
        if self.mode == 'min':
            return current < (self.best * (1.0 - self.min_delta) if self.relative else (self.best - self.min_delta))
        else:
            return current > (self.best * (1.0 + self.min_delta) if self.relative else (self.best + self.min_delta))

    def step(self, current, model, epoch_idx: int):
        monitor_val = current if self.smooth_beta == 0 else (self.smooth_beta*(self.ema or current) + (1-self.smooth_beta)*current)
        self.ema = monitor_val
        if self.best is None or self._improved(monitor_val):
            self.best = monitor_val; self.best_state = deepcopy(model.state_dict()); self.wait = 0; self.best_epoch = epoch_idx
            return False
        self.wait += 1
        if epoch_idx + 1 < self.min_epochs: return False
        if self.wait >= self.patience:
            self.stop = True
            if self.restore_best_weights and self.best_state is not None:
                model.load_state_dict(self.best_state)
            return True
        return False

def visualize_loss(train_losses, val_losses, title, save=True, out_dir="./loss_plots_store"):
    """에폭별 train/val loss 그래프를 파일로 저장"""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6,4)); ax = plt.gca()

    def _to_arr(xs):
        import numpy as np
        try: return np.asarray(xs, dtype=float)
        except Exception: return np.array(xs, dtype=float)

    tr = _to_arr(train_losses); ax.plot(range(1, len(tr)+1), tr, marker='o', lw=1.5, label='Train Loss')
    if val_losses:
        va = _to_arr(val_losses); ax.plot(range(1, len(va)+1), va, marker='o', lw=1.5, label='Validation Loss')
    ax.set_title(str(title)); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.4); ax.legend(loc='upper right')
    safe = re.sub(r'[^\\w\\-_.]', '_', str(title))
    path = os.path.join(out_dir, f"{safe}.png")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# -------------------------
# 업장 컨텍스트: store_rm7
# -------------------------
def make_store_context(df_store: pd.DataFrame) -> pd.DataFrame:
    """업장-날짜 단위 총매출 → 7일 이동평균(store_rm7)"""
    g = df_store.copy()
    g['영업일자'] = pd.to_datetime(g['영업일자'])
    agg = (g.groupby('영업일자', as_index=False)['매출수량'].sum()
             .sort_values('영업일자'))
    agg['store_rm7'] = agg['매출수량'].rolling(7, min_periods=1).mean().astype('float32')
    return agg[['영업일자','store_rm7']]

def _store_from_key(key: str) -> str:
    """'영업장명_메뉴명' → '영업장명'"""
    return str(key).split('_', 1)[0]

# -------------------------
# 모델 정의 (메뉴 임베딩 포함)
# -------------------------
class MRBlock(nn.Module):
    """간단 멀티리졸루션 블록: 풀링 → MLP → 전역 pooling(mean/max) → horizon 투사"""
    def __init__(self, in_dim, horizon, pool_k: int, mlp_dim=128, mlp_layers=2):
        super().__init__()
        self.pool_k = pool_k
        layers = [nn.Linear(in_dim, mlp_dim), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_dim, mlp_dim), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(mlp_dim)
        self.drop = nn.Dropout(p=0.1)
        self.proj = nn.Linear(mlp_dim * 2, horizon)

    def forward(self, x):  # x: (B,T,F)
        x_ch = x.transpose(1, 2)  # (B,F,T)
        x_pool = F.avg_pool1d(x_ch, kernel_size=self.pool_k, stride=self.pool_k, ceil_mode=True) if self.pool_k > 1 else x_ch
        x_pool = x_pool.transpose(1, 2)  # (B,T',F)
        z = self.mlp(x_pool)             # (B,T',D)
        z = self.norm(z); z = self.drop(z)
        z_mean, z_max = z.mean(dim=1), z.amax(dim=1)
        z_cat = torch.cat([z_mean, z_max], dim=-1)  # (B,2D)
        return self.proj(z_cat)                     # (B,H)

class NHiTSWithEmbeddingMR(nn.Module):
    """
    N-HiTS 변형: 입력(연속피처 + 달력/메뉴 임베딩) → MRBlocks → horizon-wise head
    - item(menu) 임베딩을 추가하여 업장 내 메뉴 간 파라미터 공유 + 개별성 반영
    """
    def __init__(self, lookback, input_dim, horizon,
                 pools=(28,7,1),
                 weekday_vocab=7, weekday_emb_dim=2,
                 season_vocab=4,  season_emb_dim=2,
                 month_vocab=12,  month_emb_dim=3,
                 item_vocab=None, item_emb_dim=4,
                 emb_dropout=0.25, use_sigmoid_output=False,
                 mr_mlp_dim=96):
        super().__init__()
        self.lookback, self.horizon = lookback, horizon
        self.use_sigmoid_output = use_sigmoid_output
        self.use_residual_base = True
        self.r_scale_param = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))

        # 임베딩
        self.weekday_emb = nn.Embedding(weekday_vocab, weekday_emb_dim)
        self.season_emb  = nn.Embedding(season_vocab,  season_emb_dim)
        self.month_emb   = nn.Embedding(month_vocab,   month_emb_dim)

        self.item_emb = None
        cat_dim = weekday_emb_dim + season_emb_dim + month_emb_dim
        if item_vocab is not None:
            self.item_emb = nn.Embedding(item_vocab, item_emb_dim)
            cat_dim += item_emb_dim

        self.post_emb_norm = nn.LayerNorm(input_dim + cat_dim)
        self.post_emb_drop = nn.Dropout(emb_dropout)

        # MR Blocks
        self.blocks = nn.ModuleList([
            MRBlock(in_dim=input_dim + cat_dim, horizon=horizon, pool_k=p, mlp_dim=mr_mlp_dim, mlp_layers=2)
            for p in pools
        ])

        # Horizon 위치 임베딩 + 미래 달력 임베딩
        self.h_emb = nn.Embedding(horizon, 8)
        self.wd_future_emb = nn.Embedding(7, 3)
        self.mm_future_emb = nn.Embedding(12, 2)
        self.prox_lin = nn.Linear(1, 2)   # future_prox → 2D

        # Horizon-wise head
        aux_dim = 8 + 3 + 2 + 2  # =15
        in_dim = 1 + aux_dim     # rep_exp(1) + aux
        self.head = nn.Sequential(nn.Linear(in_dim,128), nn.GELU(), nn.Linear(128,1))
        self.ratio_head = nn.Sequential(nn.Linear(in_dim,64), nn.GELU(), nn.Linear(64,1))

    def forward(self, x_num, x_weekday, x_season, x_month,
                future_weekday=None, future_month=None, future_prox=None,
                base_last=None, x_item_id=None):
        """
        x_num:    (B,T,F) 연속 피처(스케일된 값 포함)
        x_*:      (B,T)   달력 인덱스
        x_item_id:(B,) or (B,T) 메뉴 id
        future_*: (B,H)   H-step 미래 달력 인덱스/근접도
        base_last:(B,) or (B,H) 베이스라인(rolling_mean_7) 마지막 스텝
        """
        # 임베딩 결합
        w = self.weekday_emb(x_weekday)
        s = self.season_emb(x_season)
        m = self.month_emb(x_month)

        if (self.item_emb is not None) and (x_item_id is not None):
            if x_item_id.dim() == 1:
                x_item_id = x_item_id[:, None].expand(-1, x_num.size(1))
            it = self.item_emb(x_item_id)
            x = torch.cat([x_num, w, s, m, it], dim=-1)
        else:
            x = torch.cat([x_num, w, s, m], dim=-1)

        x = self.post_emb_norm(x)
        x = self.post_emb_drop(x)

        reps = [b(x) for b in self.blocks]       # list of (B,H)
        rep = torch.stack(reps, dim=0).mean(dim=0)  # (B,H)

        B, H = rep.size(); device = rep.device
        if future_weekday is None:
            future_weekday = (x_weekday[:, -1].unsqueeze(1) + torch.arange(1, H+1, device=device)) % 7
        if future_month is None:
            future_month = x_month[:, -1].unsqueeze(1).expand(B, H)
        if future_prox is None:
            future_prox = torch.zeros(B, H, device=device)

        he = self.h_emb(torch.arange(H, device=device).unsqueeze(0).expand(B, -1))
        we = self.wd_future_emb(future_weekday)
        me = self.mm_future_emb(future_month)
        pe = self.prox_lin(future_prox.unsqueeze(-1))
        aux = torch.cat([he, we, me, pe], dim=-1)  # (B,H,15)

        rep_exp = rep.unsqueeze(-1)                # (B,H,1)
        feat = torch.cat([rep_exp, aux], dim=-1)   # (B,H,16)

        r_scale = F.softplus(self.r_scale_param)
        main = self.head(feat).squeeze(-1)         # (B,H)
        if self.use_sigmoid_output:
            main = torch.sigmoid(main)

        if self.use_residual_base:
            base = x_num[:, -1, FEAT['rolling_mean_7']] if base_last is None else base_last
            if base.dim() == 1:
                base = base.unsqueeze(1).expand(B, H)
            ratio = torch.tanh(self.ratio_head(feat).squeeze(-1))  # (-1,1)
            out = main + base * (r_scale * ratio)
        else:
            out = main
        return out

# -------------------------
# 학습(업장 단위)
# -------------------------
def train_nhits_storewise(train_df: pd.DataFrame,
                          use_validation: bool = True,
                          lr: float = 1e-3,
                          weight_decay: float = 1e-5,
                          max_grad_norm: float = 1.0,
                          emb_dropout: float = 0.25,
                          mr_mlp_dim: int = 96,
                          plot_dir: str = "./loss_plots_store"):
    """
    업장 단위 글로벌 학습:
    - 업장 내 모든 메뉴의 윈도우를 하나로 합쳐 공동 학습
    - 메뉴별 스케일러(타깃/롤링/델타) + 업장 컨텍스트 스케일러를 보관
    - 메뉴 임베딩으로 고유성 반영
    """
    trained = {}

    df = train_df.copy()
    if '영업장명' not in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].apply(_store_from_key)
    df['영업일자'] = pd.to_datetime(df['영업일자'])

    global HOLIDAYS_INT
    if HOLIDAYS_INT is None:
        years = df['영업일자'].dt.year
        HOLIDAYS_INT = build_holidays_int(range(int(years.min()) - 1, int(years.max()) + 2))

    for store, g_store in tqdm(df.groupby('영업장명'), desc='Training (store-wise)'):
        g_store = g_store.sort_values('영업일자')
        menus = sorted(g_store['영업장명_메뉴명'].unique())
        item2id = {m: i for i, m in enumerate(menus)}

        # 업장 컨텍스트 + 스케일러
        scaler_store_rm = MinMaxScaler()
        store_ctx_full = make_store_context(g_store)
        scaler_store_rm.fit(store_ctx_full[['store_rm7']])
        store_ctx_full[['store_rm7']] = scaler_store_rm.transform(store_ctx_full[['store_rm7']])

        # 수집 버퍼
        X_list, y_list = [], []
        wd_list, ss_list, mm_list = [], [], []
        it_list, last_date_list = [], []
        scaler_pack_per_item = {}
        last_seq_per_item = {}
        lower_bounds = {}

        # 메뉴별 윈도우 생성
        for menu, g_menu in g_store.groupby('영업장명_메뉴명'):
            g_menu = g_menu.sort_values('영업일자')
            if len(g_menu) < LOOKBACK + PREDICT + MIN_SEQUENCE_COUNT:
                continue

            scaler_y, scaler_rm, scaler_delta = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()

            # 피처 빌드(+ 개별 스케일러 학습)
            feat = build_features(g_menu, scaler_y, scaler_rm, scaler_delta, fit=True, date_col='영업일자')
            # 업장 컨텍스트 병합
            feat = feat.merge(store_ctx_full, on='영업일자', how='left')
            feat['store_rm7'] = feat['store_rm7'].fillna(0.0).astype('float32')

            vals = feat[FEATURES].values.astype(np.float32)
            tgt  = feat['clipped_SQ'].values.astype(np.float32)
            wd   = feat['weekday'].values.astype(np.int64)
            ss   = feat['season'].values.astype(np.int64)
            mm   = feat['month_idx'].values.astype(np.int64)

            total_seq = len(feat) - LOOKBACK - PREDICT + 1
            if total_seq <= 0:
                continue

            X_np  = sliding_window_view(vals, LOOKBACK, axis=0)[:total_seq]
            y_np  = sliding_window_view(tgt,  LOOKBACK+PREDICT, axis=0)[:total_seq, LOOKBACK:]
            wd_np = sliding_window_view(wd,   LOOKBACK, axis=0)[:total_seq]
            ss_np = sliding_window_view(ss,   LOOKBACK, axis=0)[:total_seq]
            mm_np = sliding_window_view(mm,   LOOKBACK, axis=0)[:total_seq]

            X_list.append(X_np); y_list.append(y_np)
            wd_list.append(wd_np); ss_list.append(ss_np); mm_list.append(mm_np)
            it_list.append(np.full((total_seq,), item2id[menu], dtype=np.int64))

            dates_np = feat['영업일자'].values.astype('datetime64[ns]')
            last_dates = pd.to_datetime(dates_np[LOOKBACK - 1 : LOOKBACK - 1 + total_seq])
            last_date_list.append(last_dates.values)

            # 추론용 캐시
            scaler_pack_per_item[menu] = (scaler_y, scaler_rm, scaler_delta)

            # 마지막 LOOKBACK 윈도우 저장(테스트 구간이 짧을 때 사용)
            last_seq_per_item[menu] = {
                'X_num': feat[FEATURES].values[-LOOKBACK:],
                'weekday': feat['weekday'].values[-LOOKBACK:],
                'season':  feat['season'].values[-LOOKBACK:],
                'month_idx': feat['month_idx'].values[-LOOKBACK:]
            }

            # 메뉴별 하한값(1% 분위수, 최소 1.0)
            lb = max(np.quantile(g_menu['매출수량'].values, 0.01), 1.0) if '매출수량' in g_menu.columns else 1.0
            lower_bounds[menu] = float(lb)

        if not X_list:
            # 학습 가능한 윈도우가 없음
            continue

        # 병합
        X_all  = np.concatenate(X_list, axis=0)
        y_all  = np.concatenate(y_list, axis=0)
        wd_all = np.concatenate(wd_list, axis=0)
        ss_all = np.concatenate(ss_list, axis=0)
        mm_all = np.concatenate(mm_list, axis=0)
        it_all = np.concatenate(it_list, axis=0)
        last_dates_all = np.concatenate(last_date_list, axis=0)

        # 미래 달력 사전계산
        fwd_np, fmm_np, fpx_np = make_future_calendar_fast(last_dates_all, PREDICT, HOLIDAYS_INT, K=10)

        # 텐서화
        X = torch.tensor(X_all,  dtype=torch.float32, device=DEVICE)
        y = torch.tensor(y_all,  dtype=torch.float32, device=DEVICE)
        wd= torch.tensor(wd_all, dtype=torch.long,    device=DEVICE)
        ss= torch.tensor(ss_all, dtype=torch.long,    device=DEVICE)
        mm= torch.tensor(mm_all, dtype=torch.long,    device=DEVICE)
        it= torch.tensor(it_all, dtype=torch.long,    device=DEVICE)
        fwd = torch.tensor(fwd_np, dtype=torch.long,  device=DEVICE)
        fmm = torch.tensor(fmm_np, dtype=torch.long,  device=DEVICE)
        fpx = torch.tensor(fpx_np, dtype=torch.float32, device=DEVICE)

        # split
        split = int(len(X) * 0.8) if use_validation else len(X)
        Xtr, Xval = X[:split], X[split:]
        ytr, yval = y[:split], y[split:]
        wdtr, wdval = wd[:split], wd[split:]
        sstr, ssval = ss[:split], ss[split:]
        mmtr, mmval = mm[:split], mm[split:]
        ittr, itval = it[:split], it[split:]
        fwd_tr, fwd_val = fwd[:split], fwd[split:]
        fmm_tr, fmm_val = fmm[:split], fmm[split:]
        fpx_tr, fpx_val = fpx[:split], fpx[split:]

        # 모델
        model = NHiTSWithEmbeddingMR(
            lookback=LOOKBACK, input_dim=len(FEATURES), horizon=PREDICT,
            pools=(28,7,3,1),
            item_vocab=len(menus), item_emb_dim=4,
            emb_dropout=emb_dropout, use_sigmoid_output=False,
            mr_mlp_dim=mr_mlp_dim
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
        early = EarlyStopping(patience=PATIENCE, min_epochs=8, restore_best_weights=True)

        # Huber delta: 업장 전체 분포 기반
        delta_y = estimate_delta_from_y(g_store['매출수량'].values, clip_min=10.0, clip_max=80.0)

        train_losses, val_losses = [], []
        for ep in range(EPOCHS):
            model.train()
            idx = torch.randperm(len(Xtr), device=DEVICE)
            sum_loss = 0.0; n_obs = 0
            for i in range(0, len(Xtr), BATCH_SIZE):
                b = idx[i:i+BATCH_SIZE]
                Xb, yb = Xtr[b], ytr[b]
                pred = model(Xb, wdtr[b], sstr[b], mmtr[b],
                             future_weekday=fwd_tr[b], future_month=fmm_tr[b], future_prox=fpx_tr[b],
                             base_last=Xb[:, -1, FEAT['rolling_mean_7']], x_item_id=ittr[b])
                eps_dyn = (0.25 * Xb[:, -1, FEAT['rolling_mean_7']]).unsqueeze(1).expand_as(pred).clamp(5e-4, 2e-2)
                loss = weighted_huber_smape(pred, yb, w=None, delta=delta_y, eps=eps_dyn, alpha=0.7)
                opt.zero_grad(set_to_none=True); loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                opt.step()
                sum_loss += loss.item() * yb.size(0); n_obs += yb.size(0)
            train_losses.append(sum_loss / max(n_obs, 1))

            if use_validation and len(Xval) > 0:
                model.eval()
                with torch.no_grad():
                    pred_v = model(Xval, wdval, ssval, mmval,
                                   future_weekday=fwd_val, future_month=fmm_val, future_prox=fpx_val,
                                   base_last=Xval[:, -1, FEAT['rolling_mean_7']], x_item_id=itval)
                    eps_dyn_v = (0.25 * Xval[:, -1, FEAT['rolling_mean_7']]).unsqueeze(1).expand_as(pred_v).clamp(5e-4, 2e-2)
                    v = weighted_huber_smape(pred_v, yval, w=None, delta=delta_y, eps=eps_dyn_v, alpha=0.7).item()
                    val_losses.append(v); sch.step(v)
                    if early.step(v, model, ep):
                        # print(f"[{store}] Early stop @ {ep+1} | best val={min(val_losses):.6f}")
                        break

        # 저장 (업장 단위 패키지)
        trained[store] = {
            'model': model.eval(),
            'menus': menus,
            'item2id': item2id,
            'scalers': scaler_pack_per_item,       # 메뉴별 (scaler_y, scaler_rm, scaler_delta)
            'store_ctx_scaler': scaler_store_rm,   # 업장 컨텍스트 스케일러
            'last_sequences': last_seq_per_item,   # 메뉴별 마지막 LOOKBACK 시퀀스
            'lower_bounds': lower_bounds,          # 메뉴별 하한값
            'feature_order': FEATURES,
        }

        visualize_loss(train_losses, val_losses if use_validation else None,
                       f"{store}_STOREWISE", save=True, out_dir=plot_dir)

    return trained

# -------------------------
# 추론(업장 단위)
# -------------------------
def inverse_clipped_from_scaler(scaler_xy: MinMaxScaler, scaled_vals: np.ndarray) -> np.ndarray:
    """MinMaxScaler(in)으로 스케일한 'clipped_SQ' 역변환"""
    dummy = np.zeros((len(scaled_vals), 1), dtype=np.float32)
    dummy[:, 0] = scaled_vals
    # 학습 시 y만 fit 했으므로 1-column inverse가 안전
    return scaler_xy.inverse_transform(dummy)[:, 0]

def predict_nhits_storewise(test_df: pd.DataFrame,
                            trained: dict,
                            test_prefix: str,
                            discontinued: dict | None = None,
                            rule: str = 'after',   # 'after' or 'on_or_after'
                            grace_days: int = 0):
    """
    업장 단위 글로벌 모델로 예측.
    - test_df: 예측 대상 28일 히스토리 포함(매출수량 유효) → 다음 7일 예측
    - trained: train_nhits_storewise() 결과
    - discontinued: {'업장_메뉴': 'YYYY-MM-DD', ...} 단종 일자(선택)
    """
    results = []
    df = test_df.copy()
    if '영업장명' not in df.columns:
        df['영업장명'] = df['영업장명_메뉴명'].apply(_store_from_key)
    df['영업일자'] = pd.to_datetime(df['영업일자'])

    cut_map = None
    if discontinued is not None:
        cut_map = {k: (v if isinstance(v, pd.Timestamp) else pd.to_datetime(v)) for k, v in discontinued.items()}
        if grace_days != 0:
            for k in cut_map:
                cut_map[k] = cut_map[k] + pd.Timedelta(days=grace_days)

    global HOLIDAYS_INT
    if HOLIDAYS_INT is None:
        years = df['영업일자'].dt.year
        HOLIDAYS_INT = build_holidays_int(range(int(years.min()) - 1, int(years.max()) + 2))

    for store, g_store in df.groupby('영업장명', sort=False):
        if store not in trained:
            continue
        pack = trained[store]
        model = pack['model']
        item2id = pack['item2id']
        menus = pack['menus']
        scaler_store_rm = pack['store_ctx_scaler']

        # 업장 컨텍스트(테스트 28일 히스토리로 계산 → 스케일)
        store_ctx = make_store_context(g_store)
        store_ctx[['store_rm7']] = scaler_store_rm.transform(store_ctx[['store_rm7']])

        # 업장 마지막 관측일
        last_obs_date = pd.to_datetime(g_store['영업일자'].max())

        # 공통: 미래 달력 텐서(업장 단위 동일)
        horizon_dates = pd.date_range(start=last_obs_date + pd.Timedelta(days=1), periods=PREDICT, freq='D')
        future_wd_np = np.array([d.dayofweek for d in horizon_dates], dtype=np.int64)[None, :]   # (1,H)
        future_mm_np = np.array([d.month-1 for d in horizon_dates], dtype=np.int64)[None, :]     # (1,H)
        future_px_np = holiday_prox_searchsorted(horizon_dates, HOLIDAYS_INT, K=10)[None, :]     # (1,H)

        future_wd = torch.tensor(future_wd_np, dtype=torch.long,   device=DEVICE)
        future_mm = torch.tensor(future_mm_np, dtype=torch.long,   device=DEVICE)
        future_px = torch.tensor(future_px_np, dtype=torch.float32, device=DEVICE)

        for menu, g_menu in g_store.groupby('영업장명_메뉴명', sort=False):
            if menu not in pack['scalers']:
                continue
            scaler_y, scaler_rm, scaler_delta = pack['scalers'][menu]
            lower_bound = pack['lower_bounds'].get(menu, 1.0)

            st = g_menu.sort_values('영업일자').copy()
            st['영업일자'] = pd.to_datetime(st['영업일자'])

            # 피처 빌드(스케일러 사용) + 업장 컨텍스트 병합
            ft = build_features(st, scaler_y, scaler_rm, scaler_delta, fit=False, date_col='영업일자')
            ft = ft.merge(store_ctx, on='영업일자', how='left')
            ft['store_rm7'] = ft['store_rm7'].fillna(0.0).astype('float32')

            # 최근 LOOKBACK 시퀀스
            if len(ft) < LOOKBACK:
                # 학습 때 저장해둔 마지막 윈도우 사용
                last_seq = pack['last_sequences'].get(menu, None)
                if last_seq is None:
                    continue
                x_num_np  = np.asarray(last_seq['X_num'], dtype=np.float32)
                weekday_np= np.asarray(last_seq['weekday'], dtype=np.int64)
                season_np = np.asarray(last_seq['season'],  dtype=np.int64)
                month_np  = np.asarray(last_seq['month_idx'], dtype=np.int64)
            else:
                recent = ft.iloc[-LOOKBACK:].copy()
                x_num_np  = recent[FEATURES].values.astype(np.float32)
                weekday_np= recent['weekday'].values.astype(np.int64)
                season_np = recent['season'].values.astype(np.int64)
                month_np  = recent['month_idx'].values.astype(np.int64)

            # 텐서 구성
            x_num = torch.tensor(x_num_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)   # (1,T,F)
            wd    = torch.tensor(weekday_np, dtype=torch.long,   device=DEVICE).unsqueeze(0)  # (1,T)
            ss    = torch.tensor(season_np,  dtype=torch.long,   device=DEVICE).unsqueeze(0)  # (1,T)
            mm    = torch.tensor(month_np,   dtype=torch.long,   device=DEVICE).unsqueeze(0)  # (1,T)
            item_id = torch.tensor([item2id[menu]], dtype=torch.long, device=DEVICE)          # (1,)

            model.eval()
            with torch.no_grad():
                pred_scaled = model(x_num, wd, ss, mm,
                                    future_weekday=future_wd, future_month=future_mm, future_prox=future_px,
                                    base_last=x_num[:, -1, FEAT['rolling_mean_7']],
                                    x_item_id=item_id).squeeze(0).detach().cpu().numpy()

            vals_real = inverse_clipped_from_scaler(scaler_y, pred_scaled)
            vals_real = np.maximum(vals_real, lower_bound)

            # 단종 처리
            if cut_map is not None and menu in cut_map:
                cutoff = cut_map[menu]
                zero_mask = (horizon_dates >= cutoff) if rule == 'on_or_after' else (horizon_dates > cutoff)
                vals_real = np.where(zero_mask, 0.0, vals_real)

            # 제출 포맷용 날짜 라벨
            pred_dates = [f"{test_prefix}+{i+1}일" for i in range(PREDICT)]
            for d, v in zip(pred_dates, vals_real):
                results.append({'영업일자': d, '영업장명_메뉴명': menu, '매출수량': float(v)})

    return pd.DataFrame(results, columns=['영업일자','영업장명_메뉴명','매출수량'])

# -------------------------
# 제출 포맷 변환
# -------------------------
def convert_to_submission_format(pred_df: pd.DataFrame, sample_submission: pd.DataFrame):
    """
    sample_submission의 (영업일자, 컬럼) 구조에 pred_df 값을 채워넣음.
    pred_df: ['영업일자','영업장명_메뉴명','매출수량']
    """
    pred_dict = dict(zip(zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']), pred_df['매출수량']))
    final_df = sample_submission.copy()
    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)
    return final_df
