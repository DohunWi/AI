# ========================================
# 수정된 고급 시계열 예측 모델 - 오류 해결
# str.split() 오류 및 기타 호환성 문제 수정
# ========================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import glob
import os
import warnings
from tqdm import tqdm
import logging
from datetime import datetime, timedelta
import pickle
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedAdvancedForecastingModel:
    """수정된 고급 시계열 예측 모델 - 오류 해결 버전"""
    
    def __init__(self, sequence_length=28, prediction_length=7, device='cpu'):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.device = torch.device(device)
        
        # 모델들
        self.models = {}
        self.scalers = {}
        self.venue_profiles = {}
        self.feature_columns = []
        
        # 하이퍼파라미터 (보수적으로 설정)
        self.hyperparams = {
            'lstm_hidden_dim': 64,  # 메모리 절약
            'lstm_num_layers': 2,   # 레이어 수 감소
            'transformer_d_model': 64,  # 모델 크기 감소
            'transformer_nhead': 4,     # 헤드 수 감소
            'transformer_num_layers': 2, # 레이어 수 감소
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 16,  # 배치 크기 감소
            'epochs': 50,      # 에폭 수 감소
            'patience': 10     # 패션스 감소
        }
    
    # ==========================================
    # 1. 수정된 특성 엔지니어링 (오류 해결)
    # ==========================================
    
    def create_safe_features(self, df, is_training=True):
        """안전한 특성 생성 - str.split() 오류 해결"""
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['영업일자'])
        
        # 안전한 venue, menu 분리
        try:
            # 방법 1: str.split()의 올바른 사용
            venue_menu_split = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)
            df['venue'] = venue_menu_split[0] if len(venue_menu_split.columns) > 0 else ''
            df['menu'] = venue_menu_split[1] if len(venue_menu_split.columns) > 1 else ''
            
        except Exception as e:
            logger.warning(f"venue/menu 분리 중 오류: {e}")
            # 폴백: 수동 분리
            df['venue'] = df['영업장명_메뉴명'].apply(lambda x: x.split('_')[0] if '_' in str(x) else str(x))
            df['menu'] = df['영업장명_메뉴명'].apply(lambda x: '_'.join(x.split('_')[1:]) if '_' in str(x) else '')
        
        logger.info("기본 특성 생성 시작...")
        
        # 1. 시간 기반 특성 (안전하게)
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # 2. 순환 특성 (중요!)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 3. 계절성 특성
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        df['is_monday'] = (df['dayofweek'] == 0).astype(int)
        df['is_holiday_season'] = df['month'].isin([7, 8, 12, 1]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_month_start'] = (df['day'] <= 5).astype(int)
        df['is_month_end'] = (df['day'] >= 25).astype(int)
        
        # 4. 업장 특성
        df['is_premium_venue'] = df['venue'].isin(['담하', '미라시아']).astype(int)
        
        # 안전한 venue 인코딩
        try:
            df['venue_encoded'] = pd.Categorical(df['venue']).codes
        except:
            df['venue_encoded'] = 0
        
        # 5. 메뉴 특성 (안전하게)
        menu_str = df['menu'].fillna('').astype(str)
        df['is_coffee'] = menu_str.str.contains('커피|Coffee|coffee', case=False, na=False).astype(int)
        df['is_alcohol'] = menu_str.str.contains('맥주|소주|와인|술', case=False, na=False).astype(int)
        df['is_main_dish'] = menu_str.str.contains('불고기|갈비|찌개|국밥|정식', case=False, na=False).astype(int)
        df['is_group_menu'] = menu_str.str.contains('단체', case=False, na=False).astype(int)
        df['is_brunch'] = menu_str.str.contains('브런치', case=False, na=False).astype(int)
        df['is_premium_menu'] = menu_str.str.contains('한우|프리미엄|AUS', case=False, na=False).astype(int)
        
        # 6. 업장별 프로파일 (훈련시에만)
        if is_training:
            self._safe_calculate_venue_profiles(df)
        
        # 7. 업장별 통계 특성
        df['venue_avg_sales'] = df['venue'].map(
            lambda x: self.venue_profiles.get(x, {}).get('avg_sales', 0)
        )
        df['venue_std_sales'] = df['venue'].map(
            lambda x: self.venue_profiles.get(x, {}).get('std_sales', 0)
        )
        df['venue_zero_ratio'] = df['venue'].map(
            lambda x: self.venue_profiles.get(x, {}).get('zero_ratio', 0)
        )
        
        # NaN 값 처리
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"특성 생성 완료. 총 특성 수: {len([col for col in df.columns if col not in ['date', 'venue', 'menu', '영업일자', '영업장명_메뉴명', '매출수량']])}")
        
        return df
    
    def _safe_calculate_venue_profiles(self, df):
        """안전한 업장별 프로파일 계산"""
        
        logger.info("업장별 프로파일 계산 중...")
        
        try:
            for venue in df['venue'].unique():
                if pd.isna(venue) or venue == '':
                    continue
                    
                venue_data = df[df['venue'] == venue]
                sales_data = venue_data['매출수량']
                
                # 기본 통계 (안전하게)
                profile = {
                    'avg_sales': float(sales_data.mean()) if len(sales_data) > 0 else 0,
                    'median_sales': float(sales_data.median()) if len(sales_data) > 0 else 0,
                    'std_sales': float(sales_data.std()) if len(sales_data) > 1 else 0,
                    'zero_ratio': float((sales_data == 0).mean()) if len(sales_data) > 0 else 0,
                    'q25': float(sales_data.quantile(0.25)) if len(sales_data) > 0 else 0,
                    'q75': float(sales_data.quantile(0.75)) if len(sales_data) > 0 else 0,
                    'q95': float(sales_data.quantile(0.95)) if len(sales_data) > 0 else 0,
                    'max_sales': float(sales_data.max()) if len(sales_data) > 0 else 0
                }
                
                # 요일별 패턴 (안전하게)
                try:
                    venue_data_copy = venue_data.copy()
                    venue_data_copy['dow'] = venue_data_copy['date'].dt.dayofweek
                    
                    dow_pattern = venue_data_copy.groupby('dow')['매출수량'].mean().to_dict()
                    profile['dow_pattern'] = dow_pattern
                    
                    # 주말 효과
                    weekend_data = venue_data_copy[venue_data_copy['dow'].isin([5, 6])]['매출수량']
                    weekday_data = venue_data_copy[~venue_data_copy['dow'].isin([5, 6])]['매출수량']
                    
                    if len(weekend_data) > 0 and len(weekday_data) > 0:
                        weekend_avg = weekend_data.mean()
                        weekday_avg = weekday_data.mean()
                        profile['weekend_boost'] = float(weekend_avg / (weekday_avg + 1e-8)) if weekday_avg > 0 else 1.0
                    else:
                        profile['weekend_boost'] = 1.0
                        
                except Exception as e:
                    logger.warning(f"요일별 패턴 계산 실패 ({venue}): {e}")
                    profile['weekend_boost'] = 1.0
                    profile['dow_pattern'] = {}
                
                self.venue_profiles[venue] = profile
                
        except Exception as e:
            logger.error(f"프로파일 계산 중 오류: {e}")
            self.venue_profiles = {}
    
    # ==========================================
    # 2. 안전한 아웃라이어 처리
    # ==========================================
    
    def safe_outlier_treatment(self, df):
        """안전한 아웃라이어 처리"""
        
        logger.info("아웃라이어 처리 중...")
        
        df = df.copy()
        df['sales_processed'] = df['매출수량'].copy()
        
        try:
            for venue_menu, group in df.groupby('영업장명_메뉴명'):
                if len(group) < 5:  # 데이터가 너무 적으면 스킵
                    continue
                
                venue = str(venue_menu).split('_')[0] if '_' in str(venue_menu) else str(venue_menu)
                sales = group['매출수량']
                
                # 안전한 분위수 계산
                try:
                    q1 = sales.quantile(0.25)
                    q3 = sales.quantile(0.75)
                    iqr = q3 - q1
                    
                    if venue in ['담하', '미라시아']:
                        multiplier = 1.2  # 보수적
                        scaling = 0.95
                    else:
                        multiplier = 1.5  # 표준
                        scaling = 1.0
                    
                    upper_bound = q3 + multiplier * iqr
                    lower_bound = max(0, q1 - multiplier * iqr)
                    
                    # 클리핑
                    mask = df['영업장명_메뉴명'] == venue_menu
                    clipped_values = np.clip(df.loc[mask, '매출수량'], lower_bound, upper_bound)
                    df.loc[mask, 'sales_processed'] = clipped_values * scaling
                    
                except Exception as e:
                    logger.warning(f"아웃라이어 처리 실패 ({venue_menu}): {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"전체 아웃라이어 처리 실패: {e}")
            df['sales_processed'] = df['매출수량']
        
        return df
    
    # ==========================================
    # 3. 간소화된 모델 아키텍처
    # ==========================================
    
    class SimpleLSTM(nn.Module):
        """간소화된 LSTM 모델"""
        def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=7, dropout=0.3):
            super().__init__()
            
            self.lstm = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0
            )
            
            self.predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            output = self.predictor(last_hidden)
            return output
    
    class SimpleTransformer(nn.Module):
        """간소화된 Transformer"""
        def __init__(self, input_dim=20, d_model=64, nhead=4, num_layers=2, 
                     seq_len=28, pred_len=7, dropout=0.3):
            super().__init__()
            
            self.input_projection = nn.Linear(input_dim, d_model)
            self.positional_encoding = nn.Parameter(torch.randn(seq_len, d_model))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dropout=dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            
            self.predictor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, pred_len)
            )
            
        def forward(self, x):
            x = self.input_projection(x)
            x = x + self.positional_encoding.unsqueeze(0)
            x = self.transformer(x)
            x = x[:, -1, :]
            output = self.predictor(x)
            return output
    
    # ==========================================
    # 4. 안전한 시퀀스 생성
    # ==========================================
    
    def create_safe_sequences(self, df, mode='train'):
        """수정된 시퀀스 생성 - 특성 컬럼 문제 해결"""
        
        logger.info(f"시퀀스 생성 중 (mode: {mode})...")
        
        sequences = []
        targets = []
        metadata = []
        
        try:
            for venue_menu, group in df.groupby('영업장명_메뉴명'):
                group = group.sort_values('date').reset_index(drop=True)
                
                min_length = self.sequence_length + (self.prediction_length if mode == 'train' else 0)
                if len(group) < min_length:
                    print(f"⚠️ {venue_menu}: 데이터 부족 ({len(group)} < {min_length})")
                    continue
                
                # 🔥 핵심 수정: 특성 컬럼 선택 로직 개선
                exclude_cols = ['date', 'venue', 'menu', '영업일자', '영업장명_메뉴명', '매출수량', 'sales_processed']
                
                # 수치형 컬럼만 선택하되, 더 관대한 조건 적용
                all_numeric_cols = []
                for col in group.columns:
                    if col not in exclude_cols:
                        if pd.api.types.is_numeric_dtype(group[col]):
                            # NaN이 전부가 아니고, 분산이 0이 아닌 컬럼만
                            if not group[col].isna().all() and group[col].var() != 0:
                                all_numeric_cols.append(col)
                
                print(f"📊 {venue_menu}: 사용 가능한 특성 {len(all_numeric_cols)}개")
                print(f"    특성 목록: {all_numeric_cols[:10]}...")  # 처음 10개만 출력
                
                if len(all_numeric_cols) == 0:
                    print(f"❌ {venue_menu}: 사용 가능한 특성이 없음")
                    continue
                
                # feature_columns 설정 (전역적으로)
                if not self.feature_columns:
                    self.feature_columns = all_numeric_cols
                    print(f"✅ 전역 특성 컬럼 설정: {len(self.feature_columns)}개")
                
                # 현재 그룹에서 사용할 특성 (전역 특성과 교집합)
                available_features = [col for col in self.feature_columns if col in group.columns]
                
                if len(available_features) == 0:
                    print(f"❌ {venue_menu}: 공통 특성이 없음")
                    continue
                
                print(f"✅ {venue_menu}: {len(available_features)}개 특성 사용")
                
                # 타겟 컬럼
                target_col = 'sales_processed' if 'sales_processed' in group.columns else '매출수량'
                
                # venue/menu 안전 분리
                try:
                    venue = str(venue_menu).split('_')[0] if '_' in str(venue_menu) else str(venue_menu)
                    menu = '_'.join(str(venue_menu).split('_')[1:]) if '_' in str(venue_menu) else ''
                except:
                    venue = str(venue_menu)
                    menu = ''
                
                if mode == 'predict':
                    # 예측 모드
                    seq_data = group[available_features + [target_col]].tail(self.sequence_length)
                    if len(seq_data) == self.sequence_length:
                        seq_array = seq_data[available_features].values.astype(np.float32)
                        # NaN 처리
                        seq_array = np.nan_to_num(seq_array, nan=0.0, posinf=0.0, neginf=0.0)
                        sequences.append(seq_array)
                        metadata.append({
                            'venue_menu': venue_menu,
                            'venue': venue,
                            'menu': menu,
                            'last_date': group['date'].iloc[-1]
                        })
                else:
                    # 훈련 모드 - 연속적인 시퀀스만 생성 (step_size 제거)
                    seq_generated = 0
                    max_sequences_per_venue = 5  # 메모리 절약을 위해 제한
                    
                    for i in range(0, len(group) - min_length + 1):
                        if seq_generated >= max_sequences_per_venue:
                            break
                            
                        seq_data = group[available_features + [target_col]].iloc[i:i+self.sequence_length]
                        target_data = group[target_col].iloc[i+self.sequence_length:i+self.sequence_length+self.prediction_length]
                        
                        if len(seq_data) == self.sequence_length and len(target_data) == self.prediction_length:
                            seq_array = seq_data[available_features].values.astype(np.float32)
                            target_array = target_data.values.astype(np.float32)
                            
                            # NaN 처리
                            seq_array = np.nan_to_num(seq_array, nan=0.0, posinf=0.0, neginf=0.0)
                            target_array = np.nan_to_num(target_array, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            sequences.append(seq_array)
                            targets.append(target_array)
                            metadata.append({
                                'venue_menu': venue_menu,
                                'venue': venue,
                                'menu': menu
                            })
                            seq_generated += 1
        
        except Exception as e:
            logger.error(f"시퀀스 생성 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), np.array([]), []
        
        X = np.array(sequences) if sequences else np.empty((0, self.sequence_length, len(self.feature_columns) if self.feature_columns else 1))
        y = np.array(targets) if targets else np.empty((0, self.prediction_length))
        
        print(f"🎯 최종 시퀀스 생성 결과:")
        print(f"   X: {X.shape} (시퀀스수, 시간단계, 특성수)")
        print(f"   y: {y.shape} (시퀀스수, 예측일수)")
        print(f"   메타데이터: {len(metadata)}개")
        
        return X, y, metadata
    
    # ==========================================
    # 5. 안전한 모델 학습
    # ==========================================
    
    def train_safe_models(self, train_df):
        """안전한 모델 학습"""
        
        logger.info("안전한 모델 학습 시작...")
        
        try:
            # 특성 생성
            train_df = self.create_safe_features(train_df, is_training=True)
            train_df = self.safe_outlier_treatment(train_df)
            
            # 시퀀스 생성
            X, y, metadata = self.create_safe_sequences(train_df, mode='train')
            
            if len(X) == 0:
                logger.error("학습할 시퀀스 데이터가 없습니다!")
                return
            
            logger.info(f"학습 데이터: X {X.shape}, y {y.shape}")
            
            # 안전한 스케일링
            try:
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
                self.scalers['feature_scaler'] = scaler
                
                # 타겟 스케일링
                y_flat = y.reshape(-1, 1)
                target_scaler = RobustScaler()
                y_scaled = target_scaler.fit_transform(y_flat).reshape(y.shape)
                self.scalers['target_scaler'] = target_scaler
                
            except Exception as e:
                logger.error(f"스케일링 실패: {e}")
                X_scaled = X
                y_scaled = y
            
            # PyTorch 데이터 준비
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            y_tensor = torch.FloatTensor(y_scaled).to(self.device)
            
            # 데이터 분할
            split_idx = max(1, int(len(X_tensor) * 0.8))
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]
            
            logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
            
            # 1. Simple LSTM 학습
            logger.info("Simple LSTM 학습 중...")
            try:
                lstm_model = self._train_safe_pytorch_model(
                    self.SimpleLSTM(
                        input_dim=X_scaled.shape[-1],
                        hidden_dim=self.hyperparams['lstm_hidden_dim'],
                        num_layers=self.hyperparams['lstm_num_layers'],
                        output_dim=self.prediction_length,
                        dropout=self.hyperparams['dropout']
                    ),
                    X_train, y_train, X_val, y_val, 'Simple LSTM'
                )
                self.models['simple_lstm'] = lstm_model
                logger.info("✅ Simple LSTM 학습 완료")
            except Exception as e:
                logger.error(f"LSTM 학습 실패: {e}")
            
            # 2. Simple Transformer 학습
            logger.info("Simple Transformer 학습 중...")
            try:
                transformer_model = self._train_safe_pytorch_model(
                    self.SimpleTransformer(
                        input_dim=X_scaled.shape[-1],
                        d_model=self.hyperparams['transformer_d_model'],
                        nhead=self.hyperparams['transformer_nhead'],
                        num_layers=self.hyperparams['transformer_num_layers'],
                        seq_len=self.sequence_length,
                        pred_len=self.prediction_length,
                        dropout=self.hyperparams['dropout']
                    ),
                    X_train, y_train, X_val, y_val, 'Simple Transformer'
                )
                self.models['simple_transformer'] = transformer_model
                logger.info("✅ Simple Transformer 학습 완료")
            except Exception as e:
                logger.error(f"Transformer 학습 실패: {e}")
            
            # 3. LightGBM 학습 (간소화)
            logger.info("LightGBM 학습 중...")
            try:
                lgb_model = self._train_safe_lightgbm(X_scaled, y, metadata)
                self.models['lightgbm'] = lgb_model
                logger.info("✅ LightGBM 학습 완료")
            except Exception as e:
                logger.error(f"LightGBM 학습 실패: {e}")
            
            logger.info(f"모델 학습 완료! 총 {len(self.models)}개 모델")
            
        except Exception as e:
            logger.error(f"전체 모델 학습 실패: {e}")
            raise
    
    def _train_safe_pytorch_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """안전한 PyTorch 모델 학습"""
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.hyperparams['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
        criterion = nn.MSELoss()  # 간단한 MSE 사용
        
        # 작은 배치 사이즈로 데이터 로더
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparams['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.hyperparams['epochs']):
            try:
                # 훈련
                model.train()
                train_loss = 0
                batch_count = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()
                    train_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    train_loss /= batch_count
                
                # 검증
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val).item()
                
                scheduler.step(val_loss)
                
                # 조기 종료
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"{model_name} Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
                
                if patience_counter >= self.hyperparams['patience']:
                    logger.info(f"{model_name} 조기 종료 at epoch {epoch}")
                    break
                    
            except Exception as e:
                logger.warning(f"Epoch {epoch} 오류: {e}")
                continue
        
        # 최고 성능 모델 로드
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)
        
        return model
    
    def _train_safe_lightgbm(self, X, y, metadata):
        """안전한 LightGBM 학습"""
        
        try:
            # 간단한 특성 생성
            X_simple = []
            y_flat = []
            
            for i, (seq, target) in enumerate(zip(X, y)):
                if i >= len(metadata):
                    break
                    
                meta = metadata[i]
                venue = meta.get('venue', '')
                
                # 간단한 통계 특성만
                seq_mean = np.mean(seq, axis=0)
                seq_last = seq[-1] if len(seq) > 0 else np.zeros(seq.shape[-1])
                
                venue_features = [
                    int(venue == '담하'),
                    int(venue == '미라시아')
                ]
                
                # 각 예측일별로
                for day_idx in range(self.prediction_length):
                    if day_idx < len(target):
                        features = np.concatenate([seq_mean, seq_last, venue_features, [day_idx]])
                        X_simple.append(features)
                        y_flat.append(target[day_idx])
            
            if len(X_simple) == 0:
                logger.warning("LightGBM용 데이터가 없습니다")
                return None
            
            X_lgb = np.array(X_simple)
            y_lgb = np.array(y_flat)
            
            # 간단한 LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100,  # 적은 트리 수
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=-1
            )
            
            lgb_model.fit(X_lgb, y_lgb)
            return lgb_model
            
        except Exception as e:
            logger.error(f"LightGBM 학습 실패: {e}")
            return None
    
    # ==========================================
    # 6. 안전한 예측
    # ==========================================
    
    def predict_safe_ensemble(self, test_df):
        """안전한 앙상블 예측"""
        
        logger.info("안전한 앙상블 예측 시작...")
        
        try:
            # 특성 생성
            test_df = self.create_safe_features(test_df, is_training=False)
            
            # 시퀀스 생성
            X, _, metadata = self.create_safe_sequences(test_df, mode='predict')
            
            if len(X) == 0:
                logger.warning("예측할 데이터가 없습니다!")
                return {}
            
            # 스케일링
            if 'feature_scaler' in self.scalers:
                X_scaled = self.scalers['feature_scaler'].transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            else:
                X_scaled = X
            
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            predictions = {}
            
            # 각 모델별 예측
            for model_name, model in self.models.items():
                logger.info(f"{model_name} 예측 중...")
                
                try:
                    if model_name in ['simple_lstm', 'simple_transformer']:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_tensor)
                            
                            # 역변환
                            if 'target_scaler' in self.scalers:
                                outputs_np = outputs.cpu().numpy()
                                outputs_reshaped = outputs_np.reshape(-1, 1)
                                outputs_inv = self.scalers['target_scaler'].inverse_transform(outputs_reshaped)
                                model_preds = outputs_inv.reshape(outputs_np.shape)
                            else:
                                model_preds = outputs.cpu().numpy()
                                
                    elif model_name == 'lightgbm' and model is not None:
                        model_preds = []
                        for i, seq in enumerate(X_scaled):
                            if i >= len(metadata):
                                break
                                
                            meta = metadata[i]
                            venue = meta.get('venue', '')
                            
                            seq_mean = np.mean(seq, axis=0)
                            seq_last = seq[-1] if len(seq) > 0 else np.zeros(seq.shape[-1])
                            venue_features = [int(venue == '담하'), int(venue == '미라시아')]
                            
                            day_preds = []
                            for day_idx in range(self.prediction_length):
                                features = np.concatenate([seq_mean, seq_last, venue_features, [day_idx]])
                                pred = model.predict([features])[0]
                                day_preds.append(pred)
                            
                            model_preds.append(day_preds)
                        
                        model_preds = np.array(model_preds)
                    else:
                        continue
                    
                    predictions[model_name] = model_preds
                    
                except Exception as e:
                    logger.warning(f"{model_name} 예측 실패: {e}")
                    continue
            
            # 앙상블 (안전한 가중치)
            ensemble_predictions = {}
            
            for i, meta in enumerate(metadata):
                venue_menu = meta['venue_menu']
                venue = meta.get('venue', '')
                
                # 간단한 가중치
                if venue in ['담하', '미라시아']:
                    weights = {'simple_lstm': 0.6, 'simple_transformer': 0.3, 'lightgbm': 0.1}
                else:
                    weights = {'simple_lstm': 0.4, 'simple_transformer': 0.4, 'lightgbm': 0.2}
                
                # 앙상블
                ensemble_pred = np.zeros(self.prediction_length)
                total_weight = 0
                
                for model_name, weight in weights.items():
                    if model_name in predictions and i < len(predictions[model_name]):
                        ensemble_pred += weight * predictions[model_name][i]
                        total_weight += weight
                
                if total_weight > 0:
                    ensemble_pred = ensemble_pred / total_weight
                
                # 후처리
                ensemble_pred = np.maximum(ensemble_pred, 0)  # 음수 제거
                
                # 업장별 상한선
                if venue in self.venue_profiles:
                    max_allowed = self.venue_profiles[venue].get('q95', 1000)
                    if venue in ['담하', '미라시아']:
                        max_allowed *= 1.3
                    else:
                        max_allowed *= 1.8
                    ensemble_pred = np.minimum(ensemble_pred, max_allowed)
                
                ensemble_predictions[venue_menu] = ensemble_pred
            
            logger.info(f"앙상블 예측 완료. {len(ensemble_predictions)}개 업장메뉴")
            return ensemble_predictions
            
        except Exception as e:
            logger.error(f"앙상블 예측 실패: {e}")
            return {}
    
    # ==========================================
    # 7. 제출 파일 생성
    # ==========================================
    
    
    def create_safe_submission(self, predictions_dict, sample_submission_path='sample_submission.csv'):
        """올바른 제출 파일 생성 - 형식 수정"""
        
        logger.info("올바른 제출 파일 생성 중...")
        
        try:
            # 샘플 제출 파일 로드
            sample_submission = pd.read_csv(sample_submission_path)
            submission = sample_submission.copy()
            
            print(f"📋 원본 제출 파일 형태: {submission.shape}")
            print(f"📋 첫 5개 행의 영업일자: {submission['영업일자'].head().tolist()}")
            
            # 모든 수치 값을 0으로 초기화
            numeric_cols = submission.select_dtypes(include=[np.number]).columns
            submission[numeric_cols] = 0.0
            
            print(f"📊 수치 컬럼 수: {len(numeric_cols)}")
            print(f"📊 예측 데이터 수: {len(predictions_dict)}")
            
            # 🔥 핵심 수정: 올바른 날짜별 매핑
            for test_idx in range(10):  # TEST_00 ~ TEST_09
                test_prefix = f"TEST_{test_idx:02d}"
                
                # 해당 테스트의 7개 행 찾기
                test_mask = submission['영업일자'].str.startswith(test_prefix)
                test_rows = submission[test_mask].index.tolist()
                
                if len(test_rows) != 7:
                    print(f"⚠️ {test_prefix}: 행 수 이상 ({len(test_rows)}개)")
                    continue
                
                print(f"✅ {test_prefix}: {len(test_rows)}개 행 처리")
                
                # 각 업장메뉴별 예측값 매핑
                mapped_count = 0
                for venue_menu, preds in predictions_dict.items():
                    if venue_menu in submission.columns:
                        # 7일간 예측값을 해당하는 7개 행에 배치
                        for day_idx, row_idx in enumerate(test_rows):
                            if day_idx < len(preds):
                                pred_value = max(0.0, float(preds[day_idx]))
                                submission.loc[row_idx, venue_menu] = pred_value
                        mapped_count += 1
                
                print(f"   → {mapped_count}개 업장메뉴 매핑 완료")
            
            # 결과 검증
            print(f"\n🔍 결과 검증:")
            total_predictions = submission[numeric_cols].sum().sum()
            zero_ratio = (submission[numeric_cols] == 0).sum().sum() / (len(submission) * len(numeric_cols))
            
            print(f"   총 예측값: {total_predictions:,.1f}")
            print(f"   0값 비율: {zero_ratio:.1%}")
            
            # 각 테스트별 통계 확인
            for test_idx in range(3):  # 처음 3개만 확인
                test_prefix = f"TEST_{test_idx:02d}"
                test_mask = submission['영업일자'].str.startswith(test_prefix)
                test_data = submission[test_mask]
                if len(test_data) > 0:
                    daily_sum = test_data[numeric_cols].sum(axis=1)
                    print(f"   {test_prefix} 일별 합계: {daily_sum.tolist()}")
            
            logger.info("제출 파일 생성 완료!")
            return submission
            
        except Exception as e:
            logger.error(f"제출 파일 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 실패시 기본 제출 파일 반환
            sample_submission = pd.read_csv(sample_submission_path)
            numeric_cols = sample_submission.select_dtypes(include=[np.number]).columns
            sample_submission[numeric_cols] = 1.0  # 기본값 1로 설정
            return sample_submission
    
    # ==========================================
    # 8. 메인 파이프라인
    # ==========================================
    
    def fit(self, train_df_path='train.csv'):
        """안전한 학습 파이프라인"""
        
        logger.info("=== 안전한 고급 모델 학습 시작 ===")
        
        try:
            # 데이터 로드
            train_df = pd.read_csv(train_df_path)
            logger.info(f"훈련 데이터: {train_df.shape}")
            
            # 모델 학습
            self.train_safe_models(train_df)
            
            logger.info("=== 모델 학습 완료 ===")
            
        except Exception as e:
            logger.error(f"학습 파이프라인 실패: {e}")
            raise
    
    def predict_all_tests_safe(self, test_files_pattern='TEST_*.csv'):
        """모든 테스트 파일 안전 예측"""
        
        logger.info("=== 모든 테스트 파일 안전 예측 시작 ===")
        
        test_files = sorted(glob.glob(test_files_pattern))
        all_predictions = {}
        
        for test_file in tqdm(test_files, desc="테스트 파일 처리"):
            try:
                logger.info(f"처리 중: {test_file}")
                
                test_df = pd.read_csv(test_file)
                test_predictions = self.predict_safe_ensemble(test_df)
                
                # 결과 저장
                for venue_menu, preds in test_predictions.items():
                    all_predictions[venue_menu] = preds
                    
            except Exception as e:
                logger.warning(f"{test_file} 처리 실패: {e}")
                continue
        
        logger.info("=== 모든 테스트 예측 완료 ===")
        return all_predictions


# ==========================================
# 수정된 실행 함수들
# ==========================================

def run_fixed_advanced_pipeline():
    """수정된 고급 파이프라인 실행"""
    
    print("🔧 수정된 고급 시계열 예측 모델 실행!")
    print("=" * 60)
    
    try:
        # 1. 모델 초기화
        model = FixedAdvancedForecastingModel(
            sequence_length=28,
            prediction_length=7,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        print(f"디바이스: {model.device}")
        
        # 2. 모델 학습
        print("\n📚 모델 학습 단계")
        print("-" * 30)
        model.fit('train.csv')
        
        # 3. 예측 수행
        print("\n🔮 예측 수행 단계")
        print("-" * 30)
        all_predictions = model.predict_all_tests_safe('TEST_*.csv')
        
        # 4. 제출 파일 생성
        print("\n📝 제출 파일 생성")
        print("-" * 30)
        submission = model.create_safe_submission(all_predictions)
        
        # 5. 결과 저장
        output_path = 'fixed_advanced_submission.csv'
        submission.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 6. 결과 분석
        print("\n📊 결과 분석")
        print("-" * 30)
        
        numeric_cols = submission.select_dtypes(include=[np.number]).columns
        total_predictions = submission[numeric_cols].sum().sum()
        mean_prediction = submission[numeric_cols].mean().mean()
        zero_ratio = (submission[numeric_cols] == 0).sum().sum() / submission[numeric_cols].size
        
        print(f"총 예측값 합계: {total_predictions:,.1f}")
        print(f"평균 예측값: {mean_prediction:.2f}")
        print(f"0 예측 비율: {zero_ratio:.2%}")
        print(f"예측 모델 수: {len(model.models)}")
        
        print(f"\n✅ 완료! 제출 파일: {output_path}")
        print("🎯 오류가 수정된 안정적인 버전입니다!")
        
        return submission, all_predictions, model
        
    except Exception as e:
        print(f"❌ 파이프라인 실행 실패: {e}")
        print("\n🔍 문제 해결 방법:")
        print("1. 데이터 파일들이 모두 있는지 확인")
        print("2. 메모리 부족시 batch_size를 더 줄여보세요")
        print("3. 빠른 테스트로 먼저 확인해보세요")
        raise

# 🔥 추가 수정: 빠른 테스트를 위한 더 큰 샘플 사용
def quick_test_fixed_model(sample_size=2000):  # 200 → 2000으로 증가
    """더 큰 샘플로 테스트"""
    
    print("⚡ 수정된 모델 빠른 테스트 v2")
    print("=" * 40)
    
    try:
        train_df = pd.read_csv('train.csv')
        print(f"전체 데이터: {train_df.shape}")
        
        # 더 큰 샘플 + 각 업장메뉴별로 균등 샘플링
        venue_menu_groups = train_df.groupby('영업장명_메뉴명')
        sampled_groups = []
        
        for venue_menu, group in venue_menu_groups:
            # 각 그룹에서 최대 50개씩 샘플링 (연속성 고려)
            if len(group) >= 35:  # 최소 길이 체크
                group_sorted = group.sort_values('영업일자')
                sample_start = max(0, len(group_sorted) - 50)  # 마지막 50개
                sampled_groups.append(group_sorted.iloc[sample_start:])
        
        if sampled_groups:
            train_sample = pd.concat(sampled_groups, ignore_index=True)
        else:
            train_sample = train_df.sample(n=min(sample_size, len(train_df))).copy()
        
        print(f"샘플 데이터: {train_sample.shape}")
        print(f"업장메뉴 수: {train_sample['영업장명_메뉴명'].nunique()}")
        
        # 각 업장메뉴별 데이터 길이 확인
        venue_lengths = train_sample.groupby('영업장명_메뉴명').size()
        sufficient_data = (venue_lengths >= 35).sum()
        print(f"충분한 데이터 (≥35일)를 가진 업장메뉴: {sufficient_data}개")
        
        if sufficient_data == 0:
            print("❌ 시퀀스 생성을 위한 충분한 데이터가 없습니다!")
            print("   → 전체 데이터를 사용하거나 sequence_length를 줄여보세요")
            return False
        
        # 모델 초기화 (더 작은 설정)
        model = FixedAdvancedForecastingModel(sequence_length=14, prediction_length=7)  # 28→14로 축소
        model.hyperparams.update({
            'epochs': 5,
            'batch_size': 8,
            'patience': 2
        })
        
        print("✅ 특성 생성 테스트...")
        enhanced_df = model.create_safe_features(train_sample)
        numeric_features = len([col for col in enhanced_df.columns if pd.api.types.is_numeric_dtype(enhanced_df[col])])
        print(f"특성 수: {numeric_features}")
        
        print("✅ 시퀀스 생성 테스트...")
        X, y, metadata = model.create_safe_sequences(enhanced_df, mode='train')
        print(f"시퀀스: X {X.shape}, y {y.shape}")
        
        if len(X) > 0 and X.shape[-1] > 0:
            print("✅ 수정된 모델이 정상 작동합니다!")
            print("🚀 전체 파이프라인 실행 가능!")
            return True
        else:
            print("⚠️ 여전히 시퀀스 생성 문제 있음")
            print(f"   X shape: {X.shape}")
            print("   → sequence_length를 더 줄이거나 전체 데이터를 사용해보세요")
            return False
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

print("🔧 수정된 함수들:")
print("  • create_safe_sequences() - 특성 컬럼 선택 로직 개선") 
print("  • quick_test_fixed_model_v2() - 더 큰 샘플로 테스트")
print("\n실행: quick_test_fixed_model_v2()")


# 실행 예시
if __name__ == "__main__":
    # 빠른 테스트 먼저
    print("🧪 빠른 테스트부터 시작...")
    if quick_test_fixed_model():
        print("\n🚀 전체 파이프라인 실행...")
        submission, predictions, model = run_fixed_advanced_pipeline()
    else:
        print("\n❌ 테스트 실패. 문제를 확인해주세요.")