import pandas as pd
import numpy as np
import torch
import glob
import os
import warnings
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
warnings.filterwarnings('ignore')


# 루트 로거 재설정: 내 코드만 INFO, 외부 라이브러리는 WARNING 이상만
logging.basicConfig(level=logging.INFO, force=True)

# Matplotlib 쪽 디버그 차단
plt.set_loglevel("warning")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

# ensemble_model.py 파일 상단에 추가
import sys
sys.path.append('./')

# 기존 LSTM 모델 함수들 import
from baseline_lstm import (
    train_lstm, predict_lstm, generate_combined_holiday_list, 
    solar_md_holidays, lunar_solar_dates
)
# mene clipping import

# 클러스터링 모델 import  
from cluster_based_model import ClusterBasedForecastingModel

class EnsemblePredictor:
    """LSTM + 클러스터링 모델 앙상블"""
    from cluster_based_model import ClusterBasedForecastingModel
    def __init__(self, high_weight_venues=['담하', '미라시아']):
        self.high_weight_venues = high_weight_venues
        self.lstm_models = None
        self.cluster_model = None
        self.ensemble_weights = {}
        self.venue_performance = {}
        
    def calculate_smape(self, y_true, y_pred):
        """SMAPE 계산"""
        mask = y_true != 0
        if mask.sum() == 0:
            return 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        return 100 * np.mean(2 * np.abs(y_pred_filtered - y_true_filtered) / 
                            (np.abs(y_true_filtered) + np.abs(y_pred_filtered)))
    
    def load_and_train_models(self, train_df_path='./train/train.csv'):
        """두 모델을 훈련합니다"""
        
        print("데이터 로딩 및 전처리...")
        train_df = pd.read_csv(train_df_path)
        
        # 매출 데이터 음수 제거
        train_df['매출수량'] = train_df['매출수량'].clip(lower=0)
        
        # 1. LSTM 모델 훈련
        print("LSTM 모델 훈련 중...")
        from korean_lunar_calendar import KoreanLunarCalendar
        
        # LSTM 전처리 (기존 코드와 동일)
        train_df = generate_combined_holiday_list(train_df, solar_md_holidays, lunar_solar_dates)
        self.lstm_models = train_lstm(train_df, use_validation=False, dropout=0.1)
        print(f"LSTM 모델 훈련 완료: {len(self.lstm_models)}개 모델")
        
        # 2. 클러스터링 모델 훈련
        print("클러스터링 모델 훈련 중...")
        cluster_train_df = train_df.copy()
        cluster_train_df['date'] = pd.to_datetime(cluster_train_df['영업일자'])
        cluster_train_df[['store', 'menu']] = cluster_train_df['영업장명_메뉴명'].str.split('_', expand=True, n=1)
        cluster_train_df['sales'] = cluster_train_df['매출수량']
        cluster_train_df['month'] = cluster_train_df['date'].dt.month
        cluster_train_df['day_of_week'] = cluster_train_df['date'].dt.dayofweek
        cluster_train_df['is_weekend'] = cluster_train_df['day_of_week'].isin([5, 6])
        
        self.cluster_model = ClusterBasedForecastingModel()
        self.cluster_model.fit(cluster_train_df)
        print("클러스터링 모델 훈련 완료")
        
    def find_optimal_weights_on_validation(self, validation_files=None):
        """검증 데이터로 최적 가중치 찾기"""
        
        if validation_files is None:
            # 마지막 몇 개 TEST 파일을 검증용으로 사용
            test_files = sorted(glob.glob('./TEST_*.csv'))
            validation_files = test_files[-2:]  # 마지막 2개 파일
        
        print("🔍 최적 가중치 탐색 중...")
        
        total_lstm_preds = []
        total_cluster_preds = []
        total_targets = []
        total_venues = []
        
        for val_file in validation_files:
            print(f"   처리 중: {val_file}")
            
            # LSTM 예측
            lstm_results = self._get_lstm_predictions(val_file)
            
            # 클러스터링 예측  
            cluster_results = self._get_cluster_predictions(val_file)
            
            # 결과 정렬 및 매칭
            matched_data = self._match_predictions(lstm_results, cluster_results)
            
            for item in matched_data:
                total_lstm_preds.extend(item['lstm_pred'])
                total_cluster_preds.extend(item['cluster_pred'])
                total_targets.extend(item['target'])
                total_venues.extend([item['venue']] * 7)  # 7일치
        
        if len(total_lstm_preds) == 0:
            print("⚠️ 검증 데이터 부족, 기본 가중치 사용")
            return {'global': 0.3}  # LSTM 30%, 클러스터링 70%
        
        # 전역 최적 가중치 찾기
        best_weight = 0.3
        best_score = float('inf')
        
        print("   가중치 최적화...")
        for w in np.arange(0.0, 1.1, 0.1):
            ensemble_pred = np.array(total_lstm_preds) * w + np.array(total_cluster_preds) * (1-w)
            score = self.calculate_smape(np.array(total_targets), ensemble_pred)
            
            if score < best_score:
                best_score = score
                best_weight = w
        
        # 업장별 최적 가중치 (고가중치 업장)
        venue_weights = {}
        for venue in self.high_weight_venues:
            venue_mask = np.array([v.startswith(venue) for v in total_venues])
            if venue_mask.sum() > 20:  # 충분한 데이터가 있을 때만
                venue_lstm = np.array(total_lstm_preds)[venue_mask]
                venue_cluster = np.array(total_cluster_preds)[venue_mask] 
                venue_target = np.array(total_targets)[venue_mask]
                
                best_venue_weight = best_weight
                best_venue_score = float('inf')
                
                for w in np.arange(0.0, 1.1, 0.05):  # 더 세밀하게
                    venue_ensemble = venue_lstm * w + venue_cluster * (1-w)
                    venue_score = self.calculate_smape(venue_target, venue_ensemble)
                    
                    if venue_score < best_venue_score:
                        best_venue_score = venue_score
                        best_venue_weight = w
                
                venue_weights[venue] = best_venue_weight
        
        self.ensemble_weights = {
            'global': best_weight,
            'venues': venue_weights
        }
        
        print(f"✅ 최적 가중치 - 전역: {best_weight:.2f}, 특별업장: {venue_weights}")
        return self.ensemble_weights
    
    def _get_lstm_predictions(self, test_file):
        """LSTM 모델로 예측"""
        test_df = pd.read_csv(test_file)
        filename = os.path.basename(test_file)
        test_prefix = filename.replace('.csv', '')
        
        # LSTM 예측 수행 (기존 함수 활용)
        pred_df = predict_lstm(test_df, self.lstm_models, test_prefix)
        
        results = []
        for _, row in pred_df.iterrows():
            venue = row['영업장명_메뉴명']
            day = row['영업일자']
            value = row['매출수량']
            results.append({
                'venue': venue,
                'day': day, 
                'prediction': value
            })
        
        return results
    
    def _get_cluster_predictions(self, test_file):
        """클러스터링 모델로 예측"""
        test_df = pd.read_csv(test_file)
        
        # 클러스터링 모델 전처리
        cluster_test_df = test_df.copy()
        cluster_test_df['date'] = pd.to_datetime(cluster_test_df['영업일자'])
        cluster_test_df[['store', 'menu']] = cluster_test_df['영업장명_메뉴명'].str.split('_', expand=True, n=1)
        cluster_test_df['sales'] = cluster_test_df['매출수량']
        cluster_test_df['month'] = cluster_test_df['date'].dt.month
        cluster_test_df['day_of_week'] = cluster_test_df['date'].dt.dayofweek
        cluster_test_df['is_weekend'] = cluster_test_df['day_of_week'].isin([5, 6])
        
        # 클러스터링 예측
        predictions, metadata = self.cluster_model.predict(cluster_test_df)
        
        results = []
        filename = os.path.basename(test_file)
        test_prefix = filename.replace('.csv', '')
        
        for i, meta in enumerate(metadata):
            venue = f"{meta['store']}_{meta['menu']}"
            for day_idx in range(7):
                day = f"{test_prefix}+{day_idx+1}일"
                value = predictions[i][day_idx] if i < len(predictions) else 0
                results.append({
                    'venue': venue,
                    'day': day,
                    'prediction': value
                })
        
        return results
    
    def _match_predictions(self, lstm_results, cluster_results):
        """두 모델의 예측 결과 매칭"""
        
        # 딕셔너리로 변환하여 빠른 검색
        lstm_dict = {}
        for item in lstm_results:
            key = (item['venue'], item['day'])
            lstm_dict[key] = item['prediction']
        
        cluster_dict = {}
        for item in cluster_results:
            key = (item['venue'], item['day'])
            cluster_dict[key] = item['prediction']
        
        # 공통 키 찾기
        common_keys = set(lstm_dict.keys()) & set(cluster_dict.keys())
        
        # 업장별로 그룹화
        venue_groups = {}
        for venue, day in common_keys:
            if venue not in venue_groups:
                venue_groups[venue] = []
            venue_groups[venue].append((venue, day))
        
        matched_data = []
        for venue, keys in venue_groups.items():
            if len(keys) == 7:  # 7일치 모두 있을 때만
                keys_sorted = sorted(keys, key=lambda x: x[1])  # 날짜순 정렬
                
                lstm_pred = [lstm_dict[key] for key in keys_sorted]
                cluster_pred = [cluster_dict[key] for key in keys_sorted]
                target = [0] * 7  # 실제 타겟은 없으므로 0으로 설정 (검증용이 아니라면)
                
                matched_data.append({
                    'venue': venue,
                    'lstm_pred': lstm_pred,
                    'cluster_pred': cluster_pred,
                    'target': target
                })
        
        return matched_data
    
    def predict_ensemble(self, test_files):
        """앙상블 예측 수행"""
        
        if self.lstm_models is None or self.cluster_model is None:
            raise ValueError("모델이 훈련되지 않았습니다. load_and_train_models()를 먼저 실행하세요.")
        
        print("🎭 앙상블 예측 시작...")
        
        # 가중치가 설정되지 않았으면 기본값 사용
        if not hasattr(self, 'ensemble_weights') or not self.ensemble_weights:
            print("기본 가중치 사용 (LSTM 30%, 클러스터링 70%)")
            self.ensemble_weights = {'global': 0.3, 'venues': {}}
        
        all_predictions = []
        
        for test_file in tqdm(test_files, desc="테스트 파일 처리"):
            # 각 모델 예측
            lstm_results = self._get_lstm_predictions(test_file)
            cluster_results = self._get_cluster_predictions(test_file)
            
            # 예측 결과 매칭
            matched_data = self._match_predictions(lstm_results, cluster_results)
            
            # 앙상블 수행
            for item in matched_data:
                venue = item['venue']
                lstm_pred = np.array(item['lstm_pred'])
                cluster_pred = np.array(item['cluster_pred'])
                
                # 업장별 가중치 결정
                venue_name = venue.split('_')[0]
                if venue_name in self.ensemble_weights.get('venues', {}):
                    weight = self.ensemble_weights['venues'][venue_name]
                else:
                    weight = self.ensemble_weights['global']
                
                # 앙상블 예측
                ensemble_pred = weight * lstm_pred + (1 - weight) * cluster_pred
                
                # 후처리
                ensemble_pred = np.maximum(ensemble_pred, 1)  # 최소값 1
                
                # 결과 저장
                filename = os.path.basename(test_file)
                test_prefix = filename.replace('.csv', '')
                
                for day_idx, pred_value in enumerate(ensemble_pred):
                    all_predictions.append({
                        'test_file': test_prefix,
                        'venue': venue,
                        'day': f"{test_prefix}+{day_idx+1}일",
                        'lstm_pred': lstm_pred[day_idx],
                        'cluster_pred': cluster_pred[day_idx],
                        'ensemble_pred': pred_value,
                        'weight_used': weight
                    })
        
        return all_predictions
    
    def create_submission_file(self, predictions, sample_submission_path='./sample_submission.csv'):
        """제출 파일 생성"""
        
        sample_submission = pd.read_csv(sample_submission_path)
        submission = sample_submission.copy()
        
        # 예측 결과를 딕셔너리로 변환
        pred_dict = {}
        for pred in predictions:
            key = (pred['day'], pred['venue'])
            pred_dict[key] = pred['ensemble_pred']
        
        # 제출 파일에 매핑
        for row_idx in submission.index:
            date = submission.loc[row_idx, '영업일자']
            for col in submission.columns[1:]:  # 메뉴명들
                value = pred_dict.get((date, col), 0)
                submission.loc[row_idx, col] = max(0, value)
        
        return submission
        # 현재 앙상블 코드에서 개별 모델 예측값을 분리해서 저장
    def extract_individual_predictions(self, test_files):
        lstm_preds = {}
        cluster_preds = {}
        
        for test_file in test_files:
            lstm_result = self._get_lstm_predictions(test_file)
            cluster_result = self._get_cluster_predictions(test_file)
            
            # 딕셔너리 형태로 저장
            for item in lstm_result:
                key = (item['venue'], item['day'])
                lstm_preds[key] = item['prediction']
                
            for item in cluster_result:
                key = (item['venue'], item['day'])
                cluster_preds[key] = item['prediction']
        
        return lstm_preds, cluster_preds

# 사용 예시
def run_ensemble_pipeline():
    """전체 앙상블 파이프라인 실행"""
    
    print("🚀 앙상블 파이프라인 시작!")
    
    # 1. 앙상블 모델 초기화
    ensemble = EnsemblePredictor(high_weight_venues=['담하', '미라시아'])
    
    # 2. 두 모델 훈련
    ensemble.load_and_train_models('./train/train.csv')
    
    # 3. 최적 가중치 찾기 (선택사항)
    ensemble.find_optimal_weights_on_validation()
    
    # 4. 테스트 예측
    test_files = sorted(glob.glob('./TEST_*.csv'))
    predictions = ensemble.predict_ensemble(test_files)
    
    # 5. 제출 파일 생성
    submission = ensemble.create_submission_file(predictions)
    
    # 6. 저장
    output_path = './ens_sub_3.csv'
    submission.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 앙상블 완료! 저장 위치: {output_path}")
    
    # 7. 결과 분석
    print("\n📊 앙상블 결과 분석:")
    pred_df = pd.DataFrame(predictions)
    
    print(f"총 예측 수: {len(pred_df)}")
    print(f"평균 LSTM 예측: {pred_df['lstm_pred'].mean():.2f}")
    print(f"평균 클러스터링 예측: {pred_df['cluster_pred'].mean():.2f}")
    print(f"평균 앙상블 예측: {pred_df['ensemble_pred'].mean():.2f}")
    
    weight_stats = pred_df['weight_used'].value_counts()
    print(f"사용된 가중치 분포:\n{weight_stats}")
    
    return submission, pred_df

# ===== 실행 방법 =====
"""
1. 이 코드를 ensemble_model.py로 저장
2. 기존 모델 코드들이 같은 디렉토리에 있는지 확인
3. 다음 명령 실행:

from ensemble_model import run_ensemble_pipeline
submission, predictions = run_ensemble_pipeline()

또는 단계별로:

ensemble = EnsemblePredictor()
ensemble.load_and_train_models()
test_files = glob.glob('./test/TEST_*.csv')
predictions = ensemble.predict_ensemble(test_files)
submission = ensemble.create_submission_file(predictions)
submission.to_csv('ensemble_result.csv', index=False)
"""

# if __name__ == "__main__":
#     submission, predictions = run_ensemble_pipeline()