import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

# ----------------------------
# Logging
# ----------------------------
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("debug.log", encoding="utf-8"),  # 파일: 무제한
        logging.StreamHandler()                              # 노트북 셀: 요약만
    ]
)
# ----------------------------

class ClusterBasedForecastingModel:
    """클러스터 기반 예측 모델 - 실제 활용 버전"""
    
    def __init__(self):
        self.menu_clusters = {}
        self.cluster_models = {}
        self.cluster_features = {}
        self.store_patterns = {}
    
    def classify_menu_stability(self, train_df):
        """메뉴별 안정성 분류"""
        
        menu_stability = {}
        
        for (store, menu), group in train_df.groupby(['store', 'menu']):
            if len(group) < 30:  # 데이터 부족시 제외
                continue
                
            zero_ratio = (group['sales'] == 0).mean()
            total_sales = group['sales'].sum()
            
            # 안정성 분류 기준
            if zero_ratio < 0.1:
                stability = 'very_stable'
            elif zero_ratio < 0.3:
                stability = 'stable'  
            elif zero_ratio < 0.6:
                stability = 'moderate'
            else:
                stability = 'unstable'
                
            menu_stability[(store, menu)] = {
                'stability': stability,
                'zero_ratio': zero_ratio,
                'total_sales': total_sales,
                'avg_sales': group['sales'].mean(),
                'volatility': group['sales'].std() / (group['sales'].mean() + 1e-8)
            }
        
        self.menu_stability_map = menu_stability
        return menu_stability
    
    def analyze_and_cluster_menus(self, train_df):
        """메뉴 클러스터링 및 클러스터별 특성 분석"""
        
        menu_features = []
        menu_names = []
        
        logging.debug("메뉴 클러스터링 시작...")
        
        # 1. 각 메뉴별 특성 추출
        grouped = train_df.groupby(['store', 'menu'])
        for (store, menu), group in tqdm(grouped, desc="메뉴 특성 추출"):
            if len(group) < 20:  # 충분한 데이터가 있는 메뉴만
                continue
                
            # 메뉴별 특성 벡터 생성
            features = {
                'avg_sales': group['sales'].mean(),
                'std_sales': group['sales'].std(),
                'zero_ratio': (group['sales'] == 0).mean(),
                'max_sales': group['sales'].max(),
                'cv': group['sales'].std() / (group['sales'].mean() + 1e-8),  # 변동계수
                'weekend_boost': group[group['is_weekend']]['sales'].mean() / (group['sales'].mean() + 1e-8),
                'seasonality_strength': self._calculate_seasonality(group),
                'trend_strength': self._calculate_trend(group['sales'].values),
            }
            
            # 메뉴명 기반 특성
            menu_lower = str(menu).lower()
            features.update({
                'is_main_dish': int(any(x in menu_lower for x in ['불고기', '갈비', '찌개', '국밥', '정식'])),
                'is_drink': int(any(x in menu_lower for x in ['콜라', '맥주', '소주', '커피', '음료', '차'])),
                'is_premium': int(any(x in menu_lower for x in ['한우', 'aus', '프리미엄'])),
                'is_group': int('단체' in menu_lower),
                'is_brunch': int('브런치' in menu_lower),
            })
            
            menu_features.append(list(features.values()))
            menu_names.append((store, menu))
        
        if len(menu_features) < 5:
            logging.debug("클러스터링에 충분한 메뉴가 없음")
            return {}
        
        # 2. K-means 클러스터링
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(menu_features)
        
        # 최적 클러스터 수 결정
        n_clusters = min(6, max(3, len(menu_features) // 8))
        logging.debug(f"클러스터 수: {n_clusters}")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 3. 클러스터별 메뉴 그룹화 및 특성 분석
        cluster_groups = {}
        cluster_characteristics = {}
        
        for cluster_id in tqdm(range(n_clusters), desc="클러스터 특성 분석"):
            cluster_mask = cluster_labels == cluster_id
            cluster_menus = [menu_names[i] for i in range(len(menu_names)) if cluster_mask[i]]
            cluster_groups[cluster_id] = cluster_menus
            
            # 클러스터 특성 계산
            cluster_data_list = []
            for store, menu in cluster_menus:
                menu_data = train_df[(train_df['store'] == store) & (train_df['menu'] == menu)]
                cluster_data_list.append(menu_data)
            
            if cluster_data_list:
                cluster_combined = pd.concat(cluster_data_list, ignore_index=True)
                
                cluster_characteristics[cluster_id] = {
                    'avg_sales': cluster_combined['sales'].mean(),
                    'zero_ratio': (cluster_combined['sales'] == 0).mean(),
                    'volatility': cluster_combined['sales'].std() / (cluster_combined['sales'].mean() + 1e-8),
                    'weekend_effect': cluster_combined[cluster_combined['is_weekend']]['sales'].mean() / 
                                    (cluster_combined[~cluster_combined['is_weekend']]['sales'].mean() + 1e-8),
                    'menu_count': len(cluster_menus),
                    'dominant_type': self._get_dominant_menu_type(cluster_menus)
                }
        
        self.menu_clusters = cluster_groups
        self.cluster_features = cluster_characteristics
        
        # 클러스터 정보 출력
        for cluster_id, info in cluster_characteristics.items():
            logging.debug(f"클러스터 {cluster_id}: {info['menu_count']}개 메뉴, "
                  f"평균매출 {info['avg_sales']:.1f}, 타입: {info['dominant_type']}")
        
        return cluster_groups
    
    def _calculate_seasonality(self, group):
        """계절성 강도 계산"""
        if 'month' not in group.columns:
            return 0
        monthly_avg = group.groupby('month')['sales'].mean()
        if len(monthly_avg) < 2:
            return 0
        return monthly_avg.std() / (monthly_avg.mean() + 1e-8)
    
    def _calculate_trend(self, sales):
        """트렌드 강도 계산"""
        if len(sales) < 2:
            return 0
        x = np.arange(len(sales))
        return abs(np.polyfit(x, sales, 1)[0])
    
    def _get_dominant_menu_type(self, cluster_menus):
        """클러스터의 주요 메뉴 타입 결정"""
        type_counts = {'main': 0, 'drink': 0, 'premium': 0, 'group': 0, 'brunch': 0, 'other': 0}
        
        for store, menu in cluster_menus:
            menu_lower = str(menu).lower()
            if any(x in menu_lower for x in ['불고기', '갈비', '찌개', '국밥', '정식']):
                type_counts['main'] += 1
            elif any(x in menu_lower for x in ['콜라', '맥주', '소주', '커피', '음료']):
                type_counts['drink'] += 1
            elif any(x in menu_lower for x in ['한우', 'aus', '프리미엄']):
                type_counts['premium'] += 1
            elif '단체' in menu_lower:
                type_counts['group'] += 1
            elif '브런치' in menu_lower:
                type_counts['brunch'] += 1
            else:
                type_counts['other'] += 1
        
        return max(type_counts, key=type_counts.get)
    
    def get_menu_cluster(self, store, menu):
        """특정 메뉴의 클러스터 ID 반환"""
        for cluster_id, menus in self.menu_clusters.items():
            if (store, menu) in menus:
                return cluster_id
        return -1  # 클러스터에 없음
    
    def create_cluster_features(self, df, mode='train'):
        """클러스터 정보를 활용한 피처 생성"""
        
        sequences = []
        targets = []
        metadata = []
        
        for (store, menu), group in df.groupby(['store', 'menu']):
            group = group.sort_values('date').reset_index(drop=True)
            
            min_length = 28 + (7 if mode == 'train' else 0)
            if len(group) < min_length:
                continue
            
            # 클러스터 정보 가져오기
            cluster_id = self.get_menu_cluster(store, menu)
            cluster_info = self.cluster_features.get(cluster_id, {})
            
            if mode == 'predict':
                seq_data = group.tail(28)
                features = self._create_cluster_based_features(seq_data, store, menu, cluster_id, cluster_info)
                sequences.append(features)
                metadata.append({'store': store, 'menu': menu, 'cluster': cluster_id})
                
            else:
                for i in range(len(group) - min_length + 1):
                    seq_data = group.iloc[i:i+28]
                    target_data = group.iloc[i+28:i+35]
                    
                    features = self._create_cluster_based_features(seq_data, store, menu, cluster_id, cluster_info)
                    target = target_data['sales'].values
                    
                    sequences.append(features)
                    targets.append(target)
                    metadata.append({'store': store, 'menu': menu, 'cluster': cluster_id})
        
        X = np.array(sequences) if sequences else np.empty((0, 60))
        y = np.array(targets) if targets else np.empty((0, 7))
        
        return X, y, metadata
    
    def _create_cluster_based_features(self, seq_data, store, menu, cluster_id, cluster_info):
        """클러스터 정보를 활용한 피처 생성"""
        
        sales = seq_data['sales'].values
        
        # 1. 기본 통계 피처
        basic_features = [
            np.mean(sales), np.median(sales), np.std(sales),
            np.min(sales), np.max(sales),
            np.mean(sales[-7:]), np.mean(sales[-14:]),
            sales[-1] if len(sales) > 0 else 0,
            (sales == 0).mean(),
            self._calculate_trend(sales),
        ]
        
        # 2. 클러스터 기반 피처 (핵심!)
        cluster_features = [
            cluster_id if cluster_id != -1 else 0,  # 클러스터 ID
            cluster_info.get('avg_sales', 0) / (np.mean(sales) + 1e-8),  # 클러스터 평균 대비 비율
            cluster_info.get('zero_ratio', 0),  # 클러스터 0매출 비율
            cluster_info.get('volatility', 0),  # 클러스터 변동성
            cluster_info.get('weekend_effect', 1),  # 클러스터 주말효과
        ]
        
        # 3. 클러스터별 상대적 성과
        cluster_avg = cluster_info.get('avg_sales', np.mean(sales))
        relative_performance = [
            np.mean(sales) / (cluster_avg + 1e-8),  # 현재 성과 vs 클러스터 평균
            (sales > cluster_avg).mean(),  # 클러스터 평균 초과 비율
            np.std(sales) / (cluster_info.get('volatility', 1) * cluster_avg + 1e-8),  # 상대적 변동성
        ]
        
        # 4. 클러스터 타입별 피처
        cluster_type = cluster_info.get('dominant_type', 'other')
        type_features = [
            int(cluster_type == 'main'),
            int(cluster_type == 'drink'), 
            int(cluster_type == 'premium'),
            int(cluster_type == 'group'),
            int(cluster_type == 'brunch'),
        ]
        
        # 5. 시간 기반 피처
        time_features = [
            seq_data['month'].iloc[-1],
            seq_data['day_of_week'].iloc[-1],
            seq_data['is_weekend'].sum(),
            np.sin(2 * np.pi * seq_data['month'].iloc[-1] / 12),
            np.cos(2 * np.pi * seq_data['month'].iloc[-1] / 12),
            np.sin(2 * np.pi * seq_data['day_of_week'].iloc[-1] / 7),
            np.cos(2 * np.pi * seq_data['day_of_week'].iloc[-1] / 7),
        ]
        
        # 6. 업장 기반 피처
        store_features = self._get_store_features(store)
        
        # 7. 라그 피처
        lag_features = [
            sales[-1] if len(sales) >= 1 else 0,
            sales[-7] if len(sales) >= 7 else 0,
            sales[-14] if len(sales) >= 14 else 0,
        ]
        
        # 모든 피처 결합
        all_features = (basic_features + cluster_features + relative_performance + 
                       type_features + time_features + store_features + lag_features)
        
        # NaN 처리 및 크기 조정
        all_features = [0.0 if pd.isna(x) or np.isinf(x) else float(x) for x in all_features]
        
        # 60개 피처로 맞추기
        while len(all_features) < 60:
            all_features.append(0.0)
        all_features = all_features[:60]
        
        return all_features
    
    def _get_store_features(self, store):
        """업장 원-핫 인코딩"""
        stores = ['담하', '미라시아', '포레스트릿', '카페테리아', '화담숲주막', 
                 '화담숲카페', '느티나무 셀프BBQ', '연회장', '라그로타']
        return [int(store == s) for s in stores]
    
    def fit(self, train_df):
        """클러스터 기반 모델 학습"""
        logging.debug("클러스터 기반 모델 학습 시작...")
        
        # 1. 메뉴 클러스터링
        self.analyze_and_cluster_menus(train_df)
        
        # 2. 클러스터 기반 피처 생성
        X, y, metadata = self.create_cluster_features(train_df, mode='train')
        
        if len(X) == 0:
            logging.debug("학습 데이터 부족")
            return
        
        logging.debug(f"클러스터 피처: {X.shape}, 타겟: {y.shape}")
        
        # 3. 전체 모델 학습
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        
        # 매출 여부 분류기
        has_sales = (y > 0).any(axis=1)
        self.zero_classifier = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight='balanced'
        )
        self.zero_classifier.fit(X, has_sales)
        
        # 4. 클러스터별 전용 모델 학습
        sales_mask = has_sales
        if sales_mask.sum() > 20:
            
            # 전체 모델
            self.global_model = MultiOutputRegressor(
                LGBMRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, 
                            random_state=42, verbosity=-1)
            )
            with tqdm(total=100, desc="전체 모델 학습") as pbar:
                self.global_model.fit(X[sales_mask], y[sales_mask])
                pbar.update(100)
            
            # 클러스터별 모델 학습 (가장 유용한 부분)
            logging.debug("클러스터별 전용 모델 학습 중...")
            self.cluster_models = {}
            
            # 학습할 클러스터들 필터링
            valid_clusters = []
            for cluster_id in self.menu_clusters.keys():
                cluster_mask = np.array([meta['cluster'] == cluster_id for meta in metadata])
                cluster_sales_mask = sales_mask & cluster_mask
                if cluster_sales_mask.sum() > 10:
                    valid_clusters.append((cluster_id, cluster_sales_mask))

            # tqdm으로 클러스터별 모델 학습 진행률 표시
            for cluster_id, cluster_sales_mask in tqdm(valid_clusters, desc="클러스터 모델"):
                from xgboost import XGBRegressor
                
                cluster_model = MultiOutputRegressor(
                    XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08,
                            random_state=42, verbosity=0)
                )
                
                cluster_model.fit(X[cluster_sales_mask], y[cluster_sales_mask])
                self.cluster_models[cluster_id] = cluster_model
                
                # 클러스터 정보 표시
                cluster_info = self.cluster_features[cluster_id]
                tqdm.write(f"클러스터 {cluster_id} 완료: {cluster_info['dominant_type']}, "
                        f"{cluster_sales_mask.sum()}개 샘플")
        
        logging.debug(f"전체 모델 + {len(self.cluster_models)}개 클러스터 모델 학습 완료")
    
    def predict(self, test_df):
        """클러스터 기반 예측"""
        if not hasattr(self, 'global_model'):
            logging.debug("모델이 학습되지 않음")
            return np.zeros((len(test_df), 7)), []
        
        X, _, metadata = self.create_cluster_features(test_df, mode='predict')
        
        if len(X) == 0:
            return np.zeros((0, 7)), []
        
        logging.debug("매출 여부 예측 중...")
        with tqdm(total=100, desc="분류 예측") as pbar:
            has_sales_prob = self.zero_classifier.predict_proba(X)[:, 1]
            pbar.update(100)
        
        # 예측 수행
        predictions = np.zeros((len(X), 7))
        
        for i in tqdm(range(len(X)), desc="예측 진행"):
            cluster_id = metadata[i]['cluster']
            
            # 클러스터 전용 모델이 있으면 사용, 없으면 전체 모델 사용
            if cluster_id in self.cluster_models:
                cluster_pred = self.cluster_models[cluster_id].predict(X[i:i+1])
                global_pred = self.global_model.predict(X[i:i+1])
                
                # 클러스터 모델과 전체 모델의 가중 평균
                cluster_weight = 0.7
                predictions[i] = cluster_weight * cluster_pred[0] + (1-cluster_weight) * global_pred[0]
            else:
                # 클러스터 모델이 없으면 전체 모델만 사용
                predictions[i] = self.global_model.predict(X[i:i+1])[0]
        
        # Zero-inflation 적용
        threshold = 0.3
        zero_mask = has_sales_prob < threshold
        predictions[zero_mask] = 0
        
        # 후처리
        predictions = np.maximum(predictions, 0)
        # predictions = np.round(predictions, 2)
        
        return predictions, metadata

# 실행 함수
def run_cluster_based_pipeline():
    """클러스터 기반 파이프라인 실행"""
    
    # 데이터 로드
    train_df = pd.read_csv('./train/train.csv')
    train_df['date'] = pd.to_datetime(train_df['영업일자'])
    train_df[['store', 'menu']] = train_df['영업장명_메뉴명'].str.split('_', expand=True, n=1)
    train_df['sales'] = train_df['매출수량']
    train_df['month'] = train_df['date'].dt.month
    train_df['day_of_week'] = train_df['date'].dt.dayofweek
    train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6])
    
    # 클러스터 기반 모델 학습
    cluster_model = ClusterBasedForecastingModel()
    cluster_model.fit(train_df)
    
    # 제출 파일 생성
    submission = pd.read_csv('sample_submission.csv')
    
    import glob
    test_files = sorted(glob.glob('TEST_*.csv'))
    
    for test_idx, test_file in enumerate(test_files):
        logging.debug(f"클러스터 기반 처리: {test_file}")
        
        test_df = pd.read_csv(test_file)
        test_df['date'] = pd.to_datetime(test_df['영업일자'])
        test_df[['store', 'menu']] = test_df['영업장명_메뉴명'].str.split('_', expand=True, n=1)
        test_df['sales'] = test_df['매출수량']
        test_df['month'] = test_df['date'].dt.month
        test_df['day_of_week'] = test_df['date'].dt.dayofweek
        test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6])
        
        # 예측
        predictions, metadata = cluster_model.predict(test_df)
        
        # 제출 파일에 매핑
        test_case = f"TEST_{test_idx:02d}"
        test_rows = submission[submission['영업일자'].str.contains(test_case, na=False)].index.tolist()
        
        if len(test_rows) == 7 and len(predictions) > 0:
            numeric_cols = submission.select_dtypes(include=[np.number]).columns
            
            for day_idx, row_idx in enumerate(test_rows):
                for col_idx, col in enumerate(numeric_cols):
                    if col_idx < len(predictions):
                        pred_value = predictions[col_idx, day_idx]
                        submission.loc[row_idx, col] = max(0.0, pred_value)
    
    submission.to_csv('cluster_based_submission.csv', index=False)
    logging.debug("cluster_based_submission.csv 생성 완료!")
    
    return submission

# 실행
#cluster_submission = run_cluster_based_pipeline