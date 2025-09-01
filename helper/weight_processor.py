import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

class WeightedTrainingPreprocessor:
    """
    가중치 기반 학습을 위한 전처리기
    1. 화담숲 매장들의 휴업 기간 (12-2월)
    2. 메뉴 출시 전 기간 (초반 100일+ 연속 0)
    """
    
    def __init__(self, 
                 min_consecutive_zeros=100,
                 closure_weight=0.1,
                 pre_launch_weight=0.05,
                 verbose=True):
        """
        Parameters:
        - min_consecutive_zeros: 출시 전 판단을 위한 최소 연속 0일 수
        - closure_weight: 휴업 기간 가중치
        - pre_launch_weight: 출시 전 기간 가중치  
        - verbose: 진행상황 출력 여부
        """
        self.min_consecutive_zeros = min_consecutive_zeros
        self.closure_weight = closure_weight
        self.pre_launch_weight = pre_launch_weight
        self.verbose = verbose
        
        # 분석 결과 저장
        self.launch_analysis = {}
        self.closure_analysis = {}
        self.weight_stats = {}
    
    def analyze_menu_launches(self, train_df):
        """메뉴별 출시점 분석"""
        
        if self.verbose:
            print("메뉴 출시점 분석 시작...")
        
        target_stores = ['담하', '라그로타', '미라시아']
        launch_info = {}
        
        grouped = train_df.groupby(['store', 'menu'])
        iterator = tqdm(grouped, desc="메뉴 출시점 분석") if self.verbose else grouped
        
        for (store, menu), group in iterator:
            # 대상 매장만 분석
            if store not in target_stores:
                continue
            
            group = group.sort_values('date').reset_index(drop=True)
            sales = group['sales'].values
            
            # 초반 연속 0 확인
            consecutive_zeros = 0
            first_sale_idx = None
            
            for i, sale in enumerate(sales):
                if sale == 0:
                    consecutive_zeros += 1
                else:
                    first_sale_idx = i
                    break
            
            # 출시 전 기간 판단
            is_pre_launch_pattern = (
                consecutive_zeros >= self.min_consecutive_zeros and
                first_sale_idx is not None
            )
            
            if is_pre_launch_pattern:
                launch_info[(store, menu)] = {
                    'pre_launch_days': consecutive_zeros,
                    'first_sale_date': group.iloc[first_sale_idx]['date'],
                    'total_days': len(group),
                    'post_launch_days': len(group) - consecutive_zeros
                }
                
                if self.verbose and consecutive_zeros > 150:  # 큰 케이스만 출력
                    tqdm.write(f"{store}_{menu}: {consecutive_zeros}일 출시 전")
        
        self.launch_analysis = launch_info
        
        if self.verbose:
            print(f"출시 전 패턴 발견: {len(launch_info)}개 메뉴")
        
        return launch_info
    
    def analyze_seasonal_closures(self, train_df):
        """계절적 휴업 분석"""
        
        closure_stores = ['화담숲주막', '화담숲카페']
        closure_months = [12, 1, 2]
        
        closure_info = {}
        
        for store in closure_stores:
            store_data = train_df[train_df['store'] == store]
            
            # 휴업 기간 데이터 확인
            closure_data = store_data[store_data['month'].isin(closure_months)]
            total_closure_days = len(closure_data)
            zero_days = (closure_data['sales'] == 0).sum()
            
            closure_info[store] = {
                'total_closure_days': total_closure_days,
                'zero_days': zero_days,
                'closure_ratio': zero_days / total_closure_days if total_closure_days > 0 else 0
            }
        
        self.closure_analysis = closure_info
        
        if self.verbose:
            print("\n=== 휴업 기간 분석 ===")
            for store, info in closure_info.items():
                print(f"{store}: {info['total_closure_days']}일 중 {info['zero_days']}일 휴업 "
                     f"({info['closure_ratio']:.1%})")
        
        return closure_info
    
    def create_sample_weights(self, train_df):
        """샘플별 가중치 생성"""
        
        if self.verbose:
            print("\n샘플 가중치 생성 중...")
        
        # 기본 가중치 (모두 1.0)
        weights = np.ones(len(train_df))
        
        # 통계 초기화
        stats = {
            'total_samples': len(train_df),
            'closure_samples': 0,
            'pre_launch_samples': 0,
            'normal_samples': 0
        }
        
        for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="가중치 계산"):
            store = row['store']
            menu = row['menu']
            month = row['month']
            date = row['date']
            
            current_weight = 1.0
            
            # 1. 휴업 기간 가중치 적용
            if store in ['화담숲주막', '화담숲카페'] and month in [12, 1, 2]:
                current_weight = self.closure_weight
                stats['closure_samples'] += 1
            
            # 2. 출시 전 기간 가중치 적용
            elif (store, menu) in self.launch_analysis:
                launch_info = self.launch_analysis[(store, menu)]
                first_sale_date = launch_info['first_sale_date']
                
                if date < first_sale_date:
                    current_weight = self.pre_launch_weight
                    stats['pre_launch_samples'] += 1
                else:
                    stats['normal_samples'] += 1
            else:
                stats['normal_samples'] += 1
            
            weights[idx] = current_weight
        
        self.weight_stats = stats
        
        if self.verbose:
            self._print_weight_statistics()
        
        return weights
    
    def _print_weight_statistics(self):
        """가중치 통계 출력"""
        
        stats = self.weight_stats
        
        print(f"\n=== 가중치 통계 ===")
        print(f"전체 샘플: {stats['total_samples']:,}개")
        print(f"일반 가중치 (1.0): {stats['normal_samples']:,}개 "
             f"({stats['normal_samples']/stats['total_samples']:.1%})")
        print(f"휴업 가중치 ({self.closure_weight}): {stats['closure_samples']:,}개 "
             f"({stats['closure_samples']/stats['total_samples']:.1%})")
        print(f"출시 전 가중치 ({self.pre_launch_weight}): {stats['pre_launch_samples']:,}개 "
             f"({stats['pre_launch_samples']/stats['total_samples']:.1%})")
        
        # 효과적인 샘플 수 계산
        effective_samples = (
            stats['normal_samples'] +
            stats['closure_samples'] * self.closure_weight +
            stats['pre_launch_samples'] * self.pre_launch_weight
        )
        
        print(f"효과적인 샘플 수: {effective_samples:,.0f}개 "
             f"({effective_samples/stats['total_samples']:.1%})")
    
    def process_training_data(self, train_df):
        """전체 전처리 파이프라인 실행"""
        
        if self.verbose:
            print("=== 가중치 기반 훈련 데이터 전처리 ===")
        
        # 1. 메뉴 출시점 분석
        self.analyze_menu_launches(train_df)
        
        # 2. 휴업 기간 분석
        self.analyze_seasonal_closures(train_df)
        
        # 3. 샘플 가중치 생성
        weights = self.create_sample_weights(train_df)
        
        # 4. 가중치 정보를 데이터프레임에 추가
        processed_df = train_df.copy()
        processed_df['sample_weight'] = weights
        
        # 5. 추가 피처 생성
        processed_df = self._add_weight_related_features(processed_df)
        
        return processed_df, weights
    
    def _add_weight_related_features(self, df):
        """가중치 관련 피처 추가"""
        
        # 휴업 기간 피처
        df['is_closure_period'] = (
            (df['store'].isin(['화담숲주막', '화담숲카페'])) & 
            (df['month'].isin([12, 1, 2]))
        ).astype(int)
        
        # 출시 전 기간 피처
        df['is_pre_launch'] = 0
        for (store, menu), launch_info in self.launch_analysis.items():
            mask = (df['store'] == store) & (df['menu'] == menu) & \
                   (df['date'] < launch_info['first_sale_date'])
            df.loc[mask, 'is_pre_launch'] = 1
        
        # 매장별 특수 상태 피처
        df['store_special_status'] = 0  # 0: 정상, 1: 휴업, 2: 출시전
        df.loc[df['is_closure_period'] == 1, 'store_special_status'] = 1
        df.loc[df['is_pre_launch'] == 1, 'store_special_status'] = 2
        
        return df
    
    def get_analysis_summary(self):
        """분석 결과 요약 반환"""
        
        summary = {
            'launch_patterns': len(self.launch_analysis),
            'closure_analysis': self.closure_analysis,
            'weight_stats': self.weight_stats,
            'total_downweighted_samples': (
                self.weight_stats.get('closure_samples', 0) +
                self.weight_stats.get('pre_launch_samples', 0)
            )
        }
        
        return summary


class WeightedClusterModel:
    """가중치를 지원하는 클러스터 모델 래퍼"""
    
    def __init__(self, base_model_class):
        self.base_model_class = base_model_class
        self.base_model = None
        self.preprocessor = WeightedTrainingPreprocessor()
        
    def fit(self, train_df):
        """가중치 기반 학습"""
        
        print("=== 가중치 기반 클러스터 모델 학습 ===")
        
        # 1. 가중치 전처리
        processed_df, sample_weights = self.preprocessor.process_training_data(train_df)
        
        # 2. 기본 모델 초기화
        self.base_model = self.base_model_class()
        
        # 3. 가중치 적용 학습
        if hasattr(self.base_model, 'fit_with_weights'):
            # 모델이 가중치를 직접 지원하는 경우
            self.base_model.fit_with_weights(processed_df, sample_weights)
        else:
            # 가중치 샘플링을 통한 간접 적용
            self._fit_with_weighted_sampling(processed_df, sample_weights)
    
    def _fit_with_weighted_sampling(self, df, weights):
        """가중치 샘플링을 통한 학습"""
        
        # 가중치에 따른 샘플링 확률 계산
        sampling_probs = weights / weights.sum()
        
        # 가중치 기반 샘플링 (복원추출)
        n_samples = len(df)
        sampled_indices = np.random.choice(
            len(df), 
            size=n_samples, 
            p=sampling_probs, 
            replace=True
        )
        
        # 샘플링된 데이터로 학습
        sampled_df = df.iloc[sampled_indices].reset_index(drop=True)
        self.base_model.fit(sampled_df)
    
    def predict(self, test_df):
        """예측"""
        return self.base_model.predict(test_df)


# 사용 예시
def run_weighted_training_pipeline():
    """가중치 기반 훈련 파이프라인 실행"""
    
    # 데이터 로드
    train_df = pd.read_csv('./train/train.csv')
    train_df['date'] = pd.to_datetime(train_df['영업일자'])
    train_df[['store', 'menu']] = train_df['영업장명_메뉴명'].str.split('_', expand=True, n=1)
    train_df['sales'] = train_df['매출수량']
    train_df['month'] = train_df['date'].dt.month
    train_df['day_of_week'] = train_df['date'].dt.dayofweek
    train_df['is_weekend'] = train_df['day_of_week'].isin([5, 6])
    
    # 가중치 기반 전처리
    preprocessor = WeightedTrainingPreprocessor(
        min_consecutive_zeros=100,
        closure_weight=0.1,
        pre_launch_weight=0.05
    )
    
    processed_df, weights = preprocessor.process_training_data(train_df)
    
    # 분석 결과 출력
    summary = preprocessor.get_analysis_summary()
    print(f"\n총 {summary['total_downweighted_samples']:,}개 샘플의 가중치가 낮아졌습니다.")
    
    return processed_df, weights, preprocessor


if __name__ == "__main__":
    # 테스트 실행
    processed_df, weights, preprocessor = run_weighted_training_pipeline()
    
    print("\n가중치 기반 전처리 완료!")
    print("다음 단계: 기존 모델에 sample_weight 적용")