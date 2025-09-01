import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SimpleEffectiveClipping:
    """단순하고 효과적인 클리핑 - 특정 업장 제외 기능 추가"""
    
    def __init__(self, train_csv_path: str = './train.csv', exclude_venues: List[str] = None):
        """
        Args:
            train_csv_path: 훈련 데이터 경로
            exclude_venues: 클리핑에서 제외할 업장 리스트 (예: ['담하', '미라시아'])
        """
        self.train_path = train_csv_path
        self.menu_bounds = {}
        self.exclude_venues = exclude_venues if exclude_venues else []
        print(f"클리핑 제외 업장: {self.exclude_venues}")
    
    def is_venue_excluded(self, menu_name: str) -> bool:
        """메뉴가 제외할 업장에 속하는지 확인"""
        if not self.exclude_venues:
            return False
        
        # '영업장명_메뉴명' 형태에서 업장명 추출
        venue_name = menu_name.split('_')[0] if '_' in menu_name else menu_name
        return venue_name in self.exclude_venues
    
    def calculate_menu_bounds(self):
        """메뉴별 클리핑 경계값 계산 - 단순한 방법들 제공"""
        
        train_df = pd.read_csv(self.train_path)
        excluded_menus = []
        included_menus = []
        
        for menu in train_df['영업장명_메뉴명'].unique():
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu]['매출수량'].values
            
            if len(menu_data) >= 10:
                if self.is_venue_excluded(menu):
                    # 제외 업장은 경계값을 매우 높게 설정 (사실상 클리핑 안함)
                    bounds = {
                        'historical_max': float(np.max(menu_data)) * 10,  # 10배로 설정
                        'p99': float(np.max(menu_data)) * 10,
                        'p95': float(np.max(menu_data)) * 10,
                        'p90': float(np.max(menu_data)) * 10,
                        'p85': float(np.max(menu_data)) * 10,
                        'p80': float(np.max(menu_data)) * 10,
                        'p75': float(np.max(menu_data)) * 10,
                        'iqr_15': float(np.max(menu_data)) * 10,
                        'iqr_10': float(np.max(menu_data)) * 10,
                        'mean_plus_2std': float(np.max(menu_data)) * 10,
                        'mean_plus_3std': float(np.max(menu_data)) * 10,
                        'zero_ratio': float(np.mean(menu_data == 0)),
                        'mean': float(np.mean(menu_data)),
                        'count': len(menu_data),
                        'excluded': True  # 제외 표시
                    }
                    excluded_menus.append(menu)
                else:
                    # 일반 메뉴는 기존과 동일하게 처리
                    bounds = {
                        'historical_max': float(np.max(menu_data)),
                        'p99': float(np.percentile(menu_data, 99)),
                        'p95': float(np.percentile(menu_data, 95)),
                        'p90': float(np.percentile(menu_data, 90)),
                        'p85': float(np.percentile(menu_data, 85)),
                        'p80': float(np.percentile(menu_data, 80)),
                        'p75': float(np.percentile(menu_data, 75)),
                    }
                    
                    # IQR 기반 이상치 경계
                    q1 = np.percentile(menu_data, 25)
                    q3 = np.percentile(menu_data, 75)
                    iqr = q3 - q1
                    bounds['iqr_15'] = max(1, q3 + 1.5 * iqr)
                    bounds['iqr_10'] = max(1, q3 + 1.0 * iqr)
                    
                    # 평균 기반 경계
                    mean_val = np.mean(menu_data)
                    std_val = np.std(menu_data)
                    bounds['mean_plus_2std'] = mean_val + 2 * std_val
                    bounds['mean_plus_3std'] = mean_val + 3 * std_val
                    
                    # 기본 통계
                    bounds['zero_ratio'] = float(np.mean(menu_data == 0))
                    bounds['mean'] = float(mean_val)
                    bounds['count'] = len(menu_data)
                    bounds['excluded'] = False
                    included_menus.append(menu)
                
                self.menu_bounds[menu] = bounds
        
        print(f"클리핑 제외 메뉴 수: {len(excluded_menus)}")
        print(f"클리핑 적용 메뉴 수: {len(included_menus)}")
        if excluded_menus[:5]:  # 처음 5개만 출력
            print(f"제외 메뉴 예시: {excluded_menus[:5]}")
        
        return self.menu_bounds
    
    def apply_clipping_method(self, predictions_df: pd.DataFrame, method_key: str) -> pd.DataFrame:
        """통합 클리핑 메서드"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        clipped_count = 0
        excluded_count = 0
        
        for col in menu_columns:
            if col in self.menu_bounds:
                if self.menu_bounds[col].get('excluded', False):
                    # 제외 업장은 최소값만 1로 보정
                    clipped_df[col] = np.maximum(clipped_df[col].values, 1)
                    excluded_count += 1
                else:
                    # 일반 메뉴는 지정된 방법으로 클리핑
                    upper_bound = self.menu_bounds[col][method_key]
                    clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
                    clipped_count += 1
            else:
                # 경계값이 없는 메뉴는 최소값만 1로 보정
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        print(f"  → 클리핑 적용: {clipped_count}개, 제외: {excluded_count}개 메뉴")
        
        return clipped_df
    
    def apply_method_1_historical_max(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 1: 역사적 최댓값 기준 (가장 관대)"""
        return self.apply_clipping_method(predictions_df, 'historical_max')
    
    def apply_method_2_iqr_standard(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 2: 표준 IQR 이상치 기준 (통계적으로 가장 타당)"""
        return self.apply_clipping_method(predictions_df, 'iqr_15')
    
    def apply_method_3_p90(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 3: 90% 백분위수 기준 (보수적)"""
        return self.apply_clipping_method(predictions_df, 'p90')
    
    def apply_method_4_p85(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 4: 85% 백분위수 기준 (더 보수적)"""
        return self.apply_clipping_method(predictions_df, 'p85')
    
    def apply_method_5_p80(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 5: 80% 백분위수 기준 (매우 보수적)"""
        return self.apply_clipping_method(predictions_df, 'p80')
    
    def apply_method_6_conservative_iqr(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 6: 보수적 IQR 기준 (Q3 + 1.0*IQR)"""
        return self.apply_clipping_method(predictions_df, 'iqr_10')
    
    def compare_all_methods(self, predictions_df: pd.DataFrame) -> Dict:
        """모든 방법 비교"""
        
        methods = {
            'historical_max': self.apply_method_1_historical_max,
            'iqr_standard': self.apply_method_2_iqr_standard,
            'p90': self.apply_method_3_p90,
            'p85': self.apply_method_4_p85,
            'p80': self.apply_method_5_p80,
            'iqr_conservative': self.apply_method_6_conservative_iqr,
        }
        
        results = {}
        original_values = predictions_df.select_dtypes(include=[np.number]).values.flatten()
        original_sum = np.sum(original_values)
        original_max = np.max(original_values)
        original_mean = np.mean(original_values)
        
        print("\n클리핑 방법별 비교 결과:")
        print(f"원본 - 총합: {original_sum:.1f}, 최댓값: {original_max:.1f}, 평균: {original_mean:.2f}")
        print("-" * 70)
        
        for method_name, method_func in methods.items():
            print(f"\n{method_name} 처리 중...")
            clipped_df = method_func(predictions_df)
            clipped_values = clipped_df.select_dtypes(include=[np.number]).values.flatten()
            
            clipped_sum = np.sum(clipped_values)
            clipped_max = np.max(clipped_values)
            clipped_mean = np.mean(clipped_values)
            
            reduction_ratio = (original_sum - clipped_sum) / original_sum * 100
            
            print(f"{method_name:15s}: 총합감소 {reduction_ratio:5.1f}%, "
                  f"최댓값 {original_max:.1f}→{clipped_max:.1f}, "
                  f"평균 {original_mean:.2f}→{clipped_mean:.2f}")
            
            # 결과 저장
            output_file = f"ensemble_submission_{method_name}_exclude_{len(self.exclude_venues)}venues.csv"
            clipped_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            
            results[method_name] = {
                'clipped_df': clipped_df,
                'reduction_ratio': reduction_ratio,
                'output_file': output_file
            }
        
        return results
    
    def show_menu_examples(self, top_n: int = 10):
        """메뉴별 클리핑 경계값 예시"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        print(f"\n상위 {top_n}개 메뉴의 클리핑 경계값 예시:")
        print("-" * 80)
        
        # 역사적 최댓값 기준으로 정렬
        sorted_menus = sorted(
            self.menu_bounds.items(),
            key=lambda x: x[1]['historical_max'] if not x[1].get('excluded', False) else 0,
            reverse=True
        )
        
        for i, (menu, bounds) in enumerate(sorted_menus[:top_n]):
            short_menu = menu[:40] + "..." if len(menu) > 40 else menu
            
            if bounds.get('excluded', False):
                print(f"{i+1:2d}. {short_menu} [제외 업장]")
                print(f"     → 클리핑 제외, 최소값만 1로 보정")
            else:
                print(f"{i+1:2d}. {short_menu}")
                print(f"     역사최댓값: {bounds['historical_max']:6.1f} | "
                      f"99%: {bounds['p99']:6.1f} | "
                      f"95%: {bounds['p95']:6.1f} | "
                      f"90%: {bounds['p90']:6.1f} | "
                      f"85%: {bounds['p85']:6.1f}")
                print(f"     IQR(1.5): {bounds['iqr_15']:6.1f} | "
                      f"IQR(1.0): {bounds['iqr_10']:6.1f} | "
                      f"0비율: {bounds['zero_ratio']*100:4.1f}%")
            print()
    
    def show_excluded_venue_stats(self):
        """제외 업장의 통계 정보 출력"""
        if not self.exclude_venues:
            print("제외된 업장이 없습니다.")
            return
        
        train_df = pd.read_csv(self.train_path)
        
        print(f"\n제외 업장 통계:")
        print("-" * 50)
        
        for venue in self.exclude_venues:
            venue_data = train_df[train_df['영업장명_메뉴명'].str.startswith(f"{venue}_")]
            if len(venue_data) > 0:
                venue_menus = venue_data['영업장명_메뉴명'].nunique()
                venue_total_sales = venue_data['매출수량'].sum()
                venue_avg_sales = venue_data['매출수량'].mean()
                venue_max_sales = venue_data['매출수량'].max()
                
                print(f"{venue}:")
                print(f"  메뉴 수: {venue_menus}")
                print(f"  총 매출: {venue_total_sales:,.0f}")
                print(f"  평균 매출: {venue_avg_sales:.1f}")
                print(f"  최대 매출: {venue_max_sales:.1f}")
                print()

def quick_test_all_methods_with_exclusion(
    input_file: str = 'ensemble_submission_2.csv',
    exclude_venues: List[str] = None
):
    """빠른 테스트 - 특정 업장 제외하고 모든 방법 실행"""
    
    if exclude_venues is None:
        exclude_venues = ['담하', '미라시아']  # 기본값: 고가중치 업장
    
    print("업장 제외 클리핑 전략 테스트")
    print("=" * 50)
    print(f"제외할 업장: {exclude_venues}")
    
    clipper = SimpleEffectiveClipping(exclude_venues=exclude_venues)
    predictions_df = pd.read_csv(input_file)
    
    # 메뉴별 경계값 계산
    print("\n메뉴별 클리핑 경계값 계산 중...")
    clipper.calculate_menu_bounds()
    
    # 제외 업장 통계
    clipper.show_excluded_venue_stats()
    
    # 예시 출력
    clipper.show_menu_examples(15)
    
    # 모든 방법 비교
    print("\n모든 클리핑 방법 테스트 중...")
    results = clipper.compare_all_methods(predictions_df)
    
    print(f"\n권장 순서:")
    print(f"1. p90 또는 p85부터 시작 (보수적)")
    print(f"2. 점수 개선되면 p80 시도 (더 보수적)")
    print(f"3. iqr_conservative도 테스트")
    print(f"\n※ {exclude_venues} 업장은 클리핑에서 제외됩니다")
    
    return results, clipper

# 사용 예시
if __name__ == "__main__":
    # 담하, 미라시아 제외하고 테스트
    results, clipper = quick_test_all_methods_with_exclusion(
        input_file='ensemble_submission_2.csv',
        exclude_venues=['담하', '미라시아']
    )
    
    # 또는 다른 업장들 제외
    # results, clipper = quick_test_all_methods_with_exclusion(
    #     input_file='ensemble_submission_2.csv',
    #     exclude_venues=['포레스트릿', '카페테리아']
    # )