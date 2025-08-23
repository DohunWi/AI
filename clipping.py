import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class SimpleEffectiveClipping:
    """단순하고 효과적인 클리핑 - 복잡한 통계 없이 실용적 접근"""
    
    def __init__(self, train_csv_path: str = './train.csv'):
        self.train_path = train_csv_path
        self.menu_bounds = {}
    
    def calculate_menu_bounds(self):
        """메뉴별 클리핑 경계값 계산 - 단순한 방법들 제공"""
        
        train_df = pd.read_csv(self.train_path)
        
        for menu in train_df['영업장명_메뉴명'].unique():
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu]['매출수량'].values
            
            if len(menu_data) >= 10:
                # 여러 방법으로 경계값 계산
                bounds = {
                    'historical_max': float(np.max(menu_data)),  # 역사적 최댓값
                    'p99': float(np.percentile(menu_data, 99)),   # 99%
                    'p95': float(np.percentile(menu_data, 95)),   # 95%
                    'p90': float(np.percentile(menu_data, 90)),   # 90%
                    'p85': float(np.percentile(menu_data, 85)),   # 85%
                    'p80': float(np.percentile(menu_data, 80)),   # 80%
                    'p75': float(np.percentile(menu_data, 75)),   # 75%
                }
                
                # IQR 기반 이상치 경계
                q1 = np.percentile(menu_data, 25)
                q3 = np.percentile(menu_data, 75)
                iqr = q3 - q1
                bounds['iqr_15'] = max(1, q3 + 1.5 * iqr)  # 최소 1 보장
                bounds['iqr_10'] = max(1, q3 + 1.0 * iqr)  # 최소 1 보장
                
                # 평균 기반 경계
                mean_val = np.mean(menu_data)
                std_val = np.std(menu_data)
                bounds['mean_plus_2std'] = mean_val + 2 * std_val
                bounds['mean_plus_3std'] = mean_val + 3 * std_val
                
                # 기본 통계
                bounds['zero_ratio'] = float(np.mean(menu_data == 0))
                bounds['mean'] = float(mean_val)
                bounds['count'] = len(menu_data)
                
                self.menu_bounds[menu] = bounds
        
        return self.menu_bounds
    
    def apply_method_1_historical_max(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 1: 역사적 최댓값 기준 (가장 관대)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['historical_max']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
    def apply_method_2_iqr_standard(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 2: 표준 IQR 이상치 기준 (통계적으로 가장 타당)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['iqr_15']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
    def apply_method_3_p90(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 3: 90% 백분위수 기준 (보수적)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['p90']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
    def apply_method_4_p85(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 4: 85% 백분위수 기준 (더 보수적)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['p85']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
    def apply_method_5_p80(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 5: 80% 백분위수 기준 (매우 보수적)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['p80']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
    def apply_method_6_conservative_iqr(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """방법 6: 보수적 IQR 기준 (Q3 + 1.0*IQR)"""
        if not self.menu_bounds:
            self.calculate_menu_bounds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        for col in menu_columns:
            if col in self.menu_bounds:
                upper_bound = self.menu_bounds[col]['iqr_10']
                clipped_df[col] = np.clip(clipped_df[col].values, 1, upper_bound)
            else:
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        return clipped_df
    
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
        
        print("클리핑 방법별 비교 결과:")
        print(f"원본 - 총합: {original_sum:.1f}, 최댓값: {original_max:.1f}, 평균: {original_mean:.2f}")
        print("-" * 70)
        
        for method_name, method_func in methods.items():
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
            output_file = f"ensemble_submission_{method_name}.csv"
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
            key=lambda x: x[1]['historical_max'],
            reverse=True
        )
        
        for i, (menu, bounds) in enumerate(sorted_menus[:top_n]):
            short_menu = menu[:40] + "..." if len(menu) > 40 else menu
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

def quick_test_all_methods(input_file: str = 'ensemble_submission_1.csv'):
    """빠른 테스트 - 모든 방법 한 번에 실행"""
    
    print("간단하고 효과적인 클리핑 전략 테스트")
    print("=" * 50)
    
    clipper = SimpleEffectiveClipping()
    predictions_df = pd.read_csv(input_file)
    
    # 메뉴별 경계값 계산
    print("메뉴별 클리핑 경계값 계산 중...")
    clipper.calculate_menu_bounds()
    
    # 예시 출력
    clipper.show_menu_examples(15)
    
    # 모든 방법 비교
    print("\n모든 클리핑 방법 테스트 중...")
    results = clipper.compare_all_methods(predictions_df)
    
    print(f"\n권장 순서:")
    print(f"1. p90 또는 p85부터 시작 (보수적)")
    print(f"2. 점수 개선되면 p80 시도 (더 보수적)")
    print(f"3. iqr_conservative도 테스트")
    
    return results, clipper

if __name__ == "__main__":
    # 모든 방법 테스트
    results, clipper = quick_test_all_methods()