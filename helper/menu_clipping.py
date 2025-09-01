import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class SystematicClippingStrategy:
    """통계적 근거에 기반한 체계적 클리핑 전략"""
    
    def __init__(self, train_csv_path: str = './train.csv'):
        self.train_path = train_csv_path
        self.menu_statistics = {}
        self.clipping_thresholds = {}
        
    def analyze_menu_distributions(self):
        """메뉴별 분포 특성 상세 분석"""
        train_df = pd.read_csv(self.train_path)
        
        menu_data = {}
        for _, row in train_df.iterrows():
            menu = row['영업장명_메뉴명']
            sales = row['매출수량']
            
            if pd.notna(sales) and menu:
                if menu not in menu_data:
                    menu_data[menu] = []
                menu_data[menu].append(sales)
        
        for menu, sales_list in menu_data.items():
            if len(sales_list) >= 30:  # 충분한 데이터가 있는 메뉴만
                sales_array = np.array(sales_list)
                non_zero_sales = sales_array[sales_array > 0]
                
                # 기본 통계
                stats_dict = {
                    'count': len(sales_array),
                    'zero_count': np.sum(sales_array == 0),
                    'zero_ratio': np.mean(sales_array == 0),
                    'mean': np.mean(sales_array),
                    'std': np.std(sales_array),
                    'cv': np.std(sales_array) / (np.mean(sales_array) + 1e-8),  # 변동계수
                    'skewness': stats.skew(sales_array),  # 왜도
                    'kurtosis': stats.kurtosis(sales_array),  # 첨도
                }
                
                # 백분위수들
                percentiles = [50, 75, 90, 95, 99, 99.5]
                for p in percentiles:
                    stats_dict[f'p{p}'] = np.percentile(sales_array, p)
                
                # 0이 아닌 값들의 통계 (더 의미있는 분석)
                if len(non_zero_sales) > 0:
                    stats_dict.update({
                        'non_zero_mean': np.mean(non_zero_sales),
                        'non_zero_std': np.std(non_zero_sales),
                        'non_zero_cv': np.std(non_zero_sales) / (np.mean(non_zero_sales) + 1e-8),
                        'non_zero_p95': np.percentile(non_zero_sales, 95),
                        'non_zero_p99': np.percentile(non_zero_sales, 99),
                    })
                
                # IQR 기반 이상치 탐지
                q1 = np.percentile(sales_array, 25)
                q3 = np.percentile(sales_array, 75)
                iqr = q3 - q1
                stats_dict.update({
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'iqr_upper': q3 + 1.5 * iqr,  # 전통적 이상치 경계
                    'iqr_upper_strict': q3 + 1.0 * iqr,  # 더 엄격한 경계
                })
                
                # 분포 형태 분석
                if stats_dict['skewness'] > 2:
                    distribution_type = 'highly_right_skewed'  # 매우 오른쪽 치우침
                elif stats_dict['skewness'] > 1:
                    distribution_type = 'right_skewed'
                elif stats_dict['skewness'] < -1:
                    distribution_type = 'left_skewed'
                else:
                    distribution_type = 'symmetric'
                
                stats_dict['distribution_type'] = distribution_type
                
                self.menu_statistics[menu] = stats_dict
        
        return self.menu_statistics
    
    def determine_statistical_thresholds(self):
        """통계적 근거에 기반한 클리핑 임계값 결정"""
        
        if not self.menu_statistics:
            self.analyze_menu_distributions()
        
        for menu, stats in self.menu_statistics.items():
            
            # 방법 1: 분포 특성 기반 임계값
            if stats['distribution_type'] == 'highly_right_skewed':
                # 매우 치우친 분포: 보수적 접근
                threshold_percentile = 90
            elif stats['distribution_type'] == 'right_skewed':
                # 일반적인 치우침: 중간 접근
                threshold_percentile = 95
            else:
                # 대칭 분포: 관대한 접근
                threshold_percentile = 99
            
            # 방법 2: 변동계수 기반 조정
            cv = stats['cv']
            if cv > 3:  # 매우 불안정
                cv_adjustment = -5  # 더 보수적
            elif cv > 2:  # 불안정
                cv_adjustment = -2
            elif cv < 0.5:  # 매우 안정적
                cv_adjustment = +3  # 더 관대
            else:
                cv_adjustment = 0
            
            final_percentile = max(85, min(99, threshold_percentile + cv_adjustment))
            
            # 방법 3: IQR 기반 검증
            iqr_threshold = stats['iqr_upper']
            percentile_threshold = stats[f'p{final_percentile}']
            
            # 두 방법 중 더 보수적인 값 선택
            conservative_threshold = min(iqr_threshold, percentile_threshold)
            
            # 방법 4: 0값 비율 고려한 최종 조정
            if stats['zero_ratio'] > 0.8:
                # 거의 팔리지 않는 메뉴: 매우 보수적
                final_threshold = stats['q3']  # Q3로 제한
            elif stats['zero_ratio'] > 0.5:
                # 가끔 팔리는 메뉴: 보수적
                final_threshold = min(conservative_threshold, stats['p90'])
            else:
                # 자주 팔리는 메뉴: 상대적으로 관대
                final_threshold = conservative_threshold
            
            # 최소값 보장 (너무 작으면 의미 없음)
            final_threshold = max(final_threshold, 1)
            
            self.clipping_thresholds[menu] = {
                'threshold': final_threshold,
                'method_used': f'percentile_{final_percentile}_with_iqr_validation',
                'distribution_type': stats['distribution_type'],
                'cv': cv,
                'zero_ratio': stats['zero_ratio'],
                'original_max': stats['p99.5'],
                'reasoning': self._generate_reasoning(stats, final_threshold, final_percentile)
            }
        
        return self.clipping_thresholds
    
    def _generate_reasoning(self, stats: Dict, threshold: float, percentile: int) -> str:
        """클리핑 결정의 논리적 근거 생성"""
        
        reasons = []
        
        # 분포 형태 근거
        if stats['distribution_type'] == 'highly_right_skewed':
            reasons.append(f"매우 치우친 분포(왜도:{stats['skewness']:.1f})로 인한 보수적 접근")
        elif stats['distribution_type'] == 'right_skewed':
            reasons.append(f"오른쪽 치우친 분포(왜도:{stats['skewness']:.1f})로 인한 중간 접근")
        
        # 변동성 근거
        if stats['cv'] > 3:
            reasons.append(f"높은 변동성(CV:{stats['cv']:.1f})으로 인한 보수적 조정")
        elif stats['cv'] < 0.5:
            reasons.append(f"낮은 변동성(CV:{stats['cv']:.1f})으로 인한 관대한 조정")
        
        # 0값 비율 근거
        if stats['zero_ratio'] > 0.8:
            reasons.append(f"높은 0값 비율({stats['zero_ratio']*100:.1f}%)로 인한 Q3 제한")
        elif stats['zero_ratio'] > 0.5:
            reasons.append(f"중간 0값 비율({stats['zero_ratio']*100:.1f}%)로 인한 보수적 접근")
        
        # IQR 검증 근거
        iqr_upper = stats['iqr_upper']
        if threshold <= iqr_upper:
            reasons.append(f"IQR 이상치 기준({iqr_upper:.1f}) 준수")
        
        return " | ".join(reasons) if reasons else f"{percentile}% 백분위수 적용"
    
    def apply_systematic_clipping(self, predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """체계적 클리핑 적용"""
        
        if not self.clipping_thresholds:
            self.determine_statistical_thresholds()
        
        clipped_df = predictions_df.copy()
        menu_columns = [col for col in clipped_df.columns if col != '영업일자']
        
        clipping_log = []
        
        for col in menu_columns:
            if col in self.clipping_thresholds:
                threshold_info = self.clipping_thresholds[col]
                threshold = threshold_info['threshold']
                
                original_values = clipped_df[col].values
                clipped_values = np.clip(original_values, 1, threshold)
                clipped_df[col] = clipped_values
                
                # 클리핑 로그
                clipped_count = np.sum(original_values != clipped_values)
                if clipped_count > 0:
                    clipping_log.append({
                        'menu': col,
                        'threshold': threshold,
                        'original_max': np.max(original_values),
                        'clipped_count': clipped_count,
                        'method': threshold_info['method_used'],
                        'reasoning': threshold_info['reasoning']
                    })
            else:
                # 통계가 없는 메뉴는 기본 클리핑
                clipped_df[col] = np.maximum(clipped_df[col].values, 1)
        
        summary = {
            'total_menus_analyzed': len(self.clipping_thresholds),
            'clipped_menus': len(clipping_log),
            'clipping_details': clipping_log
        }
        
        return clipped_df, summary
    
    def explain_clipping_decisions(self, top_n: int = 10):
        """클리핑 결정에 대한 상세 설명"""
        
        if not self.clipping_thresholds:
            self.determine_statistical_thresholds()
        
        print("=== 클리핑 결정의 통계적 근거 ===\n")
        
        # 임계값 기준으로 정렬 (낮은 순)
        sorted_thresholds = sorted(
            self.clipping_thresholds.items(), 
            key=lambda x: x[1]['threshold']
        )
        
        for i, (menu, info) in enumerate(sorted_thresholds[:top_n]):
            print(f"{i+1}. {menu[:50]}{'...' if len(menu) > 50 else ''}")
            print(f"   클리핑 임계값: {info['threshold']:.1f}")
            print(f"   원본 최댓값: {info['original_max']:.1f}")
            print(f"   적용 방법: {info['method_used']}")
            print(f"   근거: {info['reasoning']}")
            print(f"   분포 타입: {info['distribution_type']}")
            print(f"   변동계수: {info['cv']:.2f}")
            print(f"   0값 비율: {info['zero_ratio']*100:.1f}%")
            print()

def run_systematic_clipping(input_file: str, output_file: str):
    """체계적 클리핑 실행"""
    
    print("통계적 근거 기반 체계적 클리핑 시작...")
    
    clipper = SystematicClippingStrategy()
    
    # 분포 분석
    print("1. 메뉴별 분포 분석...")
    clipper.analyze_menu_distributions()
    
    # 임계값 결정
    print("2. 통계적 임계값 결정...")
    clipper.determine_statistical_thresholds()
    
    # 결정 과정 설명
    print("3. 클리핑 결정 근거:")
    clipper.explain_clipping_decisions(15)
    
    # 클리핑 적용
    print("4. 클리핑 적용...")
    predictions_df = pd.read_csv(input_file)
    clipped_df, summary = clipper.apply_systematic_clipping(predictions_df)
    
    # 결과 저장
    clipped_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"5. 결과 저장: {output_file}")
    print(f"   분석된 메뉴: {summary['total_menus_analyzed']}개")
    print(f"   클리핑된 메뉴: {summary['clipped_menus']}개")
    
    return clipped_df, summary, clipper

if __name__ == "__main__":
    # 체계적 클리핑 실행
    clipped_df, summary, clipper = run_systematic_clipping(
        'ensemble_submission_1.csv',
        'ensemble_submission_systematic.csv'
    )