# =========================
# 앙상블 모델 성능 모니터링 시스템
# - 개별 모델 성능 분석
# - 앙상블 기여도 분석
# - 시각화 및 리포트 생성
# - 실시간 성능 추적
# =========================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from korean_lunar_calendar import KoreanLunarCalendar

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class EnsembleMonitor:
    """앙상블 모델 성능 모니터링 클래스"""
    
    def __init__(self, output_dir="./monitoring_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
        
        self.performance_log = []
        self.ensemble_weights_log = []
        self.prediction_log = []
        
    def analyze_ensemble_performance(self, trained_ensembles, validation_data=None):
        """앙상블 성능 종합 분석"""
        print("🔍 앙상블 성능 분석 시작...")
        
        analysis_results = {}
        
        for key, pack in trained_ensembles.items():
            print(f"\n📊 분석 중: {key}")
            
            ensemble = pack['ensemble']
            store_name = pack['store_name']
            
            # 개별 모델 분석
            model_performance = self._analyze_individual_models(ensemble, key)
            
            # 앙상블 가중치 분석
            weight_analysis = self._analyze_ensemble_weights(ensemble, key)
            
            # 예측 다양성 분석
            diversity_analysis = self._analyze_prediction_diversity(ensemble, key)
            
            analysis_results[key] = {
                'store_name': store_name,
                'model_performance': model_performance,
                'weight_analysis': weight_analysis,
                'diversity_analysis': diversity_analysis,
                'ensemble_info': {
                    'ensemble_size': len(ensemble.models),
                    'use_stacking': ensemble.use_stacking,
                    'calibration_factor': pack['calib_a']
                }
            }
        
        # 전체 리포트 생성
        self._generate_comprehensive_report(analysis_results)
        
        # 시각화
        self._create_performance_visualizations(analysis_results)
        
        return analysis_results
    
    def _analyze_individual_models(self, ensemble, menu_key):
        """개별 모델 성능 분석"""
        model_info = []
        
        for i, (model, config) in enumerate(zip(ensemble.models, ensemble.model_configs)):
            model_data = {
                'model_id': i,
                'architecture': config['arch'],
                'learning_rate': config['learning_rate'],
                'weight_decay': config['weight_decay'],
                'ensemble_weight': ensemble.ensemble_weights[i] if ensemble.ensemble_weights is not None else 1.0/len(ensemble.models),
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            }
            model_info.append(model_data)
        
        return pd.DataFrame(model_info)
    
    def _analyze_ensemble_weights(self, ensemble, menu_key):
        """앙상블 가중치 분석"""
        if ensemble.ensemble_weights is None:
            return {"message": "Uniform weights used"}
        
        weights = ensemble.ensemble_weights
        architectures = [config['arch'] for config in ensemble.model_configs]
        
        weight_analysis = {
            'weights': weights,
            'architectures': architectures,
            'dominant_model': architectures[np.argmax(weights)],
            'weight_entropy': -np.sum(weights * np.log(weights + 1e-8)),  # 다양성 측정
            'weight_concentration': np.max(weights),  # 집중도 측정
            'effective_models': 1 / np.sum(weights ** 2)  # 유효 모델 수
        }
        
        return weight_analysis
    
    def _analyze_prediction_diversity(self, ensemble, menu_key):
        """예측 다양성 분석 (가상 데이터로 테스트)"""
        # 실제 구현에서는 검증 데이터를 사용해야 함
        diversity_metrics = {
            'prediction_variance': 0.0,  # 모델간 예측 분산
            'correlation_matrix': None,   # 모델간 예측 상관관계
            'disagreement_rate': 0.0     # 모델간 불일치 비율
        }
        
        return diversity_metrics
    
    def monitor_prediction_quality(self, predictions, actual_values, menu_key, date_range):
        """예측 품질 실시간 모니터링"""
        
        # 성능 지표 계산
        mse = mean_squared_error(actual_values, predictions)
        mae = mean_absolute_error(actual_values, predictions)
        smape = self._calculate_smape(actual_values, predictions)
        
        # 예측 품질 메트릭
        quality_metrics = {
            'menu_key': menu_key,
            'date_range': date_range,
            'mse': mse,
            'mae': mae,
            'smape': smape,
            'rmse': np.sqrt(mse),
            'mape': np.mean(np.abs((actual_values - predictions) / (actual_values + 1e-8))) * 100,
            'prediction_bias': np.mean(predictions - actual_values),
            'prediction_std': np.std(predictions),
            'actual_std': np.std(actual_values)
        }
        
        # 로그에 추가
        self.prediction_log.append(quality_metrics)
        
        # 이상 징후 감지
        anomalies = self._detect_prediction_anomalies(quality_metrics)
        
        return quality_metrics, anomalies
    
    def _calculate_smape(self, actual, predicted):
        """SMAPE 계산"""
        mask = (np.abs(actual) > 0)
        if mask.sum() == 0:
            return 0.0
        
        denominator = (np.abs(actual[mask]) + np.abs(predicted[mask]))
        return 200.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / denominator)
    
    def _detect_prediction_anomalies(self, metrics):
        """예측 이상 징후 감지"""
        anomalies = []
        
        # SMAPE 임계값 체크
        if metrics['smape'] > 50:
            anomalies.append({
                'type': 'HIGH_SMAPE',
                'value': metrics['smape'],
                'threshold': 50,
                'severity': 'HIGH'
            })
        
        # 편향 체크
        if abs(metrics['prediction_bias']) > metrics['actual_std'] * 0.5:
            anomalies.append({
                'type': 'HIGH_BIAS',
                'value': metrics['prediction_bias'],
                'threshold': metrics['actual_std'] * 0.5,
                'severity': 'MEDIUM'
            })
        
        # 분산 체크
        variance_ratio = metrics['prediction_std'] / (metrics['actual_std'] + 1e-8)
        if variance_ratio > 2.0 or variance_ratio < 0.5:
            anomalies.append({
                'type': 'VARIANCE_MISMATCH',
                'value': variance_ratio,
                'threshold': '0.5-2.0',
                'severity': 'MEDIUM'
            })
        
        return anomalies
    
    def _generate_comprehensive_report(self, analysis_results):
        """종합 분석 리포트 생성"""
        report_path = f"{self.output_dir}/reports/ensemble_analysis_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🎯 앙상블 모델 성능 분석 리포트\n\n")
            f.write(f"생성 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 전체 요약
            f.write("## 📊 전체 요약\n\n")
            total_menus = len(analysis_results)
            stacking_count = sum(1 for r in analysis_results.values() if r['ensemble_info']['use_stacking'])
            
            f.write(f"- **분석 대상 메뉴**: {total_menus}개\n")
            f.write(f"- **스태킹 사용**: {stacking_count}/{total_menus}개 메뉴\n")
            f.write(f"- **평균 앙상블 크기**: {np.mean([r['ensemble_info']['ensemble_size'] for r in analysis_results.values()]):.1f}개 모델\n\n")
            
            # 업장별 분석
            f.write("## 🏪 업장별 분석\n\n")
            
            store_summary = {}
            for menu_key, result in analysis_results.items():
                store_name = result['store_name']
                if store_name not in store_summary:
                    store_summary[store_name] = []
                store_summary[store_name].append(result)
            
            for store_name, store_results in store_summary.items():
                f.write(f"### {store_name}\n\n")
                f.write(f"- **메뉴 수**: {len(store_results)}개\n")
                
                # 가중치 집중도 분석
                concentrations = []
                dominant_archs = []
                
                for result in store_results:
                    weight_analysis = result['weight_analysis']
                    if isinstance(weight_analysis, dict) and 'weight_concentration' in weight_analysis:
                        concentrations.append(weight_analysis['weight_concentration'])
                        dominant_archs.append(weight_analysis['dominant_model'])
                
                if concentrations:
                    f.write(f"- **평균 가중치 집중도**: {np.mean(concentrations):.3f}\n")
                    f.write(f"- **주력 아키텍처**: {max(set(dominant_archs), key=dominant_archs.count)}\n")
                
                f.write("\n")
            
            # 모델 아키텍처 분석
            f.write("## 🏗️ 모델 아키텍처 분석\n\n")
            
            all_archs = []
            all_weights = []
            
            for result in analysis_results.values():
                weight_analysis = result['weight_analysis']
                if isinstance(weight_analysis, dict) and 'architectures' in weight_analysis:
                    all_archs.extend(weight_analysis['architectures'])
                    all_weights.extend(weight_analysis['weights'])
            
            if all_archs:
                arch_performance = pd.DataFrame({
                    'architecture': all_archs,
                    'weight': all_weights
                })
                
                arch_summary = arch_performance.groupby('architecture')['weight'].agg(['mean', 'count', 'std'])
                arch_summary = arch_summary.sort_values('mean', ascending=False)
                
                f.write("| 아키텍처 | 평균 가중치 | 사용 횟수 | 가중치 표준편차 |\n")
                f.write("|----------|------------|-----------|----------------|\n")
                
                for arch in arch_summary.index:
                    mean_weight = arch_summary.loc[arch, 'mean']
                    count = arch_summary.loc[arch, 'count']
                    std_weight = arch_summary.loc[arch, 'std']
                    f.write(f"| {arch} | {mean_weight:.3f} | {count} | {std_weight:.3f} |\n")
                
                f.write("\n")
            
            # 추천사항
            f.write("## 💡 개선 추천사항\n\n")
            
            # 성능이 낮은 업장 식별
            low_performance_stores = []
            for store_name, store_results in store_summary.items():
                avg_concentration = np.mean([
                    r['weight_analysis'].get('weight_concentration', 0.5) 
                    for r in store_results 
                    if isinstance(r['weight_analysis'], dict)
                ])
                if avg_concentration > 0.8:  # 한 모델에 너무 집중
                    low_performance_stores.append(store_name)
            
            if low_performance_stores:
                f.write("### 앙상블 다양성 개선 필요\n")
                f.write(f"**대상 업장**: {', '.join(low_performance_stores)}\n")
                f.write("- 모델 아키텍처 다양성 증가\n")
                f.write("- 하이퍼파라미터 범위 확대\n")
                f.write("- 특성 엔지니어링 강화\n\n")
            
            f.write("### 전반적 개선 방향\n")
            f.write("1. **스태킹 모델 확대**: 현재 성능이 좋은 메뉴에 우선 적용\n")
            f.write("2. **아키텍처 최적화**: 성능이 좋은 아키텍처 비중 증가\n")
            f.write("3. **업장별 특화**: 중요 업장(담하, 미라시아)에 더 큰 앙상블 적용\n")
            f.write("4. **동적 가중치**: 시간에 따른 모델 성능 변화 반영\n\n")
        
        print(f"📝 리포트 생성 완료: {report_path}")
    
    def _create_performance_visualizations(self, analysis_results):
        """성능 시각화 생성"""
        print("📈 시각화 생성 중...")
        
        # 1. 앙상블 가중치 분포 히트맵
        self._plot_ensemble_weights_heatmap(analysis_results)
        
        # 2. 아키텍처별 성능 비교
        self._plot_architecture_performance(analysis_results)
        
        # 3. 업장별 앙상블 특성 비교
        self._plot_store_ensemble_characteristics(analysis_results)
        
        # 4. 가중치 집중도 vs 다양성 산점도
        self._plot_concentration_vs_diversity(analysis_results)
    
    def _plot_ensemble_weights_heatmap(self, analysis_results):
        """앙상블 가중치 히트맵 생성"""
        # 가중치 데이터 준비
        weight_data = []
        menu_names = []
        arch_names = []
        
        for menu_key, result in analysis_results.items():
            weight_analysis = result['weight_analysis']
            if isinstance(weight_analysis, dict) and 'weights' in weight_analysis:
                weight_data.append(weight_analysis['weights'])
                menu_names.append(menu_key[:20] + '...' if len(menu_key) > 20 else menu_key)
                if not arch_names:  # 첫 번째 메뉴에서 아키텍처 이름 추출
                    arch_names = weight_analysis['architectures']
        
        if weight_data:
            weight_matrix = np.array(weight_data)
            
            plt.figure(figsize=(12, max(8, len(menu_names) * 0.3)))
            sns.heatmap(weight_matrix, 
                       xticklabels=arch_names,
                       yticklabels=menu_names,
                       annot=True, 
                       fmt='.3f',
                       cmap='YlOrRd',
                       cbar_kws={'label': '앙상블 가중치'})
            
            plt.title('메뉴별 앙상블 가중치 분포', fontsize=16, fontweight='bold')
            plt.xlabel('모델 아키텍처', fontsize=12)
            plt.ylabel('메뉴', fontsize=12)
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            plt.savefig(f"{self.output_dir}/plots/ensemble_weights_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_architecture_performance(self, analysis_results):
        """아키텍처별 성능 비교 차트"""
        arch_data = []
        
        for result in analysis_results.values():
            weight_analysis = result['weight_analysis']
            if isinstance(weight_analysis, dict) and 'architectures' in weight_analysis:
                for arch, weight in zip(weight_analysis['architectures'], weight_analysis['weights']):
                    arch_data.append({
                        'architecture': arch,
                        'weight': weight,
                        'store': result['store_name']
                    })
        
        if arch_data:
            df = pd.DataFrame(arch_data)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 아키텍처별 평균 가중치
            arch_mean = df.groupby('architecture')['weight'].mean().sort_values(ascending=True)
            arch_mean.plot(kind='barh', ax=ax1, color='skyblue', edgecolor='navy')
            ax1.set_title('아키텍처별 평균 앙상블 가중치', fontweight='bold')
            ax1.set_xlabel('평균 가중치')
            
            # 아키텍처별 사용 빈도
            arch_count = df['architecture'].value_counts().sort_values(ascending=True)
            arch_count.plot(kind='barh', ax=ax2, color='lightcoral', edgecolor='darkred')
            ax2.set_title('아키텍처별 사용 빈도', fontweight='bold')
            ax2.set_xlabel('사용 횟수')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/architecture_performance.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_store_ensemble_characteristics(self, analysis_results):
        """업장별 앙상블 특성 비교"""
        store_data = []
        
        for result in analysis_results.values():
            weight_analysis = result['weight_analysis']
            if isinstance(weight_analysis, dict):
                store_data.append({
                    'store': result['store_name'],
                    'concentration': weight_analysis.get('weight_concentration', 0.5),
                    'entropy': weight_analysis.get('weight_entropy', 1.0),
                    'effective_models': weight_analysis.get('effective_models', 1.0),
                    'ensemble_size': result['ensemble_info']['ensemble_size'],
                    'use_stacking': result['ensemble_info']['use_stacking']
                })
        
        if store_data:
            df = pd.DataFrame(store_data)
            store_summary = df.groupby('store').agg({
                'concentration': 'mean',
                'entropy': 'mean', 
                'effective_models': 'mean',
                'ensemble_size': 'mean'
            }).round(3)
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # 업장별 가중치 집중도
            store_summary['concentration'].plot(kind='bar', ax=ax1, color='orange', alpha=0.7)
            ax1.set_title('업장별 가중치 집중도\n(높을수록 특정 모델에 의존)', fontweight='bold')
            ax1.set_ylabel('집중도')
            ax1.tick_params(axis='x', rotation=45)
            
            # 업장별 가중치 엔트로피 (다양성)
            store_summary['entropy'].plot(kind='bar', ax=ax2, color='green', alpha=0.7)
            ax2.set_title('업장별 가중치 엔트로피\n(높을수록 다양한 모델 활용)', fontweight='bold')
            ax2.set_ylabel('엔트로피')
            ax2.tick_params(axis='x', rotation=45)
            
            # 업장별 유효 모델 수
            store_summary['effective_models'].plot(kind='bar', ax=ax3, color='purple', alpha=0.7)
            ax3.set_title('업장별 유효 모델 수\n(실질적으로 기여하는 모델 수)', fontweight='bold')
            ax3.set_ylabel('유효 모델 수')
            ax3.tick_params(axis='x', rotation=45)
            
            # 업장별 앙상블 크기
            store_summary['ensemble_size'].plot(kind='bar', ax=ax4, color='red', alpha=0.7)
            ax4.set_title('업장별 평균 앙상블 크기', fontweight='bold')
            ax4.set_ylabel('앙상블 크기')
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/plots/store_ensemble_characteristics.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_concentration_vs_diversity(self, analysis_results):
        """가중치 집중도 vs 다양성 산점도"""
        plot_data = []
        
        for menu_key, result in analysis_results.items():
            weight_analysis = result['weight_analysis']
            if isinstance(weight_analysis, dict):
                plot_data.append({
                    'menu': menu_key,
                    'store': result['store_name'],
                    'concentration': weight_analysis.get('weight_concentration', 0.5),
                    'diversity': weight_analysis.get('effective_models', 1.0),
                    'entropy': weight_analysis.get('weight_entropy', 1.0),
                    'calibration': result['ensemble_info']['calibration_factor']
                })
        
        if plot_data:
            df = pd.DataFrame(plot_data)
            
            plt.figure(figsize=(12, 8))
            
            # 업장별로 다른 색상
            stores = df['store'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(stores)))
            
            for store, color in zip(stores, colors):
                store_data = df[df['store'] == store]
                plt.scatter(store_data['concentration'], 
                           store_data['diversity'],
                           c=[color], 
                           s=store_data['calibration'] * 100,  # 캘리브레이션 팩터에 따른 크기
                           alpha=0.7,
                           label=store,
                           edgecolors='black',
                           linewidth=0.5)
            
            plt.xlabel('가중치 집중도 (높을수록 특정 모델 의존)', fontsize=12)
            plt.ylabel('유효 모델 수 (높을수록 다양한 모델 활용)', fontsize=12)
            plt.title('앙상블 특성 분포: 집중도 vs 다양성\n(원 크기 = 캘리브레이션 팩터)', 
                     fontsize=14, fontweight='bold')
            
            # 이상적인 영역 표시
            plt.axhline(y=3, color='red', linestyle='--', alpha=0.5, label='이상적 다양성 기준')
            plt.axvline(x=0.6, color='red', linestyle='--', alpha=0.5, label='적정 집중도 기준')
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f"{self.output_dir}/plots/concentration_vs_diversity.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_real_time_dashboard(self, analysis_results):
        """실시간 대시보드 데이터 생성 (HTML)"""
        dashboard_path = f"{self.output_dir}/ensemble_dashboard.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>앙상블 모델 모니터링 대시보드</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .metric-box {{ 
                    display: inline-block; 
                    background-color: #e8f4f8; 
                    padding: 15px; 
                    margin: 10px; 
                    border-radius: 8px; 
                    min-width: 200px;
                }}
                .store-section {{ 
                    background-color: #f9f9f9; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 8px; 
                }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .high-performance {{ color: green; font-weight: bold; }}
                .low-performance {{ color: red; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 앙상블 모델 모니터링 대시보드</h1>
                <p>실시간 업데이트: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="metric-box">
                <h3>📊 전체 통계</h3>
                <p><strong>총 메뉴 수:</strong> {len(analysis_results)}</p>
                <p><strong>평균 앙상블 크기:</strong> {np.mean([r['ensemble_info']['ensemble_size'] for r in analysis_results.values()]):.1f}</p>
                <p><strong>스태킹 사용률:</strong> {sum(1 for r in analysis_results.values() if r['ensemble_info']['use_stacking'])}/{len(analysis_results)}</p>
            </div>
        """
        
        # 업장별 섹션 추가
        store_summary = {}
        for menu_key, result in analysis_results.items():
            store_name = result['store_name']
            if store_name not in store_summary:
                store_summary[store_name] = []
            store_summary[store_name].append(result)
        
        for store_name, store_results in store_summary.items():
            html_content += f"""
            <div class="store-section">
                <h2>🏪 {store_name}</h2>
                <table>
                    <tr>
                        <th>메뉴</th>
                        <th>앙상블 크기</th>
                        <th>주력 모델</th>
                        <th>가중치 집중도</th>
                        <th>캘리브레이션</th>
                        <th>상태</th>
                    </tr>
            """
            
            for result in store_results:
                menu_key = list(analysis_results.keys())[list(analysis_results.values()).index(result)]
                ensemble_size = result['ensemble_info']['ensemble_size']
                
                weight_analysis = result['weight_analysis']
                if isinstance(weight_analysis, dict):
                    dominant_model = weight_analysis.get('dominant_model', 'N/A')
                    concentration = weight_analysis.get('weight_concentration', 0.5)
                    concentration_class = 'high-performance' if concentration < 0.7 else 'low-performance'
                else:
                    dominant_model = 'N/A'
                    concentration = 0.5
                    concentration_class = ''
                
                calibration = result['ensemble_info']['calibration_factor']
                calibration_class = 'high-performance' if 0.9 <= calibration <= 1.1 else 'low-performance'
                
                status = '✅ 정상' if concentration < 0.7 and 0.9 <= calibration <= 1.1 else '⚠️ 주의'
                
                html_content += f"""
                    <tr>
                        <td>{menu_key[:30]}{'...' if len(menu_key) > 30 else ''}</td>
                        <td>{ensemble_size}</td>
                        <td>{dominant_model}</td>
                        <td class="{concentration_class}">{concentration:.3f}</td>
                        <td class="{calibration_class}">{calibration:.3f}</td>
                        <td>{status}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
            <div class="metric-box">
                <h3>📈 성능 차트</h3>
                <p><a href="plots/ensemble_weights_heatmap.png" target="_blank">앙상블 가중치 히트맵</a></p>
                <p><a href="plots/architecture_performance.png" target="_blank">아키텍처 성능 비교</a></p>
                <p><a href="plots/store_ensemble_characteristics.png" target="_blank">업장별 특성 비교</a></p>
                <p><a href="plots/concentration_vs_diversity.png" target="_blank">집중도 vs 다양성</a></p>
            </div>
            
            <div class="metric-box">
                <h3>📝 리포트</h3>
                <p><a href="reports/ensemble_analysis_report.md" target="_blank">상세 분석 리포트</a></p>
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 대시보드 생성 완료: {dashboard_path}")
    
    def track_ensemble_evolution(self, trained_ensembles, epoch_data=None):
        """앙상블 진화 추적"""
        timestamp = pd.Timestamp.now()
        
        evolution_data = {
            'timestamp': timestamp,
            'total_ensembles': len(trained_ensembles),
            'avg_ensemble_size': np.mean([
                pack['ensemble'].ensemble_size 
                for pack in trained_ensembles.values()
            ]),
            'stacking_usage': sum(1 for pack in trained_ensembles.values() 
                                 if pack['ensemble'].use_stacking),
            'calibration_stats': {
                'mean': np.mean([pack['calib_a'] for pack in trained_ensembles.values()]),
                'std': np.std([pack['calib_a'] for pack in trained_ensembles.values()]),
                'min': np.min([pack['calib_a'] for pack in trained_ensembles.values()]),
                'max': np.max([pack['calib_a'] for pack in trained_ensembles.values()])
            }
        }
        
        self.performance_log.append(evolution_data)
        
        # 진화 추이 시각화
        if len(self.performance_log) > 1:
            self._plot_evolution_trends()
        
        return evolution_data
    
    def _plot_evolution_trends(self):
        """앙상블 진화 추이 시각화"""
        if len(self.performance_log) < 2:
            return
        
        df = pd.DataFrame(self.performance_log)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 앙상블 크기 변화
        ax1.plot(df['timestamp'], df['avg_ensemble_size'], marker='o', linewidth=2)
        ax1.set_title('평균 앙상블 크기 변화', fontweight='bold')
        ax1.set_ylabel('평균 앙상블 크기')
        ax1.grid(True, alpha=0.3)
        
        # 스태킹 사용률 변화
        stacking_rate = df['stacking_usage'] / df['total_ensembles'] * 100
        ax2.plot(df['timestamp'], stacking_rate, marker='s', color='orange', linewidth=2)
        ax2.set_title('스태킹 사용률 변화', fontweight='bold')
        ax2.set_ylabel('스태킹 사용률 (%)')
        ax2.grid(True, alpha=0.3)
        
        # 캘리브레이션 팩터 평균 변화
        calib_means = [stats['mean'] for stats in df['calibration_stats']]
        ax3.plot(df['timestamp'], calib_means, marker='^', color='green', linewidth=2)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='이상적 값 (1.0)')
        ax3.set_title('평균 캘리브레이션 팩터 변화', fontweight='bold')
        ax3.set_ylabel('평균 캘리브레이션 팩터')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 캘리브레이션 팩터 분산 변화
        calib_stds = [stats['std'] for stats in df['calibration_stats']]
        ax4.plot(df['timestamp'], calib_stds, marker='d', color='purple', linewidth=2)
        ax4.set_title('캘리브레이션 팩터 분산 변화', fontweight='bold')
        ax4.set_ylabel('캘리브레이션 팩터 표준편차')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/ensemble_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_performance_alerts(self, analysis_results):
        """성능 알림 생성"""
        alerts = []
        
        for menu_key, result in analysis_results.items():
            store_name = result['store_name']
            
            # 높은 집중도 알림
            weight_analysis = result['weight_analysis']
            if isinstance(weight_analysis, dict):
                concentration = weight_analysis.get('weight_concentration', 0.5)
                if concentration > 0.8:
                    alerts.append({
                        'type': 'HIGH_CONCENTRATION',
                        'menu': menu_key,
                        'store': store_name,
                        'value': concentration,
                        'message': f"{menu_key}의 앙상블이 특정 모델에 과도하게 의존 (집중도: {concentration:.3f})",
                        'severity': 'WARNING',
                        'recommendation': '모델 아키텍처 다양성 증가 또는 하이퍼파라미터 조정 필요'
                    })
            
            # 캘리브레이션 이상 알림
            calibration = result['ensemble_info']['calibration_factor']
            if calibration < 0.7 or calibration > 1.5:
                severity = 'CRITICAL' if calibration < 0.5 or calibration > 2.0 else 'WARNING'
                alerts.append({
                    'type': 'CALIBRATION_ANOMALY',
                    'menu': menu_key,
                    'store': store_name,
                    'value': calibration,
                    'message': f"{menu_key}의 캘리브레이션 팩터 이상 (값: {calibration:.3f})",
                    'severity': severity,
                    'recommendation': '훈련 데이터 검토 또는 모델 아키텍처 재검토 필요'
                })
            
            # 중요 업장 특별 모니터링
            if store_name in ['담하', '미라시아']:
                effective_models = weight_analysis.get('effective_models', 1.0) if isinstance(weight_analysis, dict) else 1.0
                if effective_models < 2.0:
                    alerts.append({
                        'type': 'CRITICAL_STORE_LOW_DIVERSITY',
                        'menu': menu_key,
                        'store': store_name,
                        'value': effective_models,
                        'message': f"중요 업장 {store_name}의 {menu_key} 앙상블 다양성 부족 (유효 모델: {effective_models:.1f})",
                        'severity': 'CRITICAL',
                        'recommendation': '즉시 앙상블 크기 증가 및 아키텍처 다양화 필요'
                    })
        
        # 알림 저장
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_path = f"{self.output_dir}/performance_alerts.csv"
            alerts_df.to_csv(alerts_path, index=False, encoding='utf-8')
            
            # 심각한 알림 콘솔 출력
            critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
            if critical_alerts:
                print("\n🚨 중요 알림:")
                for alert in critical_alerts:
                    print(f"   {alert['message']}")
                    print(f"   추천: {alert['recommendation']}\n")
        
        return alerts

# -------------------------
# 사용 예시 및 통합 함수
# -------------------------

def run_comprehensive_monitoring(trained_ensembles, validation_data=None):
    """종합 모니터링 실행"""
    print("🎯 앙상블 모델 종합 모니터링 시작...")
    
    # 모니터 객체 생성
    monitor = EnsembleMonitor(output_dir="./monitoring_results")
    
    # 1. 성능 분석
    analysis_results = monitor.analyze_ensemble_performance(trained_ensembles, validation_data)
    
    # 2. 실시간 대시보드 생성
    monitor.create_real_time_dashboard(analysis_results)
    
    # 3. 진화 추적
    evolution_data = monitor.track_ensemble_evolution(trained_ensembles)
    
    # 4. 성능 알림 생성
    alerts = monitor.generate_performance_alerts(analysis_results)
    
    # 5. 요약 리포트
    print("\n📊 모니터링 완료 요약:")
    print(f"   - 분석된 메뉴 수: {len(analysis_results)}")
    print(f"   - 생성된 알림 수: {len(alerts)}")
    print(f"   - 중요 알림 수: {len([a for a in alerts if a['severity'] == 'CRITICAL'])}")
    print(f"   - 결과 저장 위치: ./monitoring_results/")
    
    return {
        'analysis_results': analysis_results,
        'evolution_data': evolution_data,
        'alerts': alerts,
        'monitor': monitor
    }

def setup_continuous_monitoring(trained_ensembles, monitoring_interval_hours=6):
    """지속적 모니터링 설정 (실제 운영환경용)"""
    import schedule
    import time
    
    def monitoring_job():
        try:
            print(f"\n⏰ 정기 모니터링 실행: {pd.Timestamp.now()}")
            run_comprehensive_monitoring(trained_ensembles)
        except Exception as e:
            print(f"❌ 모니터링 중 오류 발생: {e}")
    
    # 정기 모니터링 스케줄 설정
    schedule.every(monitoring_interval_hours).hours.do(monitoring_job)
    
    print(f"🔄 지속적 모니터링 설정 완료 (간격: {monitoring_interval_hours}시간)")
    print("모니터링을 중단하려면 Ctrl+C를 누르세요.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 스케줄 체크
    except KeyboardInterrupt:
        print("\n⏹️ 모니터링이 중단되었습니다.")

# -------------------------
# 사용법 예시
# -------------------------
"""
# 1. 기본 모니터링 (한 번 실행)
monitoring_results = run_comprehensive_monitoring(trained_ensembles)

# 2. 개별 컴포넌트 사용
monitor = EnsembleMonitor()

# 성능 분석만
analysis_results = monitor.analyze_ensemble_performance(trained_ensembles)

# 알림만 생성
alerts = monitor.generate_performance_alerts(analysis_results)

# 대시보드만 생성
monitor.create_real_time_dashboard(analysis_results)

# 3. 예측 품질 실시간 모니터링 (예측 후 실제값과 비교)
actual_values = [10, 15, 20, 25, 30, 35, 40]  # 실제 매출수량
predictions = [12, 14, 22, 24, 32, 33, 38]    # 모델 예측값
quality_metrics, anomalies = monitor.monitor_prediction_quality(
    predictions, actual_values, "담하_메뉴A", "2024-01-01~2024-01-07"
)

if anomalies:
    print("예측 이상 징후 감지:", anomalies)

# 4. 지속적 모니터링 (운영환경)
# setup_continuous_monitoring(trained_ensembles, monitoring_interval_hours=6)
"""

# -------------------------
# 성능 개선 자동 추천 시스템
# -------------------------
class PerformanceOptimizer:
    """성능 분석 결과를 바탕으로 자동 개선 방안 추천"""
    
    def __init__(self):
        self.improvement_rules = {
            'high_concentration': {
                'condition': lambda x: x.get('weight_concentration', 0.5) > 0.8,
                'action': 'increase_diversity',
                'params': {'ensemble_size': '+2', 'architecture_variety': '+1'}
            },
            'low_effective_models': {
                'condition': lambda x: x.get('effective_models', 3.0) < 2.0,
                'action': 'rebalance_weights',
                'params': {'weight_regularization': 0.1, 'ensemble_size': '+1'}
            },
            'calibration_drift': {
                'condition': lambda x: abs(x.get('calibration_factor', 1.0) - 1.0) > 0.3,
                'action': 'retrain_calibration',
                'params': {'validation_split': 0.2, 'calibration_method': 'isotonic'}
            }
        }
    
    def generate_improvement_plan(self, analysis_results):
        """개선 계획 자동 생성"""
        improvement_plan = {}
        
        for menu_key, result in analysis_results.items():
            recommendations = []
            
            # 각 규칙 적용
            for rule_name, rule in self.improvement_rules.items():
                weight_analysis = result.get('weight_analysis', {})
                ensemble_info = result.get('ensemble_info', {})
                
                # 조건 확인
                context = {**weight_analysis, **ensemble_info}
                if rule['condition'](context):
                    recommendations.append({
                        'rule': rule_name,
                        'action': rule['action'],
                        'params': rule['params'],
                        'priority': self._calculate_priority(rule_name, result)
                    })
            
            if recommendations:
                # 우선순위로 정렬
                recommendations.sort(key=lambda x: x['priority'], reverse=True)
                improvement_plan[menu_key] = recommendations
        
        return improvement_plan
    
    def _calculate_priority(self, rule_name, result):
        """개선 우선순위 계산"""
        store_name = result['store_name']
        base_priority = {
            'high_concentration': 3,
            'low_effective_models': 2,
            'calibration_drift': 4
        }.get(rule_name, 1)
        
        # 중요 업장은 우선순위 증가
        if store_name in ['담하', '미라시아']:
            base_priority += 2
        
        return base_priority

# 사용 예시
"""
optimizer = PerformanceOptimizer()
improvement_plan = optimizer.generate_improvement_plan(analysis_results)

for menu_key, recommendations in improvement_plan.items():
    print(f"\n{menu_key} 개선 계획:")
    for rec in recommendations:
        print(f"  - {rec['action']}: {rec['params']} (우선순위: {rec['priority']})")
"""