document.addEventListener('DOMContentLoaded', function () {
    // --- 1. 초기 데이터 및 DOM 요소 --- //
    let predictionResults = predictionResultsRaw;
    const jobASelect = document.getElementById('jobACategorySelect');
    const jobBSelect = document.getElementById('jobBCategorySelect');
    const prioritySlider = document.getElementById('prioritySlider');
    const priorityLabel = document.getElementById('priorityLabel');
    const form = document.getElementById('predictionForm');

    // --- 2. 트렌디 색상 팔레트 정의 --- //
    const COLORS = {
        primary: '#00b894',
        primaryDark: '#008a6e',
        secondary: '#5e72e4',
        neutral: '#8898aa',
        grid: '#e9ecef',
        text: '#32325d',
        tooltipBg: '#ffffff',
        // 시나리오별 색상
        scenario1: 'rgba(0, 184, 148, 0.7)',  // Primary
        scenario2: 'rgba(94, 114, 228, 0.7)', // Secondary
        scenario3: 'rgba(136, 152, 170, 0.7)' // Neutral
    };

    // --- 3. 차트 인스턴스 --- //
    let incomeChart, satisfactionChart, incomeDistributionChart, satisfactionDistributionChart;

    // --- 4. 차트 생성 및 업데이트 로직 --- //

    /**
     * 시나리오 비교 차트(소득/만족도) 생성 함수
     */
    const createScenarioChart = (ctx, chartData, yAxisLabel) => {
        const data = {
            labels: chartData.labels,
            datasets: [{
                label: yAxisLabel,
                data: chartData.data,
                backgroundColor: [COLORS.scenario1, COLORS.scenario2, COLORS.scenario3],
                borderColor: [COLORS.scenario1, COLORS.scenario2, COLORS.scenario3],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: 'start',
                hoverBorderColor: [COLORS.primaryDark, COLORS.secondary, COLORS.neutral],
                hoverBorderWidth: 4,
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    enabled: true,
                    backgroundColor: COLORS.tooltipBg,
                    titleColor: COLORS.text,
                    bodyColor: COLORS.text,
                    borderColor: COLORS.grid,
                    borderWidth: 1,
                    padding: 10,
                    displayColors: false,
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.formattedValue}`
                    }
                },
                datalabels: {
                    anchor: 'end',
                    align: 'top',
                    color: COLORS.text,
                    font: { weight: 'bold', size: 13 },
                    formatter: (value) => yAxisLabel.includes('%') ? `${value}%` : `${value}점`
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { weight: '500' } }
                },
                y: {
                    grid: { color: COLORS.grid },
                    ticks: { precision: 2 },
                    title: { display: true, text: yAxisLabel, font: { weight: '500' } }
                }
            }
        };

        return new Chart(ctx, { type: 'bar', data, options, plugins: [ChartDataLabels] });
    };

    /**
     * 분포 차트(히스토그램) 생성 함수
     */
    const createDistributionChart = (ctx, distData, label) => {
        const data = {
            labels: distData.bins.slice(0, -1).map((bin, i) => `${bin.toFixed(1)}~${distData.bins[i+1].toFixed(1)}`),
            datasets: [{
                label: '사례 수',
                data: distData.counts,
                backgroundColor: COLORS.primary,
                borderRadius: 4,
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: true },
                datalabels: { display: false }
            },
            scales: {
                x: { title: { display: true, text: label } },
                y: { title: { display: true, text: '유사 사례 수' } }
            }
        };

        return new Chart(ctx, { type: 'bar', data, options });
    };

    /**
     * 모든 차트를 초기화하는 함수
     */
    const initCharts = () => {
        const initialLabels = [
            '현직 유지',
            jobCategoryMapJs[selectedJobACategory],
            jobCategoryMapJs[selectedJobBCategory]
        ].slice(0, predictionResults.length);

        // 소득 비교 차트
        incomeChart = createScenarioChart(document.getElementById('incomeChart').getContext('2d'), {
            labels: initialLabels,
            data: predictionResults.map(r => (r.income_change_rate * 100).toFixed(1))
        }, '월 소득 변화율 (%)');

        // 만족도 비교 차트
        satisfactionChart = createScenarioChart(document.getElementById('satisfactionChart').getContext('2d'), {
            labels: initialLabels,
            data: predictionResults.map(r => r.satisfaction_change_score.toFixed(2))
        }, '직무 만족도 변화 (점)');

        // 분포 차트 (첫 번째 추천 결과 기준)
        const recommendedResult = predictionResults[0]; // 초기값
        incomeDistributionChart = createDistributionChart(document.getElementById('incomeDistributionChart').getContext('2d'), recommendedResult.distribution.income, '소득 변화율 (%)');
        satisfactionDistributionChart = createDistributionChart(document.getElementById('satisfactionDistributionChart').getContext('2d'), recommendedResult.distribution.satisfaction, '만족도 변화 (점)');
    };

    /**
     * AI 추천 내용을 업데이트하는 함수 (로직 개선)
     */
    const updateRecommendation = () => {
        const priority = prioritySlider.value; // 0-100
        const weightIncome = priority / 100;
        const weightSatis = 1 - weightIncome;

        const incomeRates = predictionResults.map(r => r.income_change_rate);
        const satisScores = predictionResults.map(r => r.satisfaction_change_score);
        const minIncome = Math.min(...incomeRates), maxIncome = Math.max(...incomeRates);
        const minSatis = Math.min(...satisScores), maxSatis = Math.max(...satisScores);

        const normalizedScores = predictionResults.map(r => {
            const normIncome = (maxIncome - minIncome) === 0 ? 0.5 : (r.income_change_rate - minIncome) / (maxIncome - minIncome);
            const normSatis = (maxSatis - minSatis) === 0 ? 0.5 : (r.satisfaction_change_score - minSatis) / (maxSatis - minSatis);
            return (normIncome * weightIncome) + (normSatis * weightSatis);
        });

        const bestIndex = normalizedScores.indexOf(Math.max(...normalizedScores));
        const scenarioNames = ['현직 유지', jobCategoryMapJs[jobASelect.value], jobCategoryMapJs[jobBSelect.value]];
        const recommendedJobName = scenarioNames[bestIndex];

        // 추천 사유 동적 생성
        let reason = '';
        if (priority == 50) {
            reason = `<strong>소득과 만족도 간의 균형</strong>을 가장 잘 맞추는 선택지입니다.`;
        } else if (priority > 50) {
            reason = `<strong>소득(${priority}%)</strong>을 더 중요하게 고려했을 때, 소득 상승이 가장 기대되는 최적의 선택입니다.`;
        } else {
            reason = `<strong>만족도(${100 - priority}%)</strong>를 더 중요하게 고려했을 때, 가장 긍정적인 만족도 변화가 예상되는 선택입니다.`;
        }

        document.getElementById('recommendedJobName').textContent = recommendedJobName;
        document.getElementById('recommendationReason').innerHTML = reason;

        // 추천 뱃지 업데이트
        document.querySelectorAll('.result-card').forEach((card, i) => {
            card.classList.toggle('recommended', i === bestIndex);
        });

        // 분포 차트 제목 및 데이터 업데이트
        document.getElementById('distributionChartTitle').textContent = recommendedJobName;
        const recommendedDist = predictionResults[bestIndex].distribution;
        incomeDistributionChart.data.labels = recommendedDist.income.bins.slice(0, -1).map((b, i) => `${(b*100).toFixed(0)}~${(recommendedDist.income.bins[i+1]*100).toFixed(0)}%`);
        incomeDistributionChart.data.datasets[0].data = recommendedDist.income.counts;
        incomeDistributionChart.update();

        satisfactionDistributionChart.data.labels = recommendedDist.satisfaction.bins.slice(0, -1).map((b, i) => `${b.toFixed(1)}~${recommendedDist.satisfaction.bins[i+1].toFixed(1)}점`);
        satisfactionDistributionChart.data.datasets[0].data = recommendedDist.satisfaction.counts;
        satisfactionDistributionChart.update();
    };

    // --- 5. 이벤트 리스너 및 초기화 --- //

    /**
     * 드롭다운 변경 시 서버에 새로운 예측 요청을 보내는 함수
     */
    const updatePredictionResults = async () => {
        try {
            // 로딩 상태 표시
            showLoadingState(true);
            
            // 폼 데이터 수집
            const formData = new FormData(form);
            
            // AJAX 요청
            const response = await fetch(window.location.href, {
                method: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                },
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('서버 응답 오류');
            }
            
            const data = await response.json();
            predictionResults = data.prediction_results;
            
            // UI 업데이트
            updateResultCards();
            updateCharts();
            updateRecommendation();
            
        } catch (error) {
            console.error('예측 결과 업데이트 실패:', error);
            showErrorMessage('예측 결과를 업데이트하는 중 오류가 발생했습니다. 다시 시도해주세요.');
        } finally {
            // 로딩 상태 해제
            showLoadingState(false);
        }
    };

    /**
     * 로딩 상태를 표시/숨기는 함수
     */
    const showLoadingState = (isLoading) => {
        const resultCards = document.querySelectorAll('.result-card');
        const charts = document.querySelectorAll('.chart-container');
        
        if (isLoading) {
            // 결과 카드에 로딩 오버레이 추가
            resultCards.forEach(card => {
                if (!card.querySelector('.loading-overlay')) {
                    const overlay = document.createElement('div');
                    overlay.className = 'loading-overlay';
                    overlay.innerHTML = '<div class="loading-spinner"></div>';
                    card.style.position = 'relative';
                    card.appendChild(overlay);
                }
            });
            
            // 차트에 로딩 상태 추가
            charts.forEach(chart => {
                chart.style.opacity = '0.5';
            });
        } else {
            // 로딩 오버레이 제거
            document.querySelectorAll('.loading-overlay').forEach(overlay => {
                overlay.remove();
            });
            
            // 차트 투명도 복원
            charts.forEach(chart => {
                chart.style.opacity = '1';
            });
        }
    };

    /**
     * 에러 메시지를 표시하는 함수
     */
    const showErrorMessage = (message) => {
        // 기존 에러 메시지 제거
        const existingError = document.querySelector('.error-message-dynamic');
        if (existingError) {
            existingError.remove();
        }
        
        // 새 에러 메시지 생성
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message-dynamic';
        errorDiv.style.cssText = `
            background: #f8d7da;
            color: #721c24;
            padding: 12px 16px;
            border-radius: 8px;
            margin: 16px 0;
            border: 1px solid #f5c6cb;
            text-align: center;
        `;
        errorDiv.textContent = message;
        
        // 결과 카드 컨테이너 위에 삽입
        const container = document.querySelector('.result-cards-container');
        container.parentNode.insertBefore(errorDiv, container);
        
        // 3초 후 자동으로 제거
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 3000);
    };

    /**
     * 결과 카드 내용을 업데이트하는 함수
     */
    const updateResultCards = () => {
        const currentResult = predictionResults[0];
        const jobAResult = predictionResults[1];
        const jobBResult = predictionResults.length > 2 ? predictionResults[2] : null;
        
        // 직업군 A 카드 업데이트
        const jobAName = document.getElementById('jobAName');
        const jobAIncome = document.getElementById('jobAIncome');
        const jobASatis = document.getElementById('jobASatis');
        
        if (jobAName && jobAResult) {
            jobAName.textContent = jobCategoryMapJs[jobASelect.value];
            jobAIncome.textContent = formatIncomeChange(jobAResult.income_change_rate);
            jobAIncome.className = `value ${getChangeClass(jobAResult.income_change_rate)}`;
            jobASatis.textContent = formatSatisfactionChange(jobAResult.satisfaction_change_score);
            jobASatis.className = `value ${getChangeClass(jobAResult.satisfaction_change_score)}`;
        }
        
        // 직업군 B 카드 업데이트
        const jobBCard = document.querySelector('.result-card[data-scenario-id="jobB"]');
        const jobBName = document.getElementById('jobBName');
        const jobBIncome = document.getElementById('jobBIncome');
        const jobBSatis = document.getElementById('jobBSatis');
        
        if (jobBResult && jobBCard) {
            jobBCard.style.display = 'block';
            if (jobBName) jobBName.textContent = jobCategoryMapJs[jobBSelect.value];
            if (jobBIncome) {
                jobBIncome.textContent = formatIncomeChange(jobBResult.income_change_rate);
                jobBIncome.className = `value ${getChangeClass(jobBResult.income_change_rate)}`;
            }
            if (jobBSatis) {
                jobBSatis.textContent = formatSatisfactionChange(jobBResult.satisfaction_change_score);
                jobBSatis.className = `value ${getChangeClass(jobBResult.satisfaction_change_score)}`;
            }
        } else if (jobBCard) {
            jobBCard.style.display = 'none';
        }
    };

    /**
     * 차트 데이터를 업데이트하는 함수
     */
    const updateCharts = () => {
        const labels = [
            '현직 유지',
            jobCategoryMapJs[jobASelect.value],
            jobCategoryMapJs[jobBSelect.value]
        ].slice(0, predictionResults.length);
        
        // 소득 차트 업데이트
        incomeChart.data.labels = labels;
        incomeChart.data.datasets[0].data = predictionResults.map(r => (r.income_change_rate * 100).toFixed(1));
        incomeChart.update();
        
        // 만족도 차트 업데이트
        satisfactionChart.data.labels = labels;
        satisfactionChart.data.datasets[0].data = predictionResults.map(r => r.satisfaction_change_score.toFixed(2));
        satisfactionChart.update();
    };

    // 헬퍼 함수들 (백엔드와 일치)
    const formatIncomeChange = (value) => {
        value = parseFloat(value);
        const icon = value > 0.001 ? "▲" : (value < -0.001 ? "▼" : "―");
        return `${icon} ${(value * 100).toFixed(2)}%`;
    };

    const formatSatisfactionChange = (value) => {
        value = parseFloat(value);
        const icon = value > 0.001 ? "▲" : (value < -0.001 ? "▼" : "―");
        return `${icon} ${value.toFixed(2)}점`;
    };

    const getChangeClass = (value) => {
        value = parseFloat(value);
        if (value > 0.001) return 'positive-change';
        if (value < -0.001) return 'negative-change';
        return 'no-change';
    };

    // 드롭다운 변경 이벤트 리스너
    jobASelect.addEventListener('change', updatePredictionResults);
    jobBSelect.addEventListener('change', updatePredictionResults);

    // 슬라이더 이벤트
    prioritySlider.addEventListener('input', () => {
        priorityLabel.textContent = `균형 (${100 - prioritySlider.value}:${prioritySlider.value})`;
        updateRecommendation();
    });

    // 탭 전환 로직
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            tabContents.forEach(content => content.classList.toggle('active', content.id === tabId));
        });
    });
    // 첫 탭 활성화
    if (tabButtons.length > 0) {
        tabButtons[0].click();
    }

    // 초기화 실행
    initCharts();
    updateRecommendation();
});
