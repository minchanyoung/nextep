document.addEventListener('DOMContentLoaded', function () {
    // --- 1. 초기 데이터 및 DOM 요소 --- //
    const jobASelect = document.getElementById('jobACategorySelect');
    const jobBSelect = document.getElementById('jobBCategorySelect');
    const prioritySlider = document.getElementById('prioritySlider');
    const priorityLabel = document.getElementById('priorityLabel');

    // --- 2. 색상 팔레트 --- //
    const COLORS = {
        primary: '#00b894', primaryDark: '#008a6e', secondary: '#5e72e4',
        neutral: '#8898aa', grid: '#e9ecef', text: '#32325d', tooltipBg: '#ffffff',
        scenario1: 'rgba(0, 184, 148, 0.7)', scenario2: 'rgba(94, 114, 228, 0.7)', scenario3: 'rgba(136, 152, 170, 0.7)'
    };

    // --- 3. 차트 인스턴스 --- //
    let incomeChart, satisfactionChart, incomeDistributionChart, satisfactionDistributionChart;

    // --- 4. 헬퍼 함수 --- //
    const formatIncomeChange = (value) => {
        const val = parseFloat(value);
        const icon = val > 0.001 ? "▲" : (val < -0.001 ? "▼" : "―");
        return `${icon} ${(val * 100).toFixed(2)}%`;
    };
    const formatSatisfactionChange = (value) => {
        const val = parseFloat(value);
        const icon = val > 0.001 ? "▲" : (val < -0.001 ? "▼" : "―");
        return `${icon} ${val.toFixed(2)}점`;
    };
    const getChangeClass = (value) => {
        const val = parseFloat(value);
        if (val > 0.001) return 'positive-change';
        if (val < -0.001) return 'negative-change';
        return 'no-change';
    };

    // --- 5. 차트 생성 로직 --- //
    const createScenarioChart = (ctx, chartData, yAxisLabel) => {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: chartData.labels,
                datasets: [{
                    label: yAxisLabel, data: chartData.data,
                    backgroundColor: [COLORS.scenario1, COLORS.scenario2, COLORS.scenario3],
                    borderWidth: 0, borderRadius: 5
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: true, displayColors: false },
                    datalabels: {
                        anchor: 'end', align: 'top', color: COLORS.text,
                        font: { weight: 'bold', size: 13 },
                        formatter: (v) => yAxisLabel.includes('%') ? `${v}%` : `${v}점`
                    }
                },
                scales: { x: { grid: { display: false } }, y: { grid: { color: COLORS.grid }, ticks: { precision: 2 } } }
            },
            plugins: [ChartDataLabels]
        });
    };

    const createDistributionChart = (ctx, distData, label) => {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: distData.bins.slice(0, -1).map((bin, i) => `${bin.toFixed(1)}~${distData.bins[i+1].toFixed(1)}`),
                datasets: [{ label: '사례 수', data: distData.counts, backgroundColor: COLORS.primary, borderRadius: 4 }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { enabled: true }, datalabels: { display: false } },
                scales: { x: { title: { display: true, text: label } }, y: { title: { display: true, text: '유사 사례 수' } } }
            }
        });
    };

    const initCharts = () => {
        const currentData = allPredictionResults['current'];
        const jobAData = allPredictionResults[jobASelect.value];
        const jobBData = allPredictionResults[jobBSelect.value];

        incomeChart = createScenarioChart(document.getElementById('incomeChart').getContext('2d'), {
            labels: ['현직 유지', jobCategoryMapJs[jobASelect.value], jobCategoryMapJs[jobBSelect.value]],
            data: [currentData.income_change_rate * 100, jobAData.income_change_rate * 100, jobBData.income_change_rate * 100].map(v => v.toFixed(1))
        }, '월 소득 변화율 (%)');

        satisfactionChart = createScenarioChart(document.getElementById('satisfactionChart').getContext('2d'), {
            labels: ['현직 유지', jobCategoryMapJs[jobASelect.value], jobCategoryMapJs[jobBSelect.value]],
            data: [currentData.satisfaction_change_score, jobAData.satisfaction_change_score, jobBData.satisfaction_change_score].map(v => v.toFixed(2))
        }, '직무 만족도 변화 (점)');

        const recommendedResult = allPredictionResults['current'];
        if (recommendedResult?.distribution?.income) {
            incomeDistributionChart = createDistributionChart(document.getElementById('incomeDistributionChart').getContext('2d'), recommendedResult.distribution.income, '소득 변화율 (%)');
        }
        if (recommendedResult?.distribution?.satisfaction) {
            satisfactionDistributionChart = createDistributionChart(document.getElementById('satisfactionDistributionChart').getContext('2d'), recommendedResult.distribution.satisfaction, '만족도 변화 (점)');
        }
    };

    // --- 6. UI 업데이트 함수 --- //
    const updateScenarioCard = (scenarioId, jobCode) => {
        const resultData = allPredictionResults[jobCode];
        if (!resultData) return;

        // ID를 사용하여 더 안정적으로 요소를 찾습니다.
        const nameEl = document.getElementById(`${scenarioId}Name`);
        const incomeEl = document.getElementById(`${scenarioId}Income`);
        const satisEl = document.getElementById(`${scenarioId}Satis`);

        if (nameEl) nameEl.textContent = jobCategoryMapJs[jobCode];
        if (incomeEl) {
            incomeEl.textContent = formatIncomeChange(resultData.income_change_rate);
            incomeEl.className = `value ${getChangeClass(resultData.income_change_rate)}`;
        }
        if (satisEl) {
            satisEl.textContent = formatSatisfactionChange(resultData.satisfaction_change_score);
            satisEl.className = `value ${getChangeClass(resultData.satisfaction_change_score)}`;
        }
    };
    
    const updateCharts = () => {
        const jobACode = jobASelect.value;
        const jobBCode = jobBSelect.value;
        const labels = ['현직 유지', jobCategoryMapJs[jobACode], jobCategoryMapJs[jobBCode]];
        const incomeData = [
            allPredictionResults['current'].income_change_rate * 100,
            allPredictionResults[jobACode].income_change_rate * 100,
            allPredictionResults[jobBCode].income_change_rate * 100
        ].map(v => v.toFixed(1));
        const satisData = [
            allPredictionResults['current'].satisfaction_change_score,
            allPredictionResults[jobACode].satisfaction_change_score,
            allPredictionResults[jobBCode].satisfaction_change_score
        ].map(v => v.toFixed(2));

        if (incomeChart) {
            incomeChart.data.labels = labels;
            incomeChart.data.datasets[0].data = incomeData;
            incomeChart.update();
        }
        if (satisfactionChart) {
            satisfactionChart.data.labels = labels;
            satisfactionChart.data.datasets[0].data = satisData;
            satisfactionChart.update();
        }
    };

    const updateRecommendation = () => {
        const priority = prioritySlider.value / 100;
        const scenarios = {
            'current': allPredictionResults['current'],
            [jobASelect.value]: allPredictionResults[jobASelect.value],
            [jobBSelect.value]: allPredictionResults[jobBSelect.value]
        };

        const scores = Object.entries(scenarios).map(([code, data]) => ({
            code,
            income: data.income_change_rate,
            satis: data.satisfaction_change_score
        }));

        const minIncome = Math.min(...scores.map(s => s.income));
        const maxIncome = Math.max(...scores.map(s => s.income));
        const minSatis = Math.min(...scores.map(s => s.satis));
        const maxSatis = Math.max(...scores.map(s => s.satis));

        let bestScenarioCode = 'current';
        let maxScore = -Infinity;

        scores.forEach(s => {
            const normIncome = (maxIncome - minIncome) === 0 ? 0.5 : (s.income - minIncome) / (maxIncome - minIncome);
            const normSatis = (maxSatis - minSatis) === 0 ? 0.5 : (s.satis - minSatis) / (maxSatis - minSatis);
            const finalScore = (normIncome * priority) + (normSatis * (1 - priority));
            if (finalScore > maxScore) {
                maxScore = finalScore;
                bestScenarioCode = s.code;
            }
        });
        
        const scenarioNames = {
            'current': '현직 유지',
            [jobASelect.value]: jobCategoryMapJs[jobASelect.value],
            [jobBSelect.value]: jobCategoryMapJs[jobBSelect.value]
        };
        const recommendedJobName = scenarioNames[bestScenarioCode];

        let reason = '';
        if (priority == 0.5) reason = `<strong>소득과 만족도 간의 균형</strong>을 가장 잘 맞추는 선택지입니다.`;
        else if (priority > 0.5) reason = `<strong>소득(${priority*100}%)</strong>을 더 중요하게 고려했을 때, 최적의 선택입니다.`;
        else reason = `<strong>만족도(${(1-priority)*100}%)</strong>를 더 중요하게 고려했을 때, 최적의 선택입니다.`;

        document.getElementById('recommendedJobName').textContent = recommendedJobName;
        document.getElementById('recommendationReason').innerHTML = reason;

        document.querySelectorAll('.result-card').forEach(card => card.classList.remove('recommended'));
        const cardId = bestScenarioCode === 'current' ? 'current' : (bestScenarioCode === jobASelect.value ? 'jobA' : 'jobB');
        const recommendedCard = document.querySelector(`.result-card[data-scenario-id="${cardId}"]`);
        if(recommendedCard) recommendedCard.classList.add('recommended');

        // 추천 결과를 advice 폼에 전달
        updateRecommendationInputs(bestScenarioCode, Math.round(priority * 100));

        // 유사 사례 분포 데이터 업데이트
        document.getElementById('distributionChartTitle').textContent = recommendedJobName;
        updateSimilarCasesDistribution(bestScenarioCode);
    };

    const updateHiddenInputs = () => {
        const adviceForm = document.getElementById('adviceForm');
        const predictionForm = document.getElementById('predictionForm');

        if (adviceForm) {
            adviceForm.querySelector('input[name="job_A_category"]').value = jobASelect.value;
            adviceForm.querySelector('input[name="job_B_category"]').value = jobBSelect.value;
        }

        // 예측 폼의 hidden 필드도 업데이트하여 서버에 현재 선택값이 전달되도록 함
        if (predictionForm) {
            predictionForm.querySelector('input[name="job_A_category"]').value = jobASelect.value;
            predictionForm.querySelector('input[name="job_B_category"]').value = jobBSelect.value;
        }
    };

    // 추천 결과를 hidden input에 업데이트하는 함수
    const updateRecommendationInputs = (recommendedScenarioCode, priorityValue) => {
        const adviceForm = document.getElementById('adviceForm');
        if (adviceForm) {
            const recommendedInput = adviceForm.querySelector('input[name="recommended_scenario"]');
            const priorityInput = adviceForm.querySelector('input[name="priority_weight"]');

            if (recommendedInput) {
                recommendedInput.value = recommendedScenarioCode;
            }
            if (priorityInput) {
                priorityInput.value = priorityValue;
            }
        }
    };

    // 유사 사례 분포 데이터를 가져와서 차트 업데이트
    const updateSimilarCasesDistribution = async (recommendedScenario) => {
        try {
            const response = await fetch('/api/similar-cases-distribution', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_input: userInputData,
                    recommended_scenario: recommendedScenario
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.success && result.distribution) {
                const distData = result.distribution;

                // 소득 변화율 분포 차트 업데이트
                if (distData.income && incomeDistributionChart) {
                    const incomeLabels = distData.income.bins.slice(0, -1).map((b, i) =>
                        `${(b * 100).toFixed(0)}~${(distData.income.bins[i + 1] * 100).toFixed(0)}%`
                    );
                    incomeDistributionChart.data.labels = incomeLabels;
                    incomeDistributionChart.data.datasets[0].data = distData.income.counts;
                    incomeDistributionChart.update();
                }

                // 직무 만족도 변화 분포 차트 업데이트
                if (distData.satisfaction && satisfactionDistributionChart) {
                    const satisLabels = distData.satisfaction.bins.slice(0, -1).map((b, i) =>
                        `${b.toFixed(1)}~${distData.satisfaction.bins[i + 1].toFixed(1)}점`
                    );
                    satisfactionDistributionChart.data.labels = satisLabels;
                    satisfactionDistributionChart.data.datasets[0].data = distData.satisfaction.counts;
                    satisfactionDistributionChart.update();
                }

                // 차트 설명 업데이트 (샘플 크기 표시)
                const chartDescription = document.querySelector('.chart-description');
                if (chartDescription && distData.sample_size) {
                    const originalText = chartDescription.innerHTML;
                    const newText = originalText.replace(
                        /유사한 조건.*?분포입니다\./,
                        `유사한 조건을 가진 사람들 <strong>${distData.sample_size}명</strong>의 1년 후 결과 분포입니다.`
                    );
                    chartDescription.innerHTML = newText;
                }
            }
        } catch (error) {
            console.error('유사 사례 분포 데이터 가져오기 실패:', error);
            // 실패 시 기본 분포 데이터 사용
            const fallbackDist = allPredictionResults[recommendedScenario]?.distribution;
            if (fallbackDist?.income && incomeDistributionChart) {
                incomeDistributionChart.data.labels = fallbackDist.income.bins.slice(0, -1).map((b, i) =>
                    `${(b * 100).toFixed(0)}~${(fallbackDist.income.bins[i + 1] * 100).toFixed(0)}%`
                );
                incomeDistributionChart.data.datasets[0].data = fallbackDist.income.counts;
                incomeDistributionChart.update();
            }
            if (fallbackDist?.satisfaction && satisfactionDistributionChart) {
                satisfactionDistributionChart.data.labels = fallbackDist.satisfaction.bins.slice(0, -1).map((b, i) =>
                    `${b.toFixed(1)}~${fallbackDist.satisfaction.bins[i + 1].toFixed(1)}점`
                );
                satisfactionDistributionChart.data.datasets[0].data = fallbackDist.satisfaction.counts;
                satisfactionDistributionChart.update();
            }
        }
    };


    // --- 7. 이벤트 리스너 및 초기화 --- //
    jobASelect.addEventListener('change', (e) => {
        updateScenarioCard('jobA', e.target.value);
        updateCharts();
        updateRecommendation();
        updateHiddenInputs();
    });

    jobBSelect.addEventListener('change', (e) => {
        updateScenarioCard('jobB', e.target.value);
        updateCharts();
        updateRecommendation();
        updateHiddenInputs();
    });

    prioritySlider.addEventListener('input', () => {
        const p = prioritySlider.value;
        priorityLabel.textContent = `균형 (${100 - p}:${p})`;
        updateRecommendation();
    });

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
    if (tabButtons.length > 0) tabButtons[0].click();

    // --- 초기화 실행 --- //
    initCharts();
    updateRecommendation();
});
