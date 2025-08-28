const ResultPageManager = {
    init: function () {
        this.cacheDOMElements();
        this.setupInitialData();
        this.createCharts();
        this.createDistributionCharts();
        this.updateRecommendation();
        this.bindEventListeners();
        this.initTabs(); // 탭 초기화 함수 호출
    },

    cacheDOMElements: function () {
        this.elements = {
            prioritySlider: document.getElementById("prioritySlider"),
            priorityLabel: document.getElementById("priorityLabel"),
            recommendedJobName: document.getElementById("recommendedJobName"),
            recommendationReason: document.getElementById("recommendationReason"),
            incomeChartCanvas: document.getElementById('incomeChart'),
            satisfactionChartCanvas: document.getElementById('satisfactionChart'),
            incomeDistributionChartCanvas: document.getElementById('incomeDistributionChart'),
            satisfactionDistributionChartCanvas: document.getElementById('satisfactionDistributionChart'),
            distributionChartTitle: document.getElementById('distributionChartTitle'),
            adviceForm: document.querySelector('.advice-form'),
            // 탭 관련 요소 추가
            tabButtons: document.querySelectorAll('.tab-button'),
            tabContents: document.querySelectorAll('.tab-content')
        };
    },

    initTabs: function() {
        this.elements.tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                // 모든 버튼과 컨텐츠에서 active 클래스 제거
                this.elements.tabButtons.forEach(btn => btn.classList.remove('active'));
                this.elements.tabContents.forEach(content => content.classList.remove('active'));

                // 클릭된 버튼과 해당 컨텐츠에 active 클래스 추가
                button.classList.add('active');
                const activeTab = document.getElementById(button.dataset.tab);
                if (activeTab) {
                    activeTab.classList.add('active');
                }
            });
        });
        // 기본으로 첫 번째 탭을 활성화
        if (this.elements.tabButtons.length > 0) {
            this.elements.tabButtons[0].click();
        }
    },

    setupInitialData: function () {
        this.scenarios = [
            {
                id: "current",
                name: "현직 유지",
                income: predictionResultsRaw[0].income_change_rate,
                satisfaction: predictionResultsRaw[0].satisfaction_change_score,
                distribution: predictionResultsRaw[0].distribution
            },
            {
                id: "jobA",
                name: jobCategoryMapJs[selectedJobACategory] || "직업 A",
                income: predictionResultsRaw[1].income_change_rate,
                satisfaction: predictionResultsRaw[1].satisfaction_change_score,
                distribution: predictionResultsRaw[1].distribution
            }
        ];

        if (predictionResultsRaw.length > 2 && predictionResultsRaw[2]) {
            this.scenarios.push({
                id: "jobB",
                name: jobCategoryMapJs[selectedJobBCategory] || "직업 B",
                income: predictionResultsRaw[2].income_change_rate,
                satisfaction: predictionResultsRaw[2].satisfaction_change_score,
                distribution: predictionResultsRaw[2].distribution
            });
        }

        this.generateUniqueScenarioNames();
    },

    generateUniqueScenarioNames: function () {
        const nameCount = {};
        this.scenarios = this.scenarios.map((s) => {
            const base = s.name;
            nameCount[base] = (nameCount[base] || 0) + 1;
            const newName = nameCount[base] > 1 ? `${base} #${nameCount[base]}` : base;
            return { ...s, name: newName };
        });
    },

    bindEventListeners: function () {
        this.elements.prioritySlider.addEventListener("input", () => this.updateRecommendation());
        if (this.elements.adviceForm) {
            this.elements.adviceForm.addEventListener("submit", (e) => {
                this.prepareAdviceData();
            });
        }
    },

    prepareAdviceData: function () {
        document.getElementById('hiddenRecommendedJobName').value = this.elements.recommendedJobName.textContent;
        document.getElementById('hiddenRecommendationReason').value = this.elements.recommendationReason.textContent;
    },

    updateRecommendation: function () {
        const incomeWeight = this.elements.prioritySlider.value / 100;
        const satisfactionWeight = 1 - incomeWeight;

        this.elements.priorityLabel.textContent = `소득 ${Math.round(incomeWeight * 100)}% : 만족도 ${Math.round(satisfactionWeight * 100)}%`;

        const minIncome = Math.min(...this.scenarios.map(s => s.income));
        const maxIncome = Math.max(...this.scenarios.map(s => s.income));
        const minSatisfaction = Math.min(...this.scenarios.map(s => s.satisfaction));
        const maxSatisfaction = Math.max(...this.scenarios.map(s => s.satisfaction));

        let bestScenario = null;
        let maxScore = -Infinity;

        this.scenarios.forEach(scenario => {
            const normalizedIncome = this.normalize(scenario.income, minIncome, maxIncome);
            const normalizedSatisfaction = this.normalize(scenario.satisfaction, minSatisfaction, maxSatisfaction);
            const score = (normalizedIncome * incomeWeight) + (normalizedSatisfaction * satisfactionWeight);

            if (score > maxScore) {
                maxScore = score;
                bestScenario = scenario;
            }
        });

        this.elements.recommendedJobName.textContent = bestScenario.name;
        this.updateRecommendationReason(bestScenario, incomeWeight);

        // 모든 카드에서 recommended 클래스 제거
        document.querySelectorAll('.result-card').forEach(card => {
            card.classList.remove('recommended');
        });

        // 추천된 시나리오에 해당하는 카드에 recommended 클래스 추가
        const recommendedCard = document.querySelector(`.result-card[data-scenario-id="${bestScenario.id}"]`);
        if (recommendedCard) {
            recommendedCard.classList.add('recommended');
        }

        this.updateDistributionCharts(bestScenario);
    },

    updateRecommendationReason: function (bestScenario, incomeWeight) {
        const incomeText = `<strong>${(bestScenario.income * 100).toFixed(2)}%</strong>`;
        const satisText = `<strong>${bestScenario.satisfaction.toFixed(2)}점</strong>`;
        let reason = "";

        if (incomeWeight > 0.7) {
            reason = `소득 상승(${incomeText})을 가장 중요하게 고려했을 때 가장 유리한 선택입니다. 이때 예상되는 만족도 변화는 ${satisText}입니다.`;
        } else if (incomeWeight < 0.3) {
            reason = `직무 만족도 향상(${satisText})을 가장 중요하게 고려했을 때 가장 적합한 선택입니다. 이때 예상되는 소득 변화율은 ${incomeText}입니다.`;
        } else {
            reason = `소득과 만족도의 균형을 고려했을 때, 소득(${incomeText}), 만족도(${satisText}) 양쪽에서 가장 안정적인 결과를 보여줍니다.`;
        }
        this.elements.recommendationReason.innerHTML = reason; // innerHTML을 사용하여 strong 태그 렌더링
    },

    normalize: function (value, min, max) {
        if (max === min) return 0.5;
        return (value - min) / (max - min);
    },

    createCharts: function () {
        // 1. 플러그인 등록
        Chart.register(ChartDataLabels);
        Chart.defaults.font.family = "'Pretendard', sans-serif";
        Chart.defaults.plugins.datalabels.anchor = 'end';
        Chart.defaults.plugins.datalabels.align = 'top';
        Chart.defaults.plugins.datalabels.font = { weight: 'bold' };

        const labels = this.scenarios.map(s => s.name);
        const incomeData = this.scenarios.map(s => s.income * 100);
        const satisfactionData = this.scenarios.map(s => s.satisfaction);
        const count = this.scenarios.length;

        // 2. 세련된 컬러 팔레트 정의 (개선안)
        const bgColors = ['rgba(96, 165, 250, 0.6)', 'rgba(74, 222, 128, 0.6)', 'rgba(250, 204, 21, 0.6)'].slice(0, count);
        const borderColors = ['rgba(96, 165, 250, 1)', 'rgba(74, 222, 128, 1)', 'rgba(250, 204, 21, 1)'].slice(0, count);

        const createChart = (canvas, chartLabel, data, unit) => {
            if (!canvas) return;

            const dataMax = Math.max(...data, 0);
            const suggestedMax = dataMax > 0 ? dataMax * 1.2 : 1;

            new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: chartLabel,
                        data: data,
                        backgroundColor: bgColors,
                        borderColor: borderColors,
                        borderWidth: 2,
                        borderRadius: 8,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: { top: 30 }
                    },
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: '#444',
                            formatter: function(value) {
                                return value.toFixed(2) + unit;
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { display: false },
                            ticks: {
                                font: { size: 13 },
                                callback: function (value) {
                                    const label = this.getLabelForValue(value);
                                    return label.split(' ').join('\n');
                                }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            suggestedMax: suggestedMax,
                            grid: {
                                // 5. 그리드 라인 스타일 개선
                                color: '#e9e9e9',
                                borderDash: [5, 5], // 점선으로 변경
                                drawBorder: false, // 축 경계선 제거
                            },
                            ticks: {
                                font: { size: 13 },
                                callback: function (value) {
                                    return `${value.toFixed(1)} ${unit}`;
                                }
                            }
                        }
                    }
                }
            });
        };

        createChart(this.elements.incomeChartCanvas, '월 소득 변화율', incomeData, '%');
        createChart(this.elements.satisfactionChartCanvas, '직무 만족도 변화', satisfactionData, '점');
    },

    createDistributionCharts: function () {
        this.distributionCharts = {};
        const createChart = (canvas, label) => {
            if (!canvas) return null;
            return new Chart(canvas, {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: label,
                        data: [],
                        backgroundColor: 'rgba(124, 185, 232, 0.7)',
                        borderColor: 'rgba(124, 185, 232, 1)',
                        borderWidth: 2,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    layout: {
                        padding: { top: 30 }
                    },
                    plugins: {
                        legend: { display: false },
                        datalabels: {
                            color: '#555',
                            font: { weight: '500' },
                            formatter: function(value, context) {
                                return value > 0 ? value : ''; // 0보다 클 때만 라벨 표시
                            },
                            // 동적 위치 조정 로직
                            align: function(context) {
                                const chart = context.chart;
                                const area = chart.chartArea;
                                const value = context.dataset.data[context.dataIndex];
                                const topValue = chart.scales.y.max;
                                // 막대 높이가 전체의 85% 이상이면 라벨을 안쪽으로
                                return (value / topValue) > 0.85 ? 'bottom' : 'top';
                            },
                            anchor: 'end'
                        }
                    },
                    scales: {
                        y: { 
                            title: { display: true, text: '사례 수 (명)' },
                            grid: { color: '#e9e9e9', drawBorder: false },
                            ticks: { font: { size: 13 } }
                        },
                        x: { 
                            title: { display: true, text: '변화량 구간' },
                            grid: { display: false },
                            ticks: { font: { size: 12 } }
                        }
                    }
                }
            });
        };
        this.distributionCharts.income = createChart(this.elements.incomeDistributionChartCanvas, '소득 변화율 분포');
        this.distributionCharts.satisfaction = createChart(this.elements.satisfactionDistributionChartCanvas, '만족도 변화 분포');
    },

    updateDistributionCharts: function (scenario) {
        if (!this.elements.distributionChartTitle) return;
        const distributionData = scenario.distribution;
        this.elements.distributionChartTitle.textContent = scenario.name;

        if (!distributionData) {
            this.clearDistributionChart('income', '유사 사례가 부족하여 분포를 표시할 수 없습니다.');
            this.clearDistributionChart('satisfaction', '유사 사례가 부족하여 분포를 표시할 수 없습니다.');
            return;
        }

        this.updateSingleDistributionChart(this.distributionCharts.income, distributionData.income, '%');
        this.updateSingleDistributionChart(this.distributionCharts.satisfaction, distributionData.satisfaction, '점');
    },

    updateSingleDistributionChart: function (chart, data, unit) {
        if (!chart) return;
        const labels = data.bins.slice(0, -1).map((bin, i) => {
            const nextBin = data.bins[i + 1];
            // 소득 변화율(%)은 소수점 1자리까지, 만족도(점)는 소수점 1자리까지 표시
            const start = unit === '%' ? (bin * 100).toFixed(1) : bin.toFixed(1);
            const end = unit === '%' ? (nextBin * 100).toFixed(1) : nextBin.toFixed(1);
            return `${start}~${end}${unit}`;
        });
        chart.data.labels = labels;
        chart.data.datasets[0].data = data.counts;
        chart.update();
    },

    clearDistributionChart: function (chartKey, message) {
        const chart = this.distributionCharts[chartKey];
        if (!chart) return;
        chart.data.labels = [message];
        chart.data.datasets[0].data = [0];
        chart.update();
    }
};

document.addEventListener("DOMContentLoaded", () => ResultPageManager.init());
