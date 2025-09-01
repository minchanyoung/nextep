document.addEventListener('DOMContentLoaded', function () {
    // Initial data from the server (already in the HTML script block)
    let predictionResults = predictionResultsRaw;

    // DOM Elements
    const jobASelect = document.getElementById('jobACategorySelect');
    const jobBSelect = document.getElementById('jobBCategorySelect');
    const prioritySlider = document.getElementById('prioritySlider');
    const priorityLabel = document.getElementById('priorityLabel');
    const form = document.getElementById('predictionForm');
    const container = document.getElementById('predictionResultsContainer');

    // Chart instances
    let incomeChart, satisfactionChart, incomeDistributionChart, satisfactionDistributionChart;

    // Helper functions to format data (replicating Jinja2 filters)
    const formatIncomeChange = (rate) => {
        if (rate == null) return 'N/A';
        const arrow = rate > 0 ? '▲ ' : (rate < 0 ? '▼ ' : '');
        return `${arrow}${(rate * 100).toFixed(2)}%`;
    };

    const formatSatisfactionChange = (score) => {
        if (score == null) return 'N/A';
        const arrow = score > 0 ? '▲ ' : (score < 0 ? '▼ ' : '');
        return `${arrow}${score.toFixed(2)}점`;
    };

    

    // Function to update a single result card
    const updateCard = (scenarioId, resultData, jobName) => {
        const card = container.querySelector(`.result-card[data-scenario-id="${scenarioId}"]`);
        if (!card || !resultData) return;

        const nameEl = card.querySelector('h4');
        const resultItems = card.querySelectorAll('.result-item');

        if (jobName && nameEl) {
            nameEl.textContent = jobName;
        }

        if (resultItems.length > 0) {
            const incomeEl = resultItems[0].querySelector('.value');
            if (incomeEl) {
                incomeEl.textContent = formatIncomeChange(resultData.income_change_rate);
                // Use class from server response
                incomeEl.className = `value ${resultData.income_class || 'neutral'}`;
            }
        }

        if (resultItems.length > 1) {
            const satisEl = resultItems[1].querySelector('.value');
            if (satisEl) {
                satisEl.textContent = formatSatisfactionChange(resultData.satisfaction_change_score);
                // Use class from server response
                satisEl.className = `value ${resultData.satisfaction_class || 'neutral'}`;
            }
        }
    };

    // Function to update all summary cards
    const updateAllCards = (results) => {
        updateCard('current', results[0]);
        updateCard('jobA', results[1], jobCategoryMapJs[jobASelect.value]);
        
        const jobBCard = container.querySelector('.result-card[data-scenario-id="jobB"]');
        if (results.length > 2 && results[2]) {
            updateCard('jobB', results[2], jobCategoryMapJs[jobBSelect.value]);
            jobBCard.style.display = '';
        } else {
            jobBCard.style.display = 'none';
        }
    };

    // Function to update charts
    const updateCharts = (results) => {
        const labels = [
            '현직 유지',
            jobCategoryMapJs[jobASelect.value],
            jobCategoryMapJs[jobBSelect.value]
        ].slice(0, results.length);

        const incomeData = results.map(r => (r.income_change_rate * 100).toFixed(1));
        const satisfactionData = results.map(r => r.satisfaction_change_score.toFixed(2));

        // Update Income Chart
        incomeChart.data.labels = labels;
        incomeChart.data.datasets[0].data = incomeData;
        incomeChart.update();

        // Update Satisfaction Chart
        satisfactionChart.data.labels = labels;
        satisfactionChart.data.datasets[0].data = satisfactionData;
        satisfactionChart.update();
    };
    
    // Function to update AI recommendation
    const updateRecommendation = (results) => {
        const priority = prioritySlider.value / 100; // 0 for satisfaction, 1 for income
        const weightSatis = 1 - priority;
        const weightIncome = priority;

        // Normalize scores to be comparable (simple min-max scaling)
        const incomeRates = results.map(r => r.income_change_rate);
        const satisScores = results.map(r => r.satisfaction_change_score);

        const minIncome = Math.min(...incomeRates);
        const maxIncome = Math.max(...incomeRates);
        const minSatis = Math.min(...satisScores);
        const maxSatis = Math.max(...satisScores);

        const normalizedScores = results.map((r, i) => {
            const normIncome = (maxIncome - minIncome) === 0 ? 0.5 : (r.income_change_rate - minIncome) / (maxIncome - minIncome);
            const normSatis = (maxSatis - minSatis) === 0 ? 0.5 : (r.satisfaction_change_score - minSatis) / (maxSatis - minSatis);
            return (normIncome * weightIncome) + (normSatis * weightSatis);
        });

        const bestIndex = normalizedScores.indexOf(Math.max(...normalizedScores));
        
        const scenarioNames = [
            '현직 유지',
            jobCategoryMapJs[jobASelect.value],
            jobCategoryMapJs[jobBSelect.value]
        ];
        const recommendedJobName = scenarioNames[bestIndex];
        const recommendedResult = results[bestIndex];

        document.getElementById('recommendedJobName').textContent = recommendedJobName;
        
        let reason = `<strong>${recommendedJobName}</strong> 선택 시, `;
        reason += `소득 ${formatIncomeChange(recommendedResult.income_change_rate)} 및 `;
        reason += `만족도 ${formatSatisfactionChange(recommendedResult.satisfaction_change_score)}의 변화가 예상됩니다.`;
        document.getElementById('recommendationReason').innerHTML = reason;

        // Update recommendation badges on cards
        document.querySelectorAll('.result-card').forEach((card, i) => {
            const badge = card.querySelector('.recommend-badge');
            if (i === bestIndex) {
                badge.style.display = 'block';
            } else {
                badge.style.display = 'none';
            }
        });
        
        // Update distribution chart title
        document.getElementById('distributionChartTitle').textContent = recommendedJobName;
    };


    // Main function to handle dynamic updates
    const handleUpdate = async () => {
        const mainContent = document.querySelector('.main-content');
        mainContent.classList.add('loading'); // Add loading overlay

        const formData = new FormData(form);
        // Ensure the latest dropdown values are in the form data
        formData.set('job_A_category', jobASelect.value);
        formData.set('job_B_category', jobBSelect.value);

        try {
            const response = await fetch(form.action, {
                method: 'POST',
                headers: {
                    'X-Requested-With': 'XMLHttpRequest' // Important for server to know it's an AJAX call
                },
                body: new URLSearchParams(formData) // Send as form data
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const newResults = await response.json();
            console.log('Server response:', JSON.stringify(newResults)); // For debugging
            predictionResults = newResults.prediction_results; // Update global results
            
            // Update UI components with new data
            updateAllCards(predictionResults);
            updateCharts(predictionResults);
            updateRecommendation(predictionResults);
            // Note: Distribution charts would also need an update mechanism if their data changes.
            // This would require another AJAX call or for the initial call to return more data.
            // For now, we just update the title.

        } catch (error) {
            console.error('Error updating prediction:', error);
            // Optionally, display an error message to the user
        } finally {
            mainContent.classList.remove('loading'); // Remove loading overlay
        }
    };

    // --- Event Listeners ---
    jobASelect.addEventListener('change', handleUpdate);
    jobBSelect.addEventListener('change', handleUpdate);
    prioritySlider.addEventListener('input', () => {
        const satisPercent = 100 - prioritySlider.value;
        const incomePercent = prioritySlider.value;
        priorityLabel.textContent = `균형 (${satisPercent}:${incomePercent})`;
        updateRecommendation(predictionResults); // Re-calculate recommendation without a server call
    });

    // --- Tab switching logic ---
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;
            
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            tabContents.forEach(content => {
                if (content.id === tabId) {
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                }
            });
        });
    });
    // Activate the first tab by default
    if(tabButtons.length > 0) {
        tabButtons[0].classList.add('active');
        tabContents[0].classList.add('active');
    }


    // --- Chart.js Initialization ---
    const createChart = (ctx, type, labels, data, label, backgroundColor) => {
        return new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: backgroundColor,
                    borderColor: backgroundColor.map(c => c.replace('0.6', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: {
                        anchor: 'end',
                        align: 'top',
                        formatter: (value) => value,
                        font: {
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false
                    }
                }
            },
            plugins: [ChartDataLabels]
        });
    };
    
    const createDistributionChart = (ctx, data, label) => {
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => d.range),
                datasets: [{
                    label: label,
                    data: data.map(d => d.count),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    datalabels: { display: false }
                },
                scales: {
                    y: {
                        title: { display: true, text: '사례 수' }
                    }
                }
            }
        });
    };

    const initCharts = () => {
        const initialLabels = [
            '현직 유지',
            jobCategoryMapJs[selectedJobACategory],
            jobCategoryMapJs[selectedJobBCategory]
        ].slice(0, predictionResults.length);

        const incomeData = predictionResults.map(r => (r.income_change_rate * 100).toFixed(1));
        const satisfactionData = predictionResults.map(r => r.satisfaction_change_score.toFixed(2));
        
        const chartColors = [
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 99, 132, 0.6)',
            'rgba(75, 192, 192, 0.6)'
        ];

        // Scenario Comparison Charts
        incomeChart = createChart(
            document.getElementById('incomeChart').getContext('2d'),
            'bar',
            initialLabels,
            incomeData,
            '월 소득 변화율 (%)',
            chartColors
        );

        satisfactionChart = createChart(
            document.getElementById('satisfactionChart').getContext('2d'),
            'bar',
            initialLabels,
            satisfactionData,
            '직무 만족도 변화 (점)',
            chartColors
        );

        // Distribution Charts (using dummy data for now, as it's not provided dynamically)
        // In a real scenario, this data would come from an AJAX call
        const dummyDistData = [
            { range: '-20% 이상', count: 15 },
            { range: '-20% ~ -10%', count: 25 },
            { range: '-10% ~ 0%', count: 40 },
            { range: '0% ~ 10%', count: 50 },
            { range: '10% ~ 20%', count: 30 },
            { range: '20% 이상', count: 20 }
        ];
        incomeDistributionChart = createDistributionChart(
            document.getElementById('incomeDistributionChart').getContext('2d'),
            dummyDistData,
            '소득 변화율 분포'
        );
        satisfactionDistributionChart = createDistributionChart(
            document.getElementById('satisfactionDistributionChart').getContext('2d'),
            dummyDistData.map(d => ({...d, range: d.range.replace('%', '점')})),
            '만족도 변화 분포'
        );
    };

    // --- Initial UI setup ---
    initCharts();
    updateRecommendation(predictionResults); // Set initial recommendation
});

// Add some CSS for the loading state
const style = document.createElement('style');
style.innerHTML = `
.main-content.loading {
    position: relative;
    pointer-events: none;
}
.main-content.loading::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(255, 255, 255, 0.7);
    z-index: 1000;
}
.main-content.loading::after {
    content: '업데이트 중...';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 1001;
    font-size: 1.2rem;
    font-weight: bold;
    color: var(--primary-color);
}
`;
document.head.appendChild(style);