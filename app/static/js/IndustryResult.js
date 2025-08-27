document.addEventListener("DOMContentLoaded", () => {
    // console.log(rangeLimit);
    // console.log(mainType);
    // console.log(dataMap);

});

const ctx = document.getElementById('industryChart').getContext('2d');
var now = new Date();	// 현재 날짜 및 시간
var year = now.getFullYear();	// 연도
const years = [];
for (let i = 0; i < rangeLimit; i++) {
  years.push(year+i); 
}
const industryMap = {
    total: "전체",
    agriculture: "농업임업및어업",
    mining: "광업",
    manufacturing: "제조업",
    electricity: "전기가스수도하수",
    construction: "건설업",
    wholesale: "도매및소매업",
    transport: "운수및창고업",
    hospitality: "숙박및음식점업",
    it: "정보통신업",
    finance: "금융및보험업",
    realestate: "부동산업시설관리지원임대",
    professional: "전문과학및기술서비스업",
    education: "교육서비스업",
    health: "보건업및사회복지서비스업",
    culture: "오락문화및운동관련서비스업",
    other: "기타공공수리및개인서비스업"
};
const industryColors = [
    '#3498db', '#2ecc71', '#1abc9c', '#e74c3c', '#f39c12', '#d35400',
    '#9b59b6', '#34495e', '#c0392b', '#8e44ad', '#2980b9', '#27ae60',
    '#e67e22', '#16a085', '#bdc3c7', '#7f8c8d', '#95a5a6'
];
let industryData = {};
let colorIndex = 0;
for (const [key, name] of Object.entries(industryMap)) {
    let values = [];
    for (const year of years) {
        const lookupKey = `${year}_${name}`;
        console.log(lookupKey)
        if (dataMap[lookupKey]) {
            values.push(dataMap[lookupKey].data);
            console.log(dataMap[lookupKey].data)
        } else {
            values.push(null); // 또는 0 또는 NaN 등
        }
    }
    industryData[key] = {
        name: name,
        data: values,
        color: industryColors[colorIndex++ % industryColors.length]
    };
}

let selectedIndustries = ['total'];
let industryChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: years,
        datasets: []
    },
    options: {
        responsive: true,
        maintainAspectRatio: false, // CSS 크기를 따르도록 재설정
        plugins: {
            legend: { display: true }
        },
        scales: {
            y: {
                ticks: {
                    callback: val => val + '%'
                },
                min: -6,
                max: 6
            }
        }
    }
});function updateChart() {
    // 모든 데이터 값 추출
    let allValues = [];
    selectedIndustries.forEach(key => {
        const item = industryData[key];
        allValues = allValues.concat(item.data);
    });

    // 최댓값과 최솟값 계산
    const maxVal = Math.max(...allValues);
    const minVal = Math.min(...allValues);

    // 차트에 적용
    industryChart.options.scales.y.max = Math.ceil(maxVal) + 3;
    industryChart.options.scales.y.min = Math.floor(minVal) - 3;

    // 데이터셋 재설정
    industryChart.data.datasets = selectedIndustries.map(key => {
        const item = industryData[key];
        return {
            label: item.name,
            data: item.data,
            borderColor: item.color,
            backgroundColor: item.color + '20',
            fill: true,
            tension: 0.3
        };
    });

    industryChart.update();
}
function updateList() {
    const list = document.getElementById('industryList');
    list.innerHTML = '';
    selectedIndustries.forEach(key => {
        if (key === 'total') return;
        const item = document.createElement('div');
        item.className = 'industry-item';
        item.innerHTML = `
            <span>${industryData[key].name}</span>
            <div>
                <span class="industry-color" style="background:${industryData[key].color}"></span>
                <button class="remove-btn" onclick="removeIndustry('${key}')">&times;</button>
            </div>
        `;
        list.appendChild(item);
    });
}
function removeIndustry(key) {
    selectedIndustries = selectedIndustries.filter(k => k !== key);
    updateChart();
    updateList();
}
document.getElementById('addIndustryBtn').addEventListener('click', () => {
    const value = document.getElementById('industrySelect').value;
    if (value && !selectedIndustries.includes(value)) {
        selectedIndustries.push(value);
        updateChart();   // ✅ 차트 갱신
        updateList();    // ✅ 목록 갱신
    }
});
updateChart();