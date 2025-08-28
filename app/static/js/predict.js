document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predictionForm");
    const loadingOverlay = document.getElementById("loadingOverlay");

    // 성별 버튼 클릭
    document.querySelectorAll(".gender-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const value = btn.getAttribute("data-value");
            document.getElementById("gender").value = value;

            // 버튼 스타일 활성화
            document.querySelectorAll(".gender-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
        });
    });

    // 만족도 슬라이더 값 업데이트
    document.querySelectorAll(".satisfaction-slider").forEach(slider => {
        const valueDisplay = document.getElementById(slider.id + "Value");
        
        // 초기값 설정
        valueDisplay.textContent = slider.value;
        
        // 값 변경 이벤트
        slider.addEventListener("input", () => {
            valueDisplay.textContent = slider.value;
        });
    });

    form.addEventListener("submit", (e) => {
        // 기본 제출 막음
        e.preventDefault();

        // 나이 입력은 직접 받으므로 birth 처리 불필요

        // 필수 필드 유효성 검사
        const requiredFields = document.querySelectorAll('#predictionForm [required]');
        for (let i = 0; i < requiredFields.length; i++) {
            const field = requiredFields[i];
            if (field.type === 'radio') {
                const radioGroup = document.getElementsByName(field.name);
                let isChecked = false;
                for (let j = 0; j < radioGroup.length; j++) {
                    if (radioGroup[j].checked) {
                        isChecked = true;
                        break;
                    }
                }
                if (!isChecked) {
                    alert('모든 필수 항목을 입력해주세요.');
                    return;
                }
            } else if (field.value.trim() === '') {
                alert('모든 필수 항목을 입력해주세요.');
                field.focus();
                return;
            }
        }

        // 성별 hidden 필드 선택 여부 확인
        if (document.getElementById("gender").value === "") {
            alert('성별을 선택해주세요.');
            return;
        }

        // 연령 유효성 검사
        const ageInput = document.getElementById("age");
        const age = parseInt(ageInput.value);
        if (age < 15 || age > 90) {
            alert('나이는 15세부터 90세까지 입력 가능합니다.');
            ageInput.focus();
            return;
        }

        // 로딩 오버레이 표시
        loadingOverlay.style.display = "flex";
        
        // 폼 제출
        form.submit();
    });
});