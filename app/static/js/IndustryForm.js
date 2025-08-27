
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("IndustryForm");
  const loadingOverlay = document.getElementById("loadingOverlay");

  form.addEventListener("submit", function (e) {
    e.preventDefault(); // 🔸 실제 폼 제출 막기

    // 🔸 여기서 필요한 값들 읽어오기
    const range = document.getElementById("rangeLimit").value;
    //const targetIndustry = document.getElementById("industry").value;
    const detail = document.getElementById("detail").value;

    console.log("범위:", range);
    //console.log("기준 산업:", targetIndustry);
    console.log("주요 분석 항목:", detail);

    loadingOverlay.style.display = "flex";
	// 폼 제출
	
    form.submit();
  });
});