document.addEventListener("DOMContentLoaded", function() {
    // --- Accordion Logic ---
    const headers = document.querySelectorAll(".accordion-header");
    headers.forEach(header => {
        header.addEventListener("click", () => {
            // 이미 active된 다른 헤더를 비활성화하지 않고, 개별 토글만 수행
            header.classList.toggle("active");
            const body = header.nextElementSibling;
            if (body.style.display === "block") {
                body.style.display = "none";
            } else {
                body.style.display = "block";
            }
        });
    });

    // --- Feedback Widget Logic ---
    const feedbackButtons = document.querySelectorAll('.feedback-btn');
    feedbackButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            // 이벤트 버블링을 막아 아코디언이 닫히지 않도록 함
            event.stopPropagation(); 
            
            const widget = this.closest('.feedback-widget');
            if (widget) {
                const buttonsContainer = widget.querySelector('.feedback-buttons');
                const thanksMessage = widget.querySelector('.feedback-thanks');

                if (buttonsContainer) {
                    buttonsContainer.style.display = 'none';
                }
                if (thanksMessage) {
                    thanksMessage.style.display = 'inline';
                }
            }
        });
    });
});