// advice.js

document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('chatWindow');
    const aiFullResponse = document.getElementById('aiFullResponse').textContent.trim();
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');

    let chatHistory = []; // 대화 내역을 저장할 배열

    // 새로운 타이핑 효과 유틸리티 함수
    function typeMessageWithEffect(element, textToType, delayPerChar = 20, punctuationDelay = 100, onComplete = () => {}) {
        let currentIndex = 0;
        const bubble = element.querySelector('.message-bubble');
        bubble.innerHTML = '<span class="streaming-cursor">|</span>'; // 초기 커서 표시

        function typeNextChar() {
            if (currentIndex < textToType.length) {
                const char = textToType[currentIndex];
                bubble.innerHTML = textToType.substring(0, currentIndex + 1).replace(/\n/g, '<br>') + '<span class="streaming-cursor">|</span>';
                currentIndex++;
                
                const delay = (char === '.' || char === '!' || char === '?') ? punctuationDelay : delayPerChar;
                setTimeout(typeNextChar, delay);
            } else {
                // 타이핑 완료
                bubble.innerHTML = textToType.replace(/\n/g, '<br>'); // 커서 제거
                onComplete();
            }
            chatWindow.scrollTop = chatWindow.scrollHeight;
        }
        typeNextChar();
    }

    // 초기 AI 조언을 타이핑 효과와 함께 점진적으로 표시하는 함수
    function showInitialMessageWithTyping() {
        const messageElement = createStreamingMessageContainer();
        const fullText = aiFullResponse;
        
        removeTypingIndicator();
        
        typeMessageWithEffect(messageElement, fullText, 20, 100, () => {
            chatHistory.push({ sender: 'ai', text: fullText });
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        });
    }

    // 메시지를 채팅창에 추가하는 함수
    function addMessageToChat(sender, text) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = sender === 'ai' ? '🤖' : '👤';

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        bubble.innerHTML = text.replace(/\n/g, '<br>');

        messageElement.appendChild(avatar);
        if (sender === 'user') {
            // 사용자 메시지는 오른쪽에 정렬되도록 bubble을 먼저 추가
            messageElement.insertBefore(bubble, avatar);
        } else {
            messageElement.appendChild(bubble);
        }
        
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // 타이핑 인디케이터를 보여주는 함수
    function showTypingIndicator() {
        if (chatWindow.querySelector('.typing-indicator-wrapper')) return; // 이미 있으면 추가 안함

        const indicatorWrapper = document.createElement('div');
        indicatorWrapper.classList.add('chat-message', 'ai', 'typing-indicator-wrapper');
        
        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = '🤖';

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        
        const indicator = document.createElement('div');
        indicator.classList.add('typing-indicator');
        indicator.innerHTML = '<span></span><span></span><span></span>';
        
        bubble.appendChild(indicator);
        indicatorWrapper.appendChild(avatar);
        indicatorWrapper.appendChild(bubble);
        chatWindow.appendChild(indicatorWrapper);
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // 타이핑 인디케이터를 제거하는 함수
    function removeTypingIndicator() {
        const typingIndicator = chatWindow.querySelector('.typing-indicator-wrapper');
        if (typingIndicator) {
            chatWindow.removeChild(typingIndicator);
        }
    }

    // 실시간 스트리밍 응답을 처리하는 함수
    async function sendMessageWithStreaming() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessageToChat('user', message);
        chatHistory.push({ sender: 'user', text: message });
        userInput.value = '';
        userInput.style.height = 'auto';
        
        userInput.disabled = true; // 입력 비활성화
        sendButton.disabled = false; // 버튼 비활성화

        // 스트리밍용 AI 메시지 컨테이너 생성
        const aiMessageContainer = createStreamingMessageContainer();
        let fullResponse = ''; // 전체 응답을 누적

        try {
            const response = await fetch('/ask-ai-stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory.slice(0, -1)
                }),
            });

            if (!response.ok) {
                throw new Error('서버에서 오류가 발생했습니다.');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            // 청크를 순차적으로 타이핑 효과로 표시
            const processChunk = async (chunkText) => {
                return new Promise(resolve => {
                    typeMessageWithEffect(aiMessageContainer, chunkText, 20, 100, resolve);
                });
            };

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            
                            if (data.error) {
                                throw new Error(data.error);
                            }
                            
                            if (data.chunk) {
                                fullResponse += data.chunk;
                                await processChunk(data.chunk);
                            }
                            
                            if (data.done) {
                                const finalContent = data.full_response || fullResponse;
                                finalizeStreamingMessage(aiMessageContainer, finalContent);
                                chatHistory.push({ sender: 'ai', text: finalContent });
                                userInput.disabled = false; // 입력 활성화
                                sendButton.disabled = false; // 버튼 활성화
                                userInput.focus();
                                return;
                            }
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', line);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Streaming Error:', error);
            removeStreamingMessage(aiMessageContainer);
            addMessageToChat('ai', '죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
            userInput.disabled = false; // 입력 활성화
            sendButton.disabled = false; // 버튼 활성화
            userInput.focus();
        }
    }

    // 스트리밍용 메시지 컨테이너 생성
    function createStreamingMessageContainer() {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', 'ai', 'streaming-message');

        const avatar = document.createElement('div');
        avatar.classList.add('avatar');
        avatar.innerHTML = '🤖';

        const bubble = document.createElement('div');
        bubble.classList.add('message-bubble');
        bubble.innerHTML = '<span class="streaming-cursor">|</span>';

        messageElement.appendChild(avatar);
        messageElement.appendChild(bubble);
        chatWindow.appendChild(messageElement);
        chatWindow.scrollTop = chatWindow.scrollHeight;

        return messageElement;
    }

    // 스트리밍 메시지 업데이트
    function updateStreamingMessage(container, text) {
        const bubble = container.querySelector('.message-bubble');
        bubble.innerHTML = text.replace(/\n/g, '<br>') + '<span class="streaming-cursor">|</span>';
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // 스트리밍 메시지 완료
    function finalizeStreamingMessage(container, text) {
        const bubble = container.querySelector('.message-bubble');
        bubble.innerHTML = text.replace(/\n/g, '<br>');
        container.classList.remove('streaming-message');
    }

    // 스트리밍 메시지 제거
    function removeStreamingMessage(container) {
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    }

    // 기존 방식 메시지 전송 (폴백용)
    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        addMessageToChat('user', message);
        chatHistory.push({ sender: 'user', text: message });
        userInput.value = '';
        userInput.style.height = 'auto'; // 높이 초기화
        showTypingIndicator();

        userInput.disabled = true; // 입력 비활성화
        sendButton.disabled = true; // 버튼 비활성화

        try {
            const response = await fetch('/ask-ai', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory.slice(0, -1), // 마지막 사용자 메시지는 제외하고 전송
                    streaming: false
                }),
            });

            if (!response.ok) {
                throw new Error('서버에서 오류가 발생했습니다.');
            }

            const data = await response.json();
            removeTypingIndicator();
            addMessageToChat('ai', data.reply);
            chatHistory.push({ sender: 'ai', text: data.reply });
            userInput.disabled = false; // 입력 활성화
            sendButton.disabled = false; // 버튼 활성화
            userInput.focus();

        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessageToChat('ai', '죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
            userInput.disabled = false; // 입력 활성화
            sendButton.disabled = false; // 버튼 활성화
            userInput.focus();
        }
    }

    // 스트리밍 지원 여부 확인 후 적절한 함수 사용
    function sendMessageSmartly() {
        // 현재는 스트리밍 기능을 비활성화하고 일반 방식 사용
        // 추후 스트리밍 안정화 후 활성화
        sendMessage(); // 안정적인 일반 방식 사용
        
        // if (typeof EventSource !== 'undefined') {
        //     sendMessageWithStreaming();
        // } else {
        //     sendMessage(); // 폴백
        // }
    }

    // 이벤트 리스너 설정 - 스트리밍 지원
    sendButton.addEventListener('click', sendMessageSmartly);
    userInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessageSmartly();
        }
    });

    // textarea 높이 자동 조절
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = (userInput.scrollHeight) + 'px';
    });


    // 초기화 및 첫 메시지 표시 시작
    userInput.disabled = true;
    sendButton.disabled = true;
    showTypingIndicator();
    
    // AI 조언을 타이핑 효과로 점진적으로 표시
    setTimeout(() => {
        if (aiFullResponse) {
            showInitialMessageWithTyping();
        } else {
            // AI 조언이 없는 경우, 기본 인사말 표시
            removeTypingIndicator();
            addMessageToChat('ai', '안녕하세요! 커리어 관련 질문이 있으신가요?');
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        }
    }, 1500);
});
