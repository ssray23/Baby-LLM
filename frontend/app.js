const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');

thresholdSlider.addEventListener('input', (e) => {
    thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
});

// Fallback for light-dismiss if closedby is not supported natively
const settingsModal = document.getElementById('settings-modal');
if (!('closedBy' in HTMLDialogElement.prototype)) {
    settingsModal.addEventListener('click', (event) => {
        if (event.target !== settingsModal) return;
        const rect = settingsModal.getBoundingClientRect();
        const isDialogContent = (
            rect.top <= event.clientY &&
            event.clientY <= rect.top + rect.height &&
            rect.left <= event.clientX &&
            event.clientX <= rect.left + rect.width
        );
        if (isDialogContent) return;
        settingsModal.close();
    });
}

function addMessage(text, isUser) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message');
    msgDiv.classList.add(isUser ? 'user-message' : 'ai-message');
    
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    bubble.textContent = text;
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    
    chatBox.scrollTop = chatBox.scrollHeight;
    return bubble;
}

function showTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'ai-message');
    msgDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.classList.add('avatar');
    avatar.textContent = '🤖';
    
    const bubble = document.createElement('div');
    bubble.classList.add('bubble', 'typing-indicator');
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('dot');
        bubble.appendChild(dot);
    }
    
    msgDiv.appendChild(avatar);
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text) return;
    
    // Add user message
    addMessage(text, true);
    userInput.value = '';
    
    showTypingIndicator();
    
    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                prompt: text, 
                max_tokens: 150,
                retrieval_threshold: parseFloat(thresholdSlider.value)
            })
        });
        
        removeTypingIndicator();
        
        if (response.ok) {
            const data = await response.json();
            addMessage(data.response, false);
        } else {
            const err = await response.json();
            addMessage(`Error: ${err.error}`, false);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage('Error connecting to the server. Is it running?', false);
    }
}

sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSend();
    }
});

const trainBtn = document.getElementById('train-btn');

trainBtn.addEventListener('click', async () => {
    if (trainBtn.classList.contains('loading')) return;
    
    trainBtn.classList.add('loading');
    const originalText = trainBtn.innerHTML;
    trainBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="2" x2="12" y2="6"></line>
            <line x1="12" y1="18" x2="12" y2="22"></line>
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
            <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
            <line x1="2" y1="12" x2="6" y2="12"></line>
            <line x1="18" y1="12" x2="22" y2="12"></line>
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
            <line x1="16.24" y1="4.93" x2="19.07" y2="7.76"></line>
        </svg>
        Training...
    `;
    
    // Add system message
    addMessage("Started training on data/*.txt... This may take a moment.", false);
    
    try {
        const response = await fetch('/train', { method: 'POST' });
        const data = await response.json();
        
        if (response.ok) {
            addMessage(data.response, false);
        } else {
            addMessage(`Training error: ${data.error}`, false);
        }
    } catch (error) {
        addMessage('Error connecting to the server while trying to train.', false);
    } finally {
        trainBtn.classList.remove('loading');
        trainBtn.innerHTML = originalText;
    }
});
