async function sendMessage() {
    const input = document.getElementById('user-input');
    const userMessage = input.value.trim();
    
    if (!userMessage) return;

    // Mostrar mensaje del usuario
    appendMessage('user', userMessage);
    input.value = '';

    try {
        const response = await fetch('https://tu-backend.com/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userMessage })
        });
        
        const data = await response.json();
        appendMessage('bot', data.response);

    } catch (error) {
        appendMessage('bot', "❌ Error al conectar con el servidor");
    }
}

function appendMessage(sender, text) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll
}