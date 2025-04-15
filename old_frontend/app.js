let conversationHistory = [];

async function sendMessage() {
    const input = document.getElementById('user-input');
    const userMessage = input.value.trim();
    
    if (!userMessage) return;

    // Mostrar mensaje del usuario
    appendMessage('user', userMessage);
    input.value = '';

    try {
        const response = await fetch('https://ai-chatbot-backend-a0x5.onrender.com/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: userMessage,
                history: conversationHistory // 👈 Enviar historial acumulado
            })
        });
        
        const data = await response.json();
        appendMessage('bot', data.response);

        // Actualizar el historial con ambas partes
        conversationHistory.push({ role: "user", content: userMessage });
        conversationHistory.push({ role: "assistant", content: data.response });

    } catch (error) {
        console.error("Error:", error);
        appendMessage('bot', "❌ Error: " + (error.message || "Verifica la consola para más detalles"));
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