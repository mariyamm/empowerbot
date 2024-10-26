document.getElementById("start").onclick = function() {
    const phone = document.getElementById("phone").value;
    if (!phone) {
        alert("Please enter your phone number.");
        return;
    }

    // Start the chat session
    fetch('/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phone_number: phone }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById("question-section").style.display = "flex";
            displayMessage(`You: ${data.message}`, 'user');
        }
    });
};

document.getElementById("chat").onclick = function() {
    const question = document.getElementById("question").value;
    const phone = document.getElementById("phone").value;

    if (!question) {
        alert("Please enter a question.");
        return;
    }

    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ phone_number: phone, question: question }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            displayMessage(`You: ${question}`, 'user');
            displayMessage(`Bot: ${data.message}`, 'bot');
        }
    });
};

function displayMessage(message, sender) {
    const messagesDiv = document.getElementById("messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    messageDiv.textContent = message;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;  // Scroll to the bottom
}