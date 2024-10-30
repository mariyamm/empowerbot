document.getElementById("start").onclick = function () {
    const phone = document.getElementById("phone").value;
    if (!phone) {
        alert("Please enter your phone number.");
        return;
    }

    fetch('/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phone_number: phone }),
        
    })
    .then(response => {
        
        if (!response.ok) {
            
            throw new Error(`Server responded with status: ${response.status}`);
        }
       
        return response.json();
        
    })
    .then(data => {
        console.log("Response from /start:", data);
        
        if (data.new_user) {  // if the user is new
            isCollectingInitialData = true;  // Set to true for the initial data flow
            displayMessage(data.message, 'bot');
            document.getElementById("question-section").style.display = "flex";
            document.getElementById("phone").style.display = "none";
            document.getElementById("start").style.display = "none";
        } else {
            // If it's a returning user, just start regular chat
            displayMessage(data.message, 'bot');
            isCollectingInitialData = false;
            document.getElementById("question-section").style.display = "flex";
            document.getElementById("phone").style.display = "none";
            document.getElementById("start").style.display = "none";
        }
    })
    .catch(error => {
        console.error("Error in /start fetch call:", error);
        alert("An error occurred while starting the chat. Check console for details.");
    });
};



let currentQuestionIndex = 0;
let initialResponses = {};
let isCollectingInitialData = false;
let initialQuestions = [
    "What's your name?",
    "What are your goals?",
    "What's your focus area?"
];


//TO DO: Add a data verication step to ensure that the user has entered the correct data

//TO DO Improve so that if there is no approval, it will ask again
// Function to ask the next question
function askInitialQuestions(userMessage) {
    console.log("askInitialQuestions called. currentQuestionIndex:", currentQuestionIndex);

    // Collect the user's response if it's not the first call
    if (currentQuestionIndex >= 0) {
        initialResponses[currentQuestionIndex] = userMessage;

        displayMessage(`${userMessage}`, 'user');

        //check if the user have their approval
        if (currentQuestionIndex == 0) {
            if (userMessage.toLowerCase() != "yes") {
                displayMessage("Please enter 'yes' as a sign or approval. Otherwise, I will not be able to help you.", 'bot');
                return;
            }
        }
    }

    // Check if there are more questions to ask
    if (currentQuestionIndex < initialQuestions.length) {
        const question = initialQuestions[currentQuestionIndex];
        displayMessage(`${question}`, 'bot');
        currentQuestionIndex++;
    } else {
        // All questions have been asked, collect responses
        const responses = {
            phone_number: document.getElementById("phone").value,
            approval: initialResponses[0] || '',
            name: initialResponses[1] || '',
            goals: initialResponses[2] || '',
            focus_area: initialResponses[3] || ''
        };

        console.log("All questions answered. Collecting responses:", responses);

        // Send responses to the server
        fetch('/collect_initial_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(responses)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayMessage(data.message, 'bot');
            // Reset for future use
            resetInitialData();
            isCollectingInitialData = false;
            document.getElementById("question-section").style.display = "flex";
        })
        .catch(error => {
            console.error("Error in /collect_initial_data fetch call:", error);
            alert("An error occurred while collecting initial data. Check console for details.");
        });
    }
}


// Reset function to clear data for future use
function resetInitialData() {
    currentQuestionIndex = 0;
    initialResponses = {};
    isCollectingInitialData = false;
}

// Handle "Send" button click
document.getElementById("chat").onclick = function () {
    const userMessage = document.getElementById("question").value;
    const phone = document.getElementById("phone").value;

    if (!userMessage) {
        alert("Please enter a message.");
        return;
    }
    console.log("Send button clicked. isCollectingInitialData:", isCollectingInitialData);
    if (isCollectingInitialData) {
        // Handle user input for initial questions
        //displayMessage(`You: ${userMessage}`, 'user');
        askInitialQuestions(userMessage);
        document.getElementById("question").value = ''; // Clear input for next question

    } else {
        // Send a regular chat message
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ phone_number: phone, question: userMessage }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                // Display user and bot messages in chat
                displayMessage(`${userMessage}`, 'user');
                displayMessage(`${data.message}`, 'bot');
                
                // Clear the question input after sending
                document.getElementById("question").value = '';
            }
        });
    }
};

// Display messages in the chatbox
function displayMessage(message, sender) {
    const messagesDiv = document.getElementById("messages");
    const messageDiv = document.createElement("div");
    messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
    messageDiv.textContent = message;
    messagesDiv.appendChild(messageDiv);

    // Auto-scroll to the bottom for new messages
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
