// app/static/js/chat-widget.js
// OwnBot Web Chat Widget - No Email Collection
// Version: 1.0.0

(function() {
    'use strict';
    
    // Configuration
    const config = {
        apiBaseUrl: window.ownBotConfig?.apiBaseUrl || 'https://ownbot.chat/api',
        clientId: window.ownBotConfig?.clientId || null,
        widgetPosition: window.ownBotConfig?.position || 'bottom-right',
        primaryColor: window.ownBotConfig?.primaryColor || '#2563eb',
        secondaryColor: window.ownBotConfig?.secondaryColor || '#ffffff',
        greetingMessage: window.ownBotConfig?.greetingMessage || 'Hello! How can I help you today?',
        autoOpen: window.ownBotConfig?.autoOpen !== undefined ? window.ownBotConfig.autoOpen : false,
        autoOpenDelay: window.ownBotConfig?.autoOpenDelay || 5000
    };
    
    // State management
    let state = {
        isOpen: false,
        isMinimized: false,
        isTyping: false,
        conversationId: null,
        messages: [],
        hasInteracted: false
    };
    
    // DOM Elements
    let widgetContainer, chatButton, chatWindow, messageContainer, inputField, sendButton;
    
    // Initialize the widget
    function init() {
        if (!config.clientId) {
            console.error('OwnBot: Client ID is required. Please set window.ownBotConfig.clientId');
            return;
        }
        
        createWidget();
        bindEvents();
        
        if (config.autoOpen && !state.hasInteracted) {
            setTimeout(() => {
                openChat();
            }, config.autoOpenDelay);
        }
        
        // Load previous conversation if available
        loadConversationFromStorage();
    }
    
    // Create the widget DOM elements
    function createWidget() {
        // Create main container
        widgetContainer = document.createElement('div');
        widgetContainer.id = 'ownbot-widget';
        widgetContainer.className = `ownbot-widget ownbot-${config.widgetPosition}`;
        
        // Create chat button
        chatButton = document.createElement('div');
        chatButton.className = 'ownbot-chat-button';
        chatButton.innerHTML = `
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z" fill="${config.secondaryColor}"/>
            </svg>
        `;
        
        // Create chat window
        chatWindow = document.createElement('div');
        chatWindow.className = 'ownbot-chat-window';
        chatWindow.innerHTML = `
            <div class="ownbot-header">
                <div class="ownbot-title">Chat with us</div>
                <div class="ownbot-actions">
                    <button class="ownbot-minimize-btn">−</button>
                    <button class="ownbot-close-btn">×</button>
                </div>
            </div>
            <div class="ownbot-messages"></div>
            <div class="ownbot-typing-indicator" style="display: none;">
                <div class="ownbot-typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                <span>AI is typing...</span>
            </div>
            <div class="ownbot-input-container">
                <input type="text" class="ownbot-input" placeholder="Type your message..." />
                <button class="ownbot-send-btn">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" fill="${config.secondaryColor}"/>
                    </svg>
                </button>
            </div>
        `;
        
        // Append elements to container
        widgetContainer.appendChild(chatButton);
        widgetContainer.appendChild(chatWindow);
        document.body.appendChild(widgetContainer);
        
        // Cache DOM elements
        messageContainer = chatWindow.querySelector('.ownbot-messages');
        inputField = chatWindow.querySelector('.ownbot-input');
        sendButton = chatWindow.querySelector('.ownbot-send-btn');
        
        // Apply styling
        applyStyles();
    }
    
    // Apply CSS styles dynamically
    function applyStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .ownbot-widget {
                position: fixed;
                z-index: 9999;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            }
            
            .ownbot-bottom-right {
                bottom: 20px;
                right: 20px;
            }
            
            .ownbot-bottom-left {
                bottom: 20px;
                left: 20px;
            }
            
            .ownbot-chat-button {
                width: 60px;
                height: 60px;
                background-color: ${config.primaryColor};
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                transition: all 0.3s ease;
            }
            
            .ownbot-chat-button:hover {
                transform: scale(1.05);
                box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
            }
            
            .ownbot-chat-window {
                position: absolute;
                width: 350px;
                height: 500px;
                background-color: ${config.secondaryColor};
                border-radius: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                display: flex;
                flex-direction: column;
                overflow: hidden;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.3s ease;
                bottom: 70px;
                right: 0;
            }
            
            .ownbot-widget.ownbot-bottom-left .ownbot-chat-window {
                right: auto;
                left: 0;
            }
            
            .ownbot-chat-window.ownbot-open {
                opacity: 1;
                transform: translateY(0);
            }
            
            .ownbot-chat-window.ownbot-minimized {
                height: 45px;
            }
            
            .ownbot-header {
                background-color: ${config.primaryColor};
                color: ${config.secondaryColor};
                padding: 16px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .ownbot-title {
                font-weight: 600;
                font-size: 16px;
            }
            
            .ownbot-actions {
                display: flex;
                gap: 8px;
            }
            
            .ownbot-minimize-btn, .ownbot-close-btn {
                background: none;
                border: none;
                color: ${config.secondaryColor};
                font-size: 18px;
                cursor: pointer;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 4px;
                transition: background-color 0.2s;
            }
            
            .ownbot-minimize-btn:hover, .ownbot-close-btn:hover {
                background-color: rgba(255, 255, 255, 0.2);
            }
            
            .ownbot-messages {
                flex: 1;
                padding: 16px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 12px;
            }
            
            .ownbot-message {
                max-width: 80%;
                padding: 10px 14px;
                border-radius: 18px;
                line-height: 1.4;
                word-wrap: break-word;
            }
            
            .ownbot-message-bot {
                align-self: flex-start;
                background-color: #f1f5f9;
                color: #334155;
                border-bottom-left-radius: 4px;
            }
            
            .ownbot-message-user {
                align-self: flex-end;
                background-color: ${config.primaryColor};
                color: ${config.secondaryColor};
                border-bottom-right-radius: 4px;
            }
            
            .ownbot-typing-indicator {
                padding: 8px 16px;
                display: flex;
                align-items: center;
                gap: 8px;
                color: #64748b;
                font-size: 14px;
            }
            
            .ownbot-typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .ownbot-typing-dots span {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background-color: #94a3b8;
                animation: ownbot-typing 1.4s infinite ease-in-out;
            }
            
            .ownbot-typing-dots span:nth-child(1) {
                animation-delay: -0.32s;
            }
            
            .ownbot-typing-dots span:nth-child(2) {
                animation-delay: -0.16s;
            }
            
            @keyframes ownbot-typing {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            .ownbot-input-container {
                display: flex;
                padding: 12px;
                border-top: 1px solid #e2e8f0;
                gap: 8px;
            }
            
            .ownbot-input {
                flex: 1;
                padding: 12px;
                border: 1px solid #e2e8f0;
                border-radius: 24px;
                outline: none;
                font-size: 14px;
                transition: border-color 0.2s;
            }
            
            .ownbot-input:focus {
                border-color: ${config.primaryColor};
            }
            
            .ownbot-send-btn {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: ${config.primaryColor};
                border: none;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .ownbot-send-btn:hover {
                background-color: ${config.primaryColor}dd;
            }
            
            .ownbot-send-btn:disabled {
                background-color: #cbd5e1;
                cursor: not-allowed;
            }
            
            @media (max-width: 480px) {
                .ownbot-chat-window {
                    width: 100%;
                    height: 100%;
                    bottom: 0;
                    right: 0;
                    border-radius: 0;
                }
                
                .ownbot-widget.ownbot-bottom-left .ownbot-chat-window,
                .ownbot-widget.ownbot-bottom-right .ownbot-chat-window {
                    bottom: 0;
                    right: 0;
                    left: 0;
                }
            }
        `;
        
        document.head.appendChild(style);
    }
    
    // Bind event listeners
    function bindEvents() {
        // Chat button click
        chatButton.addEventListener('click', openChat);
        
        // Close button click
        chatWindow.querySelector('.ownbot-close-btn').addEventListener('click', closeChat);
        
        // Minimize button click
        chatWindow.querySelector('.ownbot-minimize-btn').addEventListener('click', toggleMinimize);
        
        // Send button click
        sendButton.addEventListener('click', sendMessage);
        
        // Input field enter key
        inputField.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Input field focus to track interaction
        inputField.addEventListener('focus', function() {
            state.hasInteracted = true;
        });
    }
    
    // Open chat window
    function openChat() {
        state.isOpen = true;
        state.isMinimized = false;
        chatWindow.classList.add('ownbot-open');
        chatWindow.classList.remove('ownbot-minimized');
        inputField.focus();
        
        // Add greeting message if no messages yet
        if (state.messages.length === 0) {
            addMessage(config.greetingMessage, 'bot');
        }
    }
    
    // Close chat window
    function closeChat() {
        state.isOpen = false;
        chatWindow.classList.remove('ownbot-open');
    }
    
    // Toggle minimize
    function toggleMinimize() {
        state.isMinimized = !state.isMinimized;
        chatWindow.classList.toggle('ownbot-minimized');
    }
    
    // Add message to chat
    function addMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.className = `ownbot-message ownbot-message-${sender}`;
        messageElement.textContent = text;
        
        messageContainer.appendChild(messageElement);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        
        // Add to state
        state.messages.push({
            text,
            sender,
            timestamp: new Date().toISOString()
        });
        
        // Save to storage
        saveConversationToStorage();
    }
    
    // Show typing indicator
    function showTypingIndicator() {
        state.isTyping = true;
        const typingIndicator = chatWindow.querySelector('.ownbot-typing-indicator');
        typingIndicator.style.display = 'flex';
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
    
    // Hide typing indicator
    function hideTypingIndicator() {
        state.isTyping = false;
        const typingIndicator = chatWindow.querySelector('.ownbot-typing-indicator');
        typingIndicator.style.display = 'none';
    }
    
    // Send message to backend
    async function sendMessage() {
        const message = inputField.value.trim();
        
        if (!message) return;
        
        // Clear input
        inputField.value = '';
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Show typing indicator
        showTypingIndicator();
        
        try {
            // Prepare request payload
            const payload = {
                message,
                conversationId: state.conversationId,
                clientId: config.clientId
            };
            
            // Send to backend
            const response = await fetch(`${config.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Update conversation ID if received
            if (data.conversationId) {
                state.conversationId = data.conversationId;
            }
            
            // Add bot response to chat
            if (data.response) {
                addMessage(data.response, 'bot');
            } else {
                addMessage('Sorry, I encountered an error processing your request.', 'bot');
            }
        } catch (error) {
            console.error('OwnBot: Error sending message:', error);
            hideTypingIndicator();
            addMessage('Sorry, I am having trouble connecting to the server. Please try again later.', 'bot');
        }
    }
    
    // Save conversation to localStorage
    function saveConversationToStorage() {
        try {
            const data = {
                messages: state.messages,
                conversationId: state.conversationId,
                timestamp: new Date().toISOString()
            };
            
            localStorage.setItem(`ownbot-conversation-${config.clientId}`, JSON.stringify(data));
        } catch (error) {
            console.error('OwnBot: Error saving conversation:', error);
        }
    }
    
    // Load conversation from localStorage
    function loadConversationFromStorage() {
        try {
            const data = localStorage.getItem(`ownbot-conversation-${config.clientId}`);
            
            if (data) {
                const parsed = JSON.parse(data);
                state.messages = parsed.messages || [];
                state.conversationId = parsed.conversationId || null;
                
                // Render loaded messages
                if (state.messages.length > 0) {
                    state.messages.forEach(msg => {
                        addMessage(msg.text, msg.sender);
                    });
                }
            }
        } catch (error) {
            console.error('OwnBot: Error loading conversation:', error);
        }
    }
    
    // Clear conversation from storage
    function clearConversationFromStorage() {
        try {
            localStorage.removeItem(`ownbot-conversation-${config.clientId}`);
            state.messages = [];
            state.conversationId = null;
            messageContainer.innerHTML = '';
        } catch (error) {
            console.error('OwnBot: Error clearing conversation:', error);
        }
    }
    
    // Public API methods
    window.OwnBot = {
        open: openChat,
        close: closeChat,
        toggle: function() {
            if (state.isOpen) {
                closeChat();
            } else {
                openChat();
            }
        },
        sendMessage: function(message) {
            inputField.value = message;
            sendMessage();
        },
        clearHistory: clearConversationFromStorage,
        updateConfig: function(newConfig) {
            Object.assign(config, newConfig);
            // Reapply styles if colors changed
            if (newConfig.primaryColor || newConfig.secondaryColor) {
                applyStyles();
            }
        }
    };
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
