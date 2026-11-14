/**
 * Chat component — handles conversation history and SSE streaming.
 */

class ChatComponent {
    constructor() {
        this.currentConversationId = null;
        this.isGenerating = false;
        
        // DOM Elements
        this.messagesContainer = document.getElementById('chat-messages');
        this.inputField = document.getElementById('chat-input');
        this.submitBtn = document.getElementById('chat-submit');
        this.form = document.getElementById('chat-form');
        this.historyList = document.getElementById('chat-history-list');
        this.newChatBtn = document.getElementById('new-chat-btn');
        this.exportBtn = document.getElementById('export-chat-btn');
        
        this.bindEvents();
        this.loadConversations();
    }
    
    bindEvents() {
        this.inputField.addEventListener('input', () => {
            this.submitBtn.disabled = this.inputField.value.trim().length === 0 || this.isGenerating;
        });
        
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        this.newChatBtn.addEventListener('click', () => {
            this.startNewConversation();
        });
        
        if (this.exportBtn) {
            this.exportBtn.addEventListener('click', () => {
                this.exportConversation();
            });
        }
    }
    
    async loadConversations() {
        try {
            const conversations = await API.getConversations();
            this.historyList.innerHTML = '';
            
            if (conversations.length === 0) {
                this.historyList.innerHTML = '<li class="chat-history-item" style="opacity:0.5;pointer-events:none">No history</li>';
                return;
            }
            
            conversations.forEach(conv => {
                const li = document.createElement('li');
                li.className = `chat-history-item ${conv.id === this.currentConversationId ? 'active' : ''}`;
                li.textContent = conv.title || 'Conversation';
                li.onclick = () => this.loadConversation(conv.id);
                this.historyList.appendChild(li);
            });
        } catch (e) {
            console.error('Failed to load conversations', e);
        }
    }
    
    async startNewConversation() {
        this.currentConversationId = null;
        this.messagesContainer.innerHTML = `
            <div class="chat-message chat-message--system">
                <div class="chat-message__content">
                    Started a new conversation. What would you like to ask?
                </div>
            </div>
        `;
        this.loadConversations();
    }
    
    async loadConversation(id) {
        try {
            const conv = await API.getConversation(id);
            this.currentConversationId = id;
            this.loadConversations(); // update active state
            
            this.messagesContainer.innerHTML = '';
            conv.messages.forEach(msg => {
                this.appendMessage(msg.role, msg.content);
            });
            this.scrollToBottom();
        } catch (e) {
            console.error('Failed to load conversation details', e);
        }
    }
    
    appendMessage(role, content) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message chat-message--${role}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'chat-message__content';
        
        // Simple markdown parsing
        let html = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
            
        contentDiv.innerHTML = html;
        msgDiv.appendChild(contentDiv);
        
        this.messagesContainer.appendChild(msgDiv);
        this.scrollToBottom();
        return { msgDiv, contentDiv };
    }
    
    appendCitations(msgDiv, citations) {
        if (!citations || citations.length === 0) return;
        
        const citationsDiv = document.createElement('div');
        citationsDiv.className = 'chat-message__citations';
        
        citations.forEach(c => {
            const scorePercent = Math.round(c.score * 100);
            citationsDiv.innerHTML += `
                <div class="citation-card">
                    <div class="citation-card__header">
                        <strong>${c.document_name} (Page ${c.page_number})</strong>
                        <span>Relevance: ${scorePercent}%</span>
                    </div>
                    <div class="citation-card__text">${c.text}</div>
                </div>
            `;
        });
        
        msgDiv.appendChild(citationsDiv);
        this.scrollToBottom();
    }
    
    async sendMessage() {
        const text = this.inputField.value.trim();
        if (!text || this.isGenerating) return;
        
        // UI reset
        this.inputField.value = '';
        this.submitBtn.disabled = true;
        this.isGenerating = true;
        
        // Add user message
        this.appendMessage('user', text);
        
        // Create empty assistant message for streaming
        const { msgDiv, contentDiv } = this.appendMessage('assistant', '<span class="status-dot status-dot--loading"></span> Thinking...');
        let fullResponse = '';
        
        try {
            await API.chatStream(text, this.currentConversationId, (chunk, citations, data) => {
                // Keep the conversation the server created, so follow-up questions share its history
                if (data && data.conversation_id && !this.currentConversationId) {
                    this.currentConversationId = data.conversation_id;
                }
                if (fullResponse === '') {
                    contentDiv.innerHTML = ''; // Clear loading
                }
                
                if (chunk) {
                    fullResponse += chunk;
                    // Re-render full markdown
                    contentDiv.innerHTML = fullResponse
                        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                        .replace(/\*(.*?)\*/g, '<em>$1</em>')
                        .replace(/\n/g, '<br>');
                }
                
                if (citations && citations.length > 0) {
                    this.appendCitations(msgDiv, citations);
                }
                
                this.scrollToBottom();
            });
            
            // Reload history to get new ID if it was created
            await this.loadConversations();
            
        } catch (e) {
            console.error('Chat error:', e);
            contentDiv.innerHTML = `<span style="color:var(--danger)">Error: ${e.message}</span>`;
        } finally {
            this.isGenerating = false;
            this.inputField.focus();
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    exportConversation() {
        if (!this.currentConversationId) {
            App.toast('No conversation selected', 'error');
            return;
        }
        App.toast('Starting export...', 'info');
        window.location.href = `/api/export/chat/${this.currentConversationId}/txt`;
    }
}
