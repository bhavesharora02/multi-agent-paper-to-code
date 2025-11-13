// Multi-Agent Code Generation - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const specificationInput = document.getElementById('specification');
    const maxIterationsInput = document.getElementById('maxIterations');
    const processBtn = document.getElementById('processBtn');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const progressMessage = document.getElementById('progressMessage');
    const iterationCount = document.getElementById('iterationCount');
    const agentName = document.getElementById('agentName');
    const currentAgent = document.getElementById('currentAgent');
    const previewBtn = document.getElementById('previewBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const codeModal = document.getElementById('codeModal');
    const modalClose = document.getElementById('modalClose');
    const codeContent = document.getElementById('codeContent');
    const modalCodeContent = document.getElementById('modalCodeContent');
    const finalIterations = document.getElementById('finalIterations');
    const ratingValue = document.getElementById('ratingValue');
    const ratingDetails = document.getElementById('ratingDetails');
    const ratingFeedback = document.getElementById('ratingFeedback');
    const feedbackText = document.getElementById('feedbackText');
    
    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Understand code chat
    const questionInput = document.getElementById('questionInput');
    const sendQuestionBtn = document.getElementById('sendQuestionBtn');
    const chatMessages = document.getElementById('chatMessages');
    const questionChips = document.querySelectorAll('.question-chip');

    let currentTaskId = null;
    let statusCheckInterval = null;

    // Process button click
    processBtn.addEventListener('click', startProcessing);

    // Modal interactions
    previewBtn.addEventListener('click', showCodePreview);
    downloadBtn.addEventListener('click', downloadCode);
    modalClose.addEventListener('click', closeModal);
    codeModal.addEventListener('click', function(e) {
        if (e.target === codeModal) closeModal();
    });

    // Tab switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Update button states
            tabButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            
            // Update content visibility
            tabContents.forEach(content => {
                content.style.display = 'none';
                content.classList.remove('active');
            });
            
            const targetTab = document.getElementById(tabName + 'Tab');
            if (targetTab) {
                targetTab.style.display = 'block';
                targetTab.classList.add('active');
            }
        });
    });
    
    // Understand code chat
    sendQuestionBtn.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });
    
    // Question chips
    questionChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            questionInput.value = question;
            questionInput.focus();
            sendQuestion();
        });
    });
    
    // Smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    function startProcessing() {
        const specification = specificationInput.value.trim();
        const maxIterations = parseInt(maxIterationsInput.value) || 10;

        if (!specification) {
            alert('Please enter a code specification');
            return;
        }

        // Disable button
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Processing...</span>';

        // Hide results, show progress
        resultsSection.style.display = 'none';
        progressSection.style.display = 'block';

        // Scroll to progress section
        progressSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Start processing
        fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                specification: specification,
                max_iterations: maxIterations
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            currentTaskId = data.task_id;
            startStatusPolling();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error starting processing: ' + error.message);
            resetUI();
        });
    }

    function startStatusPolling() {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }

        statusCheckInterval = setInterval(() => {
            checkStatus();
        }, 1000); // Check every second
    }

    function checkStatus() {
        if (!currentTaskId) return;

        fetch(`/api/status/${currentTaskId}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                updateProgress(data);

                if (data.status === 'completed' || data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    if (data.status === 'completed') {
                        showResults(data.results);
                    } else {
                        alert('Processing failed: ' + data.message);
                        resetUI();
                    }
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
                clearInterval(statusCheckInterval);
            });
    }

    function updateProgress(data) {
        const progress = data.progress || 0;
        const message = data.message || 'Processing...';
        const iteration = data.iteration || 0;
        const agent = data.current_agent || 'unknown';

        // Update progress bar
        progressFill.style.width = progress + '%';
        progressText.textContent = progress + '%';
        progressMessage.textContent = message;
        iterationCount.textContent = iteration;
        agentName.textContent = agent;
        currentAgent.textContent = agent.charAt(0).toUpperCase() + agent.slice(1) + ' Agent';
    }

    function showResults(results) {
        if (!results) return;

        // Hide progress, show results
        progressSection.style.display = 'none';
        resultsSection.style.display = 'block';

        // Update results
        const code = results.code || '';
        codeContent.textContent = code;
        modalCodeContent.textContent = code;

        // Update summary
        finalIterations.textContent = results.iteration_count || 0;

        // Update rating display
        const rating = results.code_rating || 0;
        ratingValue.textContent = rating.toFixed(1);
        ratingDetails.textContent = results.rating_details || 'Code quality analyzed';
        
        // Update rating circle color based on rating
        const ratingCircle = document.querySelector('.rating-circle');
        if (rating >= 8) {
            ratingCircle.style.background = 'linear-gradient(135deg, #10b981, #059669)';
        } else if (rating >= 6) {
            ratingCircle.style.background = 'linear-gradient(135deg, #3b82f6, #2563eb)';
        } else if (rating >= 4) {
            ratingCircle.style.background = 'linear-gradient(135deg, #f59e0b, #d97706)';
        } else {
            ratingCircle.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
        }
        
        // Show feedback if available
        if (results.rating_feedback) {
            feedbackText.textContent = results.rating_feedback;
            ratingFeedback.style.display = 'block';
        }

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        // Reset button
        resetUI();
    }
    
    function sendQuestion() {
        const question = questionInput.value.trim();
        
        if (!question) {
            return;
        }
        
        if (!currentTaskId) {
            alert('Please generate code first');
            return;
        }
        
        // Add user message
        addMessage(question, 'user');
        questionInput.value = '';
        
        // Show loading
        const loadingId = addMessage('Thinking...', 'bot', true);
        
        // Send question to API
        fetch('/api/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                task_id: currentTaskId,
                question: question
            })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) loadingMsg.remove();
            
            if (data.error) {
                addMessage('Sorry, I encountered an error: ' + data.error, 'bot');
            } else {
                addMessage(data.answer, 'bot');
            }
        })
        .catch(error => {
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) loadingMsg.remove();
            addMessage('Sorry, I encountered an error while processing your question.', 'bot');
            console.error('Error:', error);
        });
    }
    
    function addMessage(text, type, isLoading = false) {
        const messageDiv = document.createElement('div');
        const messageId = 'msg-' + Date.now();
        messageDiv.id = messageId;
        messageDiv.className = `message ${type}-message`;
        
        const p = document.createElement('p');
        p.textContent = text;
        if (isLoading) {
            p.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + text;
        }
        messageDiv.appendChild(p);
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageId;
    }

    function showCodePreview() {
        codeModal.classList.add('active');
    }

    function closeModal() {
        codeModal.classList.remove('active');
    }

    function downloadCode() {
        if (!currentTaskId) return;

        window.location.href = `/api/download/${currentTaskId}`;
    }

    function resetUI() {
        processBtn.disabled = false;
        processBtn.innerHTML = '<i class="fas fa-play"></i> <span>Generate Code</span>';
    }

    // Syntax highlighting helper (basic)
    function highlightCode(code) {
        // Basic syntax highlighting - can be enhanced with a library like Prism.js
        return code
            .replace(/(def|class|if|else|elif|for|while|return|import|from|as|try|except|finally|with|async|await)\b/g, '<span class="keyword">$1</span>')
            .replace(/(True|False|None)\b/g, '<span class="constant">$1</span>')
            .replace(/(".*?"|'.*?')/g, '<span class="string">$1</span>')
            .replace(/#.*$/gm, '<span class="comment">$&</span>');
    }
});

