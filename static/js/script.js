document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const processBtn = document.getElementById('processBtn');
    const progressSection = document.getElementById('progressSection');
    const resultsSection = document.getElementById('resultsSection');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const resultsSummary = document.getElementById('resultsSummary');
    const previewBtn = document.getElementById('previewBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const codeModal = document.getElementById('codeModal');
    const modalClose = document.getElementById('modalClose');
    const codePreview = document.getElementById('codePreview');

    let selectedFile = null;
    let selectedFramework = 'pytorch';
    let generatedCode = '';
    let currentSessionId = null;

    // Upload area interactions
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    fileInput.addEventListener('change', handleFileSelect);

    // Framework selection
    document.querySelectorAll('input[name="framework"]').forEach(radio => {
        radio.addEventListener('change', function() {
            selectedFramework = this.value;
            updateProcessButton();
        });
    });

    // Process button
    processBtn.addEventListener('click', processFile);

    // Modal interactions
    previewBtn.addEventListener('click', showCodePreview);
    downloadBtn.addEventListener('click', downloadCode);
    modalClose.addEventListener('click', closeModal);
    codeModal.addEventListener('click', function(e) {
        if (e.target === codeModal) closeModal();
    });

    // Understand code modal interactions
    const understandBtn = document.getElementById('understandBtn');
    const understandModal = document.getElementById('understandModal');
    const understandModalClose = document.getElementById('understandModalClose');
    const questionInput = document.getElementById('questionInput');
    const sendQuestionBtn = document.getElementById('sendQuestionBtn');
    const chatMessages = document.getElementById('chatMessages');
    const questionChips = document.querySelectorAll('.question-chip');

    // Check if understand button exists
    if (understandBtn) {
        understandBtn.addEventListener('click', showUnderstandModal);
    } else {
        console.warn('Understand button not found in DOM');
    }
    understandModalClose.addEventListener('click', closeUnderstandModal);
    understandModal.addEventListener('click', function(e) {
        if (e.target === understandModal) closeUnderstandModal();
    });
    sendQuestionBtn.addEventListener('click', sendQuestion);
    questionInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuestion();
        }
    });

    // Suggested question chips
    questionChips.forEach(chip => {
        chip.addEventListener('click', function() {
            const question = this.getAttribute('data-question');
            questionInput.value = question;
            questionInput.focus();
            sendQuestion();
        });
    });

    // Smooth scrolling for navigation links
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

    function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add('drag-over');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
    }

    function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }

    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }

    function handleFile(file) {
        if (file.type !== 'application/pdf') {
            showNotification('Please select a PDF file', 'error');
            return;
        }

        selectedFile = file;
        updateProcessButton();
        
        // Update upload area to show selected file
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-file-pdf"></i>
            </div>
            <h3>${file.name}</h3>
            <p>Ready to process</p>
        `;
        
        showNotification('PDF file selected successfully!', 'success');
    }

    function updateProcessButton() {
        processBtn.disabled = !selectedFile;
        if (selectedFile) {
            processBtn.innerHTML = `
                <i class="fas fa-magic"></i>
                Generate ${selectedFramework.charAt(0).toUpperCase() + selectedFramework.slice(1)} Code
            `;
        }
    }

    function processFile() {
        if (!selectedFile) return;

        // Hide results section
        resultsSection.style.display = 'none';
        
        // Show progress section
        progressSection.style.display = 'block';
        
        // Reset progress
        updateProgress(0, 'Starting PDF processing...');
        updateSteps(1);

        // Create FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('framework', selectedFramework);

        // Upload file
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Start polling for status
                pollStatus(data.session_id);
            } else {
                throw new Error(data.error || 'Upload failed');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error uploading file: ' + error.message, 'error');
            progressSection.style.display = 'none';
        });
    }

    function pollStatus(sessionId) {
        const pollInterval = setInterval(() => {
            fetch(`/status/${sessionId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'processing') {
                    updateProgress(data.progress || 0, data.message || 'Processing...');
                } else if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    updateProgress(100, 'Processing complete!');
                    updateSteps(4);
                    
                    // Fetch results
                    fetch(`/results/${sessionId}`)
                    .then(response => response.json())
                    .then(results => {
                        // Store session ID for understand feature
                        currentSessionId = sessionId;
                        results.session_id = sessionId; // Add session ID to results
                        showResults(results);
                    })
                    .catch(error => {
                        console.error('Error fetching results:', error);
                        showNotification('Error fetching results', 'error');
                    });
                } else if (data.status === 'error') {
                    clearInterval(pollInterval);
                    showNotification('Processing failed: ' + data.error, 'error');
                    progressSection.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('Error polling status:', error);
                clearInterval(pollInterval);
                showNotification('Error checking status', 'error');
                progressSection.style.display = 'none';
            });
        }, 500); // Poll every 500ms for more responsive updates
    }

    function updateProgress(percentage, message) {
        progressFill.style.width = percentage + '%';
        progressText.textContent = message;
        
        // Update step indicators based on progress
        let activeStep = 1;
        if (percentage >= 20 && percentage < 50) {
            activeStep = 2; // AI Analysis phase
        } else if (percentage >= 50 && percentage < 80) {
            activeStep = 3; // Code Generation phase
        } else if (percentage >= 80) {
            activeStep = 4; // Complete
        }
        updateSteps(activeStep);
    }

    function updateSteps(activeStep) {
        document.querySelectorAll('.step').forEach((step, index) => {
            step.classList.remove('active');
            if (index + 1 <= activeStep) {
                step.classList.add('active');
            }
        });
    }

    function showResults(results) {
        progressSection.style.display = 'none';
        resultsSection.style.display = 'block';
        
        generatedCode = results.code;
        
        // Store session ID for understand feature
        // Extract from the results or use the last polled session ID
        // We'll need to store it when we get results
        if (results.session_id) {
            currentSessionId = results.session_id;
        }
        
        // Display results summary
        resultsSummary.innerHTML = `
            <div class="algorithm-item">
                <div class="algorithm-name">Framework: ${results.framework}</div>
                <div class="algorithm-description">Generated ${results.algorithms.length} algorithm implementations</div>
            </div>
            ${results.algorithms.map(alg => `
                <div class="algorithm-item">
                    <div class="algorithm-name">${alg.name}</div>
                    <div class="algorithm-description">${alg.description}</div>
                </div>
            `).join('')}
        `;
        
        // Ensure understand button is visible
        const understandBtnCheck = document.getElementById('understandBtn');
        if (understandBtnCheck) {
            understandBtnCheck.style.display = 'flex';
            console.log('Understand button is visible');
        } else {
            console.error('Understand button not found!');
        }
        
        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        showNotification('Code generated successfully!', 'success');
    }

    function showCodePreview() {
        codePreview.textContent = generatedCode;
        codeModal.style.display = 'flex';
        
        // Add syntax highlighting effect
        setTimeout(() => {
            codePreview.style.opacity = '1';
        }, 100);
    }

    function closeModal() {
        codeModal.style.display = 'none';
    }

    function downloadCode() {
        const blob = new Blob([generatedCode], { type: 'text/python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `generated_code_${selectedFramework}.py`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showNotification('Code downloaded successfully!', 'success');
    }

    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#48bb78' : type === 'error' ? '#f56565' : '#4299e1'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            z-index: 3000;
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-weight: 500;
            animation: slideInRight 0.3s ease;
            max-width: 400px;
        `;
        
        document.body.appendChild(notification);
        
        // Remove after 4 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }

    // Add CSS animations for notifications
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(100%);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideOutRight {
            from {
                opacity: 1;
                transform: translateX(0);
            }
            to {
                opacity: 0;
                transform: translateX(100%);
            }
        }
        
        .drag-over {
            border-color: #667eea !important;
            background: rgba(102, 126, 234, 0.05) !important;
            transform: scale(1.02);
        }
    `;
    document.head.appendChild(style);

    // Add parallax effect to floating shapes
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const shapes = document.querySelectorAll('.shape');
        shapes.forEach((shape, index) => {
            const speed = 0.5 + (index * 0.1);
            shape.style.transform = `translateY(${scrolled * speed}px) rotate(${scrolled * 0.1}deg)`;
        });
    });

    // Add typing animation to code window
    function typeCode() {
        const codeLines = document.querySelectorAll('.code-line');
        codeLines.forEach((line, index) => {
            line.style.opacity = '0';
            setTimeout(() => {
                line.style.transition = 'opacity 0.5s ease';
                line.style.opacity = '1';
            }, index * 200);
        });
    }

    // Start typing animation when page loads
    setTimeout(typeCode, 1000);

    // Understand Code Functions
    function showUnderstandModal() {
        if (!currentSessionId) {
            showNotification('No code available to understand', 'error');
            return;
        }
        understandModal.style.display = 'flex';
        questionInput.focus();
    }

    function closeUnderstandModal() {
        understandModal.style.display = 'none';
    }

    function sendQuestion() {
        const question = questionInput.value.trim();
        if (!question) {
            showNotification('Please enter a question', 'error');
            return;
        }

        if (!currentSessionId) {
            showNotification('No code available to understand', 'error');
            return;
        }

        // Disable input and button
        questionInput.disabled = true;
        sendQuestionBtn.disabled = true;

        // Add user message to chat
        addMessageToChat(question, 'user');

        // Clear input
        questionInput.value = '';

        // Add loading message
        const loadingId = addMessageToChat('Thinking...', 'bot', true);

        // Send question to backend
        fetch(`/understand/${currentSessionId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: question })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading message
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }

            if (data.error) {
                addMessageToChat(`Sorry, I encountered an error: ${data.error}`, 'bot');
            } else {
                addMessageToChat(data.answer, 'bot');
            }

            // Re-enable input and button
            questionInput.disabled = false;
            sendQuestionBtn.disabled = false;
            questionInput.focus();
        })
        .catch(error => {
            console.error('Error asking question:', error);
            
            // Remove loading message
            const loadingMsg = document.getElementById(loadingId);
            if (loadingMsg) {
                loadingMsg.remove();
            }

            addMessageToChat('Sorry, I encountered an error while processing your question. Please try again.', 'bot');
            
            // Re-enable input and button
            questionInput.disabled = false;
            sendQuestionBtn.disabled = false;
            questionInput.focus();
        });
    }

    function addMessageToChat(message, type, isLoading = false) {
        const messageId = 'msg-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;
        messageDiv.id = messageId;

        if (type === 'user') {
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-user"></i>
                </div>
                <div class="message-content">
                    <p>${escapeHtml(message)}</p>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <p>${isLoading ? '<span class="loading-dots">' + escapeHtml(message) + '</span>' : escapeHtml(message)}</p>
                </div>
            `;
        }

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        return messageId;
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});