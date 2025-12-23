// App Logic
document.addEventListener('DOMContentLoaded', () => {
    const timelineContainer = document.getElementById('timeline-container');
    const draftPreview = document.getElementById('current-draft');
    const summaryList = document.getElementById('summary-list');
    const actionList = document.getElementById('action-list');
    const questionList = document.getElementById('question-list');
    const sessionTimerHandler = document.querySelector('.session-timer');
    const updateBadge = document.querySelector('.update-badge');

    // Config
    const TIME_SPEED = 1; // 1x speed. Increase to 2 or 5 for faster demo.
    let startTime = Date.now();
    let virtualTime = 0; // seconds
    let activeSegmentIndex = 0;

    // State
    let processedInsightsIndices = new Set();

    // Render Helpers
    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `00:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }

    function createTimelineItem(segment) {
        const div = document.createElement('div');
        div.className = 'timeline-item';
        div.id = `segment-${segment.time}`;
        div.innerHTML = `
            <div class="timestamp">${formatTime(segment.time)}</div>
            <div class="content">
                <span class="speaker">${segment.speaker}:</span>
                <span class="text draft" id="text-${segment.time}"></span> 
            </div>
        `;
        return div;
    }

    function scrollToBottom() {
        timelineContainer.scrollTop = timelineContainer.scrollHeight;
    }

    // Main Loop
    setInterval(() => {
        virtualTime += 0.1 * TIME_SPEED; // Update every 100ms

        // 1. Update Timer
        sessionTimerHandler.innerText = formatTime(virtualTime);

        // 2. Process Transcript
        const currentSegment = MOCK_DATA.transcript[activeSegmentIndex];

        if (currentSegment) {
            // Check if we reached the start time of this segment
            if (virtualTime >= currentSegment.time) {
                // If item doesn't exist in DOM, create it
                let item = document.getElementById(`segment-${currentSegment.time}`);
                if (!item) {
                    item = createTimelineItem(currentSegment);
                    timelineContainer.appendChild(item);
                    scrollToBottom();
                }

                const textSpan = document.getElementById(`text-${currentSegment.time}`);

                // Simulate typing effect (Draft Mode)
                // Calculate progress based on time passed since segment start
                // Assume speaking rate of 5 chars per second approximately, or just fit into next segment
                const nextSegmentTime = MOCK_DATA.transcript[activeSegmentIndex + 1]?.time || (currentSegment.time + 5);
                const duration = nextSegmentTime - currentSegment.time;
                const progress = Math.min(1, (virtualTime - currentSegment.time + 0.5) / (duration - 1)); // -1 buffer

                const charCount = Math.floor(currentSegment.text.length * progress);
                const currentText = currentSegment.text.substring(0, charCount);

                if (progress < 1) {
                    textSpan.innerText = currentText;
                    draftPreview.innerText = "聆听中: " + currentText; // Update persistent draft bar
                } else {
                    // Finalize
                    textSpan.innerText = currentSegment.text;
                    textSpan.classList.remove('draft');
                    textSpan.classList.add('final');
                    draftPreview.innerText = "聆听中...";
                    activeSegmentIndex++;
                }
            }
        }

        // 3. Process Insights
        MOCK_DATA.insights.forEach((insight, index) => {
            if (!processedInsightsIndices.has(index) && virtualTime >= insight.trigger_time) {
                updateInsights(insight);
                processedInsightsIndices.add(index);
            }
        });

    }, 100);

    // Insights Logic
    function updateInsights(data) {
        // Flash the badge
        updateBadge.style.opacity = '1';
        setTimeout(() => { updateBadge.style.opacity = '0.5'; }, 2000);

        // Update Summary
        if (data.summary.length > 0) {
            summaryList.innerHTML = '';
            data.summary.forEach(text => {
                const li = document.createElement('li');
                li.innerText = text;
                li.style.animation = 'fadeIn 0.5s ease';
                summaryList.appendChild(li);
            });
        }

        // Update Actions
        if (data.actions.length > 0) {
            // Only add new ones for demo stability (in reality might replace)
            // But here we just clear and add for simplicity or append
            actionList.innerHTML = ''; // Clear for this demo version to match "snapshot" logic
            data.actions.forEach(act => {
                const div = document.createElement('div');
                div.className = 'list-item';
                div.innerHTML = `<i class="ph ph-check-circle" style="color:var(--accent-details)"></i> ${act.text}`;
                actionList.appendChild(div);
            });
        }

        // Update Questions
        if (data.questions.length > 0) {
            questionList.innerHTML = '';
            data.questions.forEach(q => {
                const div = document.createElement('div');
                div.className = 'list-item';
                div.innerHTML = `<i class="ph ph-question" style="color:var(--accent-secondary)"></i> ${q.text} <span style="opacity:0.6;font-size:0.8em">(${q.source_time || 'now'})</span>`;
                questionList.appendChild(div);
            });
        }
    }

    // QA Logic
    const qaInput = document.getElementById('qa-input');
    const qaSend = document.getElementById('qa-send-btn');
    const qaHistory = document.getElementById('qa-history');

    function sendQA() {
        const query = qaInput.value.trim();
        if (!query) return;

        // Add User Msg
        addBubble(query, 'user');
        qaInput.value = '';

        // Simulate thinking
        setTimeout(() => {
            // Find answer
            // Simple keyword matching
            let answer = MOCK_DATA.qa_database.default;
            for (const [key, val] of Object.entries(MOCK_DATA.qa_database)) {
                if (query.includes(key)) {
                    answer = val;
                    break;
                }
            }
            addBubble(answer, 'ai');
        }, 600);
    }

    function addBubble(text, type) {
        const div = document.createElement('div');
        div.className = `qa-msg ${type}`;
        div.innerHTML = text; // Allow HTML for citation
        qaHistory.appendChild(div);
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    qaSend.addEventListener('click', sendQA);
    qaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQA();
    });

    // Initial State
    draftPreview.innerText = "准备就绪, 等待语音输入...";
});
