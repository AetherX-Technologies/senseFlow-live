// Live ASR App - WebSocket Client with LLM Integration
// Connects to ws_server.py for real-time transcription and AI insights

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const timelineContainer = document.getElementById('timeline-container');
    const draftPreview = document.getElementById('current-draft');
    const summaryList = document.getElementById('summary-list');
    const summaryLiveList = document.getElementById('summary-live-list');
    const actionList = document.getElementById('action-list');
    const questionList = document.getElementById('question-list');
    const sessionTimer = document.querySelector('.session-timer');
    const statusPill = document.querySelector('.status-pill');
    const updateBadge = document.querySelector('.update-badge');

    // Config
    const WS_URL = 'ws://localhost:8766';

    // State
    let ws = null;
    let sessionStartTime = Date.now();
    let segments = new Map(); // segment_id -> {element, text, isFinal}
    let currentSegmentId = null;
    let waitingForAnswer = false;

    // ============ Timer ============
    function updateTimer() {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const h = Math.floor(elapsed / 3600);
        const m = Math.floor((elapsed % 3600) / 60);
        const s = elapsed % 60;
        sessionTimer.textContent = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    }
    setInterval(updateTimer, 1000);

    // ============ UI Helpers ============
    function formatTimestamp(ts) {
        const date = new Date(ts * 1000);
        return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function setStatus(status) {
        statusPill.className = 'status-pill ' + status;
        const label = statusPill.querySelector('span:last-child');
        switch(status) {
            case 'recording':
                label.textContent = '正在录音';
                break;
            case 'connecting':
                label.textContent = '连接中...';
                break;
            case 'disconnected':
                label.textContent = '已断开';
                break;
            case 'listening':
                label.textContent = '聆听中';
                break;
        }
    }

    function createTimelineItem(segmentId, timestamp) {
        const div = document.createElement('div');
        div.className = 'timeline-item';
        div.id = `segment-${segmentId}`;

        const tsDiv = document.createElement('div');
        tsDiv.className = 'timestamp';
        tsDiv.textContent = formatTimestamp(timestamp);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'content';

        const textSpan = document.createElement('span');
        textSpan.className = 'text draft';
        textSpan.id = `text-${segmentId}`;

        contentDiv.appendChild(textSpan);
        div.appendChild(tsDiv);
        div.appendChild(contentDiv);

        return div;
    }

    function scrollToBottom() {
        timelineContainer.scrollTop = timelineContainer.scrollHeight;
    }

    // ============ WebSocket ============
    function connect() {
        setStatus('connecting');
        draftPreview.textContent = '连接服务器中...';

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[WS] Connected');
            setStatus('recording');
            draftPreview.textContent = '正在聆听...';
            sessionStartTime = Date.now();
        };

        ws.onclose = () => {
            console.log('[WS] Disconnected');
            setStatus('disconnected');
            draftPreview.textContent = '连接已断开，3秒后重连...';
            setTimeout(connect, 3000);
        };

        ws.onerror = (err) => {
            console.error('[WS] Error:', err);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleEvent(data);
            } catch (e) {
                console.error('[WS] Parse error:', e);
            }
        };
    }

    // ============ Event Handlers ============
    function handleEvent(event) {
        const { type, payload, segment_id, ts } = event;

        switch (type) {
            case 'connection.established':
                console.log('[ASR] Session:', payload.session_id);
                break;

            case 'engine.ready':
                console.log('[ASR] Engine ready', payload);
                break;

            case 'vad.speech.start':
                handleSpeechStart(segment_id, ts);
                break;

            case 'vad.speech.end':
                handleSpeechEnd(segment_id);
                break;

            case 'asr.partial':
                handlePartial(segment_id, payload, ts);
                break;

            case 'asr.final':
                handleFinal(segment_id, payload);
                break;

            case 'insights.update':
                handleInsightsUpdate(payload);
                break;

            case 'qa.answer':
                handleQAAnswer(payload);
                break;

            case 'engine.error':
                console.error('[ASR] Error:', payload.error);
                break;
        }
    }

    function handleSpeechStart(segmentId, ts) {
        currentSegmentId = segmentId;
        setStatus('recording');

        // Create new timeline item
        const item = createTimelineItem(segmentId, ts);
        timelineContainer.appendChild(item);
        scrollToBottom();

        segments.set(segmentId, {
            element: item,
            text: '',
            isFinal: false
        });

        console.log(`[VAD] Speech started: ${segmentId}`);
    }

    function handleSpeechEnd(segmentId) {
        if (currentSegmentId === segmentId) {
            currentSegmentId = null;
            setStatus('listening');
            draftPreview.textContent = '正在聆听...';
        }
        console.log(`[VAD] Speech ended: ${segmentId}`);
    }

    function handlePartial(segmentId, payload, ts) {
        const text = payload.text || '';
        const isStreaming = payload.streaming === true || segmentId === 'live';

        if (isStreaming) {
            draftPreview.textContent = text || '正在聆听...';
            return;
        }

        let segment = segments.get(segmentId);
        if (!segment) {
            // Create if doesn't exist
            const item = createTimelineItem(segmentId, ts);
            timelineContainer.appendChild(item);
            segment = { element: item, text: '', isFinal: false };
            segments.set(segmentId, segment);
        }

        // Update text
        segment.text = text;
        const textSpan = document.getElementById(`text-${segmentId}`);
        if (textSpan) {
            textSpan.textContent = text;
            textSpan.className = 'text draft';
        }

        // Update draft preview
        draftPreview.textContent = text || '正在聆听...';
        scrollToBottom();
    }

    function handleFinal(segmentId, payload) {
        const text = payload.text || '';
        const duration = payload.duration_ms || 0;

        const segment = segments.get(segmentId);
        if (segment) {
            segment.text = text;
            segment.isFinal = true;

            const textSpan = document.getElementById(`text-${segmentId}`);
            if (textSpan) {
                textSpan.textContent = text;
                textSpan.className = 'text final';
            }
        }

        console.log(`[ASR] Final (${duration}ms): ${text}`);
    }

    // ============ Insights Handler ============
    function handleInsightsUpdate(payload) {
        console.log('[LLM] Insights update:', payload);

        // Flash update badge
        updateBadge.textContent = '刚刚更新';
        updateBadge.style.opacity = '1';
        setTimeout(() => { updateBadge.style.opacity = '0.7'; }, 2000);

        // Update overall summary
        if (payload.summary && payload.summary.length > 0) {
            summaryList.textContent = '';
            payload.summary.forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                li.style.animation = 'fadeIn 0.5s ease';
                summaryList.appendChild(li);
            });
        }

        // Update live summary (incremental)
        if (payload.summary_live && payload.summary_live.length > 0) {
            summaryLiveList.textContent = '';
            payload.summary_live.forEach(text => {
                const li = document.createElement('li');
                li.textContent = text;
                li.style.animation = 'fadeIn 0.5s ease';
                summaryLiveList.appendChild(li);
            });
        }

        // Update actions
        if (payload.actions && payload.actions.length > 0) {
            actionList.textContent = '';
            payload.actions.forEach(action => {
                const div = document.createElement('div');
                div.className = 'list-item';

                const icon = document.createElement('i');
                icon.className = 'ph ph-check-circle';
                icon.style.color = 'var(--accent-details)';

                const textNode = document.createTextNode(' ' + action.text);

                div.appendChild(icon);
                div.appendChild(textNode);
                div.style.animation = 'fadeIn 0.5s ease';
                actionList.appendChild(div);
            });
        }

        // Update questions
        if (payload.questions && payload.questions.length > 0) {
            questionList.textContent = '';
            payload.questions.forEach(q => {
                const div = document.createElement('div');
                div.className = 'list-item';

                const icon = document.createElement('i');
                icon.className = 'ph ph-question';
                icon.style.color = 'var(--accent-secondary)';

                const textNode = document.createTextNode(' ' + q.text);

                div.appendChild(icon);
                div.appendChild(textNode);
                div.style.animation = 'fadeIn 0.5s ease';
                questionList.appendChild(div);
            });
        }
    }

    // ============ QA Handler ============
    function handleQAAnswer(payload) {
        waitingForAnswer = false;
        const answer = payload.answer || '无法获取回答';

        // Remove typing indicator
        const typingBubble = qaHistory.querySelector('.qa-msg.typing');
        if (typingBubble) {
            typingBubble.remove();
        }

        addBubble(answer, 'ai');
        console.log('[QA] Answer received:', answer);
    }

    // ============ QA UI ============
    const qaInput = document.getElementById('qa-input');
    const qaSend = document.getElementById('qa-send-btn');
    const qaHistory = document.getElementById('qa-history');

    function sendQA() {
        const query = qaInput.value.trim();
        if (!query || waitingForAnswer) return;

        addBubble(query, 'user');
        qaInput.value = '';

        // Send to server
        if (ws && ws.readyState === WebSocket.OPEN) {
            waitingForAnswer = true;
            ws.send(JSON.stringify({
                command: 'ask',
                question: query
            }));
            // Show typing indicator
            addBubble('正在思考...', 'ai typing');
        } else {
            addBubble('未连接到服务器', 'ai');
        }
    }

    function addBubble(text, type) {
        const div = document.createElement('div');
        div.className = `qa-msg ${type}`;
        div.textContent = text;
        qaHistory.appendChild(div);
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    qaSend.addEventListener('click', sendQA);
    qaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQA();
    });

    // ============ Init ============
    function initPlaceholders() {
        const summaryPlaceholder = document.createElement('li');
        summaryPlaceholder.style.opacity = '0.6';
        summaryPlaceholder.textContent = '等待语音输入...';
        summaryList.textContent = '';
        summaryList.appendChild(summaryPlaceholder);

        const summaryLivePlaceholder = document.createElement('li');
        summaryLivePlaceholder.style.opacity = '0.6';
        summaryLivePlaceholder.textContent = '等待新增内容...';
        summaryLiveList.textContent = '';
        summaryLiveList.appendChild(summaryLivePlaceholder);

        const actionPlaceholder = document.createElement('div');
        actionPlaceholder.className = 'list-item';
        actionPlaceholder.style.opacity = '0.6';
        const actionIcon = document.createElement('i');
        actionIcon.className = 'ph ph-hourglass';
        actionPlaceholder.appendChild(actionIcon);
        actionPlaceholder.appendChild(document.createTextNode(' 暂无待办'));
        actionList.textContent = '';
        actionList.appendChild(actionPlaceholder);

        const questionPlaceholder = document.createElement('div');
        questionPlaceholder.className = 'list-item';
        questionPlaceholder.style.opacity = '0.6';
        const questionIcon = document.createElement('i');
        questionIcon.className = 'ph ph-circle-dashed';
        questionPlaceholder.appendChild(questionIcon);
        questionPlaceholder.appendChild(document.createTextNode(' 暂无问题'));
        questionList.textContent = '';
        questionList.appendChild(questionPlaceholder);
    }

    initPlaceholders();

    // Connect to WebSocket server
    connect();
});
