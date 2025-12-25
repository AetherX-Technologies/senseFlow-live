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
    const downloadNotesBtn = document.getElementById('download-notes-btn');
    const sessionTimer = document.querySelector('.session-timer');
    const statusPill = document.querySelector('.status-pill');
    const updateBadge = document.querySelector('.update-badge');
    const summaryLiveCard = document.getElementById('summary-live-card');
    const historyList = document.getElementById('history-list');
    const historyCount = document.getElementById('history-count');
    const newSessionBtn = document.getElementById('new-session-btn');
    const pauseBtn = document.getElementById('pause-btn');
    const connectionBanner = document.getElementById('connection-banner');
    const qaInput = document.getElementById('qa-input');
    const qaSend = document.getElementById('qa-send-btn');
    const qaHistory = document.getElementById('qa-history');
    const settingsBtn = document.getElementById('settings-btn');
    const settingsDrawer = document.getElementById('settings-drawer');
    const settingsOverlay = document.getElementById('settings-overlay');
    const settingsCloseBtn = document.getElementById('settings-close-btn');
    const settingsStatus = document.getElementById('settings-status');
    const settingMicSelect = document.getElementById('setting-mic');
    const settingGain = document.getElementById('setting-gain');
    const settingGainValue = document.getElementById('setting-gain-value');
    const settingNoiseGate = document.getElementById('setting-noise-gate');
    const settingNoiseGateValue = document.getElementById('setting-noise-gate-value');
    const settingVadSens = document.getElementById('setting-vad-sens');
    const settingVadSensValue = document.getElementById('setting-vad-sens-value');
    const settingPunctuation = document.getElementById('setting-punctuation');
    const settingMergeStrategy = document.getElementById('setting-merge-strategy');
    const settingModelMode = document.getElementById('setting-model-mode');
    const settingSummaryInterval = document.getElementById('setting-summary-interval');
    const settingSummaryLive = document.getElementById('setting-summary-live');
    const settingLlmEnabled = document.getElementById('setting-llm-enabled');
    const settingLlmModel = document.getElementById('setting-llm-model');
    const settingAutoScroll = document.getElementById('setting-auto-scroll');
    const settingShowTimestamps = document.getElementById('setting-show-timestamps');
    const settingExportFormat = document.getElementById('setting-export-format');
    const settingMongoEnabled = document.getElementById('setting-mongo-enabled');
    const settingRetentionDays = document.getElementById('setting-retention-days');
    const settingMaxSize = document.getElementById('setting-max-size');
    const settingAutoCleanup = document.getElementById('setting-auto-cleanup');

    // Config
    const WS_URL = 'ws://127.0.0.1:8766';
    const HISTORY_LIMIT = 50;

    // State
    let ws = null;
    let sessionStartTime = Date.now();
    const sessions = new Map();
    const historyMeta = new Map();
    let historyOrder = [];
    let liveSessionId = null;
    let activeSessionId = null;
    let currentSegmentId = null;
    let waitingForAnswer = false;
    let pendingQAForSessionId = null;
    let pendingNewSession = false;
    let serverMics = [];
    let audioPaused = false;
    let autoScrollSuspendUntil = 0;
    let autoScrollTimer = null;
    let autoScrollInProgress = false;

    const SETTINGS_KEY = 'senseflow.settings';
    const defaultSettings = {
        audio: {
            micId: '',
            gain: 1.0,
            noiseGate: 0.0,
            vadSensitivity: 0.5,
        },
        transcription: {
            punctuation: true,
            mergeStrategy: 'silence',
            modelMode: 'realtime',
        },
        summary: {
            intervalSec: 180,
            liveSummary: true,
            llmEnabled: true,
            llmModel: 'claude-haiku-4-5-20251001',
        },
        display: {
            autoScroll: true,
            showTimestamps: true,
            exportFormat: 'markdown',
        },
        storage: {
            mongoEnabled: true,
            retentionDays: 30,
            maxSizeMb: 2048,
            autoCleanup: true,
        },
    };
    let settings = loadSettings();
    let settingsUpdateTimer = null;
    let pendingSettingsSync = false;

    // ============ Timer ============
    function formatRelative(tsMs) {
        if (!tsMs) return '';
        const diffSec = Math.max(0, Math.floor((Date.now() - tsMs) / 1000));
        if (diffSec < 10) return '刚刚';
        if (diffSec < 60) return `${diffSec}s 前`;
        const m = Math.floor(diffSec / 60);
        if (m < 60) return `${m} 分钟前`;
        const h = Math.floor(m / 60);
        if (h < 24) return `${h} 小时前`;
        const d = Math.floor(h / 24);
        return `${d} 天前`;
    }

    function updateTimer() {
        const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
        const h = Math.floor(elapsed / 3600);
        const m = Math.floor((elapsed % 3600) / 60);
        const s = elapsed % 60;
        if (sessionTimer) {
            sessionTimer.textContent = `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
        }
        updateUpdateBadge();
    }
    setInterval(updateTimer, 1000);

    // ============ UI Helpers ============
    function updateUpdateBadge() {
        if (!updateBadge) return;
        const session = getActiveSession();
        if (!session || !session.lastUpdateTs) {
            updateBadge.textContent = '等待更新';
            updateBadge.style.opacity = '0.7';
            return;
        }
        const rel = formatRelative(session.lastUpdateTs);
        updateBadge.textContent = `上次更新：${rel}`;
        updateBadge.style.opacity = '0.9';
    }

    function formatTimestamp(ts) {
        if (!ts) return '';
        const date = new Date(ts * 1000);
        return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    function formatHistoryTime(ts) {
        if (!ts) return '未知';
        const date = new Date(ts * 1000);
        const local = date.toLocaleString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        return `${local} (${formatRelative(date.getTime())})`;
    }

    function formatClock(tsMs) {
        if (!tsMs) return '';
        return new Date(tsMs).toLocaleTimeString('zh-CN', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }

    function mergeHistoryMeta(existing, incoming) {
        if (!incoming) return existing || {};
        const merged = Object.assign({}, existing || {}, incoming);
        ['started_at', 'last_active', 'terminated_at'].forEach((key) => {
            if ((incoming[key] === null || incoming[key] === undefined) && existing && existing[key] !== undefined) {
                merged[key] = existing[key];
            }
        });
        if (incoming.terminated === undefined && existing && existing.terminated !== undefined) {
            merged.terminated = existing.terminated;
        }
        const incomingSort = Math.max(
            Number(incoming.sort_ts || 0),
            Number(incoming.last_active || 0),
            Number(incoming.terminated_at || 0),
            Number(incoming.started_at || 0)
        );
        const existingSort = Number(existing?.sort_ts || 0);
        if (incomingSort || existingSort) {
            merged.sort_ts = Math.max(existingSort, incomingSort);
        }
        return merged;
    }

    function getSessionSortTs(sessionId) {
        const meta = historyMeta.get(sessionId) || {};
        if (meta.sort_ts) {
            return Number(meta.sort_ts);
        }
        const lastActive = Number(meta.last_active || 0);
        const terminatedAt = Number(meta.terminated_at || 0);
        const startedAt = Number(meta.started_at || 0);
        let ts = Math.max(lastActive, terminatedAt, startedAt);
        if (!ts && sessionId === liveSessionId) {
            ts = Date.now() / 1000;
        }
        return ts;
    }

    function setSessionSortTs(sessionId, ts) {
        if (!sessionId) return;
        const meta = historyMeta.get(sessionId) || { session_id: sessionId };
        const next = Number(ts || 0);
        if (next > Number(meta.sort_ts || 0)) {
            meta.sort_ts = next;
            historyMeta.set(sessionId, meta);
        }
    }

    function showConnectionBanner(show) {
        if (!connectionBanner) return;
        connectionBanner.classList.toggle('show', show);
    }

    function setStatus(status) {
        if (!statusPill) return;
        statusPill.className = 'status-pill ' + status;
        const label = statusPill.querySelector('span:last-child');
        if (!label) return;
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
            case 'paused':
                label.textContent = '已暂停';
                break;
            case 'terminated':
                label.textContent = '已终止';
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

    function ensureTimelineItem(segmentId, timestamp) {
        let item = document.getElementById(`segment-${segmentId}`);
        if (!item) {
            item = createTimelineItem(segmentId, timestamp);
            timelineContainer.appendChild(item);
        }
        return item;
    }

    function scrollToBottom() {
        if (!settings.display.autoScroll) return;
        if (Date.now() < autoScrollSuspendUntil) return;
        autoScrollInProgress = true;
        timelineContainer.scrollTop = timelineContainer.scrollHeight;
        requestAnimationFrame(() => {
            autoScrollInProgress = false;
        });
    }

    function escapeHtml(text) {
        return String(text || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function renderMarkdown(text) {
        const content = String(text || '');
        if (window.marked && window.DOMPurify) {
            const raw = window.marked.parse(content, { breaks: true });
            return window.DOMPurify.sanitize(raw, { USE_PROFILES: { html: true } });
        }
        return escapeHtml(content).replace(/\n/g, '<br>');
    }

    function listToMarkdown(items, renderItem) {
        if (!items || items.length === 0) {
            return ['- 暂无内容'];
        }
        return items.map(renderItem);
    }

    function mergeSettings(base, patch) {
        if (!patch || typeof patch !== 'object') return base;
        const result = Array.isArray(base) ? [...base] : { ...base };
        Object.keys(patch).forEach(key => {
            if (patch[key] && typeof patch[key] === 'object' && !Array.isArray(patch[key])) {
                result[key] = mergeSettings(base[key] || {}, patch[key]);
            } else {
                result[key] = patch[key];
            }
        });
        return result;
    }

    function loadSettings() {
        try {
            const raw = localStorage.getItem(SETTINGS_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                return mergeSettings(defaultSettings, parsed);
            }
        } catch (e) {
            console.warn('Settings load failed:', e);
        }
        return JSON.parse(JSON.stringify(defaultSettings));
    }

    function saveSettings() {
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch (e) {
            console.warn('Settings save failed:', e);
        }
    }

    function applyDisplaySettings() {
        document.body.classList.toggle('hide-timestamps', !settings.display.showTimestamps);
        if (summaryLiveCard) {
            summaryLiveCard.style.display = settings.summary.liveSummary ? '' : 'none';
        }
    }

    function applySettings() {
        applyDisplaySettings();
        updateQAAvailability();
        setUpdateBadgeDefault();
    }

    function setRangeDisplay(labelEl, value, formatter) {
        if (!labelEl) return;
        labelEl.textContent = formatter(value);
    }

    function setSettingsStatus(text, state) {
        if (!settingsStatus) return;
        settingsStatus.textContent = text;
        settingsStatus.className = `drawer-status${state ? ' ' + state : ''}`;
    }

    function queueSettingsUpdate() {
        pendingSettingsSync = true;
        setSettingsStatus('待同步', 'warn');
        if (settingsUpdateTimer) {
            clearTimeout(settingsUpdateTimer);
        }
        settingsUpdateTimer = setTimeout(() => {
            sendSettingsUpdate();
        }, 400);
    }

    function sendSettingsUpdate() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            return;
        }
        ws.send(JSON.stringify({
            command: 'settings.update',
            settings: settings,
        }));
        pendingSettingsSync = false;
        setSettingsStatus('同步中', '');
    }

    async function loadMicrophones() {
        if (!settingMicSelect || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
            if (serverMics.length === 0) {
                return;
            }
        }
        try {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ command: 'audio.devices' }));
            }
            if (serverMics.length > 0) {
                updateMicOptions(serverMics, settings.audio.micId);
                return;
            }
            const devices = await navigator.mediaDevices.enumerateDevices();
            const mics = devices.filter(device => device.kind === 'audioinput');
            settingMicSelect.textContent = '';
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = '系统默认';
            settingMicSelect.appendChild(defaultOption);
            mics.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = device.label || device.deviceId || `麦克风 ${index + 1}`;
                option.textContent = device.label || `麦克风 ${index + 1}`;
                settingMicSelect.appendChild(option);
            });
            settingMicSelect.value = settings.audio.micId || '';
        } catch (e) {
            console.warn('Microphone list failed:', e);
        }
    }

    function updateMicOptions(devices, selectedId) {
        if (!settingMicSelect) return;
        settingMicSelect.textContent = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = '系统默认';
        settingMicSelect.appendChild(defaultOption);
        devices.forEach(device => {
            const option = document.createElement('option');
            option.value = device.id;
            option.textContent = device.name || `麦克风 ${device.id}`;
            settingMicSelect.appendChild(option);
        });
        if (selectedId) {
            settingMicSelect.value = selectedId;
        } else {
            settingMicSelect.value = '';
        }
    }

    // ============ Session State ============
    function createSessionState(sessionId) {
        return {
            sessionId,
            segments: new Map(),
            segmentOrder: [],
            finalSegments: new Map(),
            finalSegmentOrder: [],
            lastInsights: { summary: [], summary_live: [], actions: [], questions: [] },
            qaLog: [],
            draftPreview: '',
            lastUpdateTs: null,
            loaded: false,
            terminated: false,
        };
    }

    function ensureSession(sessionId) {
        if (!sessionId) return null;
        if (!sessions.has(sessionId)) {
            sessions.set(sessionId, createSessionState(sessionId));
        }
        return sessions.get(sessionId);
    }

    function getActiveSession() {
        return activeSessionId ? ensureSession(activeSessionId) : null;
    }

    function isSessionTerminated(sessionId) {
        if (!sessionId) return false;
        const meta = historyMeta.get(sessionId);
        return Boolean(meta && meta.terminated);
    }

    function markSessionActivity(sessionId, tsSeconds) {
        if (!sessionId) return;
        const meta = historyMeta.get(sessionId) || { session_id: sessionId };
        const nowTs = tsSeconds || (Date.now() / 1000);
        if (!meta.started_at) {
            meta.started_at = nowTs;
        }
        meta.last_active = nowTs;
        meta.sort_ts = Math.max(Number(meta.sort_ts || 0), nowTs);
        historyMeta.set(sessionId, meta);
    }

    function setActiveSession(sessionId, options = {}) {
        activeSessionId = sessionId;
        if (sessionId) {
            ensureSession(sessionId);
        }
        renderActiveSession();
        setUpdateBadgeDefault();
        renderHistoryList();
        updateQAAvailability();
        applyActiveSessionTermination();
        if (options.resetTimer) {
            sessionStartTime = Date.now();
        }
    }

    function touchHistorySession(sessionId, overrides = {}) {
        if (!sessionId) return;
        const now = Date.now() / 1000;
        const meta = historyMeta.get(sessionId) || { session_id: sessionId };
        if (!meta.started_at) {
            meta.started_at = now;
        }
        meta.last_active = meta.last_active || now;
        meta.sort_ts = Math.max(Number(meta.sort_ts || 0), now);
        Object.assign(meta, overrides);
        historyMeta.set(sessionId, meta);
        if (!historyOrder.includes(sessionId)) {
            historyOrder.unshift(sessionId);
        }
    }

    function renderActiveSession() {
        const session = getActiveSession();
        if (!session) {
            renderEmptyState();
            return;
        }
        renderTimeline(session);
        renderInsights(session);
        renderQA(session);
        const isLive = activeSessionId === liveSessionId;
        const fallbackDraft = isLive ? '正在聆听...' : '历史会话浏览中';
        draftPreview.textContent = session.draftPreview || fallbackDraft;
    }

    function renderEmptyState() {
        timelineContainer.textContent = '';
        renderSummaryList(summaryList, [], '等待语音输入...');
        renderSummaryList(summaryLiveList, [], '等待新增内容...');
        renderItemList(actionList, [], '暂无待办', 'ph ph-hourglass', 'var(--text-muted)');
        renderItemList(questionList, [], '暂无问题', 'ph ph-circle-dashed', 'var(--text-muted)');
        qaHistory.textContent = '';
        appendWelcomeMessage();
        draftPreview.textContent = '连接服务器中...';
    }

    function renderTimeline(session) {
        timelineContainer.textContent = '';
        const order = session.segmentOrder.length > 0
            ? session.segmentOrder
            : Array.from(session.segments.keys());
        order.forEach(segmentId => {
            const segment = session.segments.get(segmentId);
            if (!segment) return;
            const item = createTimelineItem(segmentId, segment.ts || Date.now() / 1000);
            const textSpan = item.querySelector('.text');
            textSpan.textContent = segment.text || '';
            textSpan.className = `text ${segment.isFinal ? 'final' : 'draft'}`;
            timelineContainer.appendChild(item);
        });
        scrollToBottom();
    }

    function renderSummaryList(container, items, placeholderText) {
        container.textContent = '';
        if (!items || items.length === 0) {
            const li = document.createElement('li');
            li.style.opacity = '0.6';
            li.textContent = placeholderText;
            container.appendChild(li);
            return;
        }
        items.forEach(text => {
            const li = document.createElement('li');
            li.textContent = text;
            li.style.animation = 'fadeIn 0.5s ease';
            container.appendChild(li);
        });
    }

    function renderItemList(container, items, placeholderText, iconClass, iconColor) {
        container.textContent = '';
        if (!items || items.length === 0) {
            const div = document.createElement('div');
            div.className = 'list-item';
            div.style.opacity = '0.6';
            const icon = document.createElement('i');
            icon.className = iconClass;
            if (iconColor) {
                icon.style.color = iconColor;
            }
            div.appendChild(icon);
            div.appendChild(document.createTextNode(' ' + placeholderText));
            container.appendChild(div);
            return;
        }
        items.forEach(item => {
            const div = document.createElement('div');
            div.className = 'list-item';

            const icon = document.createElement('i');
            icon.className = iconClass;
            if (iconColor) {
                icon.style.color = iconColor;
            }

            const textValue = typeof item === 'string' ? item : (item.text || '');
            const textNode = document.createTextNode(' ' + textValue);
            div.appendChild(icon);
            div.appendChild(textNode);
            div.style.animation = 'fadeIn 0.5s ease';
            container.appendChild(div);
        });
    }

    function renderInsights(session) {
        if (!session) return;
        renderSummaryList(summaryList, session.lastInsights.summary || [], '等待语音输入...');
        renderSummaryList(summaryLiveList, session.lastInsights.summary_live || [], '等待新增内容...');
        renderItemList(actionList, session.lastInsights.actions || [], '暂无待办', 'ph ph-check-circle', 'var(--accent-details)');
        renderItemList(questionList, session.lastInsights.questions || [], '暂无问题', 'ph ph-question', 'var(--accent-secondary)');
        updateUpdateBadge();
    }

    function setUpdateBadgeDefault() {
        if (!updateBadge) return;
        updateBadge.textContent = activeSessionId === liveSessionId ? '等待中' : '历史会话';
        updateBadge.style.opacity = '0.7';
    }

    function applyActiveSessionTermination() {
        if (!activeSessionId || !liveSessionId) {
            setPauseButtonDisabled(false);
            return;
        }
        const liveTerminated = isSessionTerminated(liveSessionId);
        if (liveTerminated && activeSessionId === liveSessionId) {
            setStatus('terminated');
            setPauseButtonDisabled(true);
            if (draftPreview) {
                draftPreview.textContent = '会话已终止';
            }
            return;
        }
        setPauseButtonDisabled(false);
    }

    function setPauseButtonDisabled(disabled) {
        if (!pauseBtn) return;
        pauseBtn.disabled = disabled;
        pauseBtn.classList.toggle('disabled', disabled);
    }

    function setPauseButtonState(paused) {
        if (!pauseBtn) return;
        const icon = pauseBtn.querySelector('i');
        const label = pauseBtn.querySelector('span');
        pauseBtn.classList.toggle('paused', paused);
        if (icon) {
            icon.className = paused ? 'ph ph-play-circle' : 'ph ph-pause-circle';
        }
        if (label) {
            label.textContent = paused ? '继续' : '暂停';
        }
        pauseBtn.title = paused ? '继续录音' : '暂停录音';
    }

    function openSettingsDrawer() {
        if (!settingsDrawer || !settingsOverlay) return;
        settingsDrawer.classList.add('open');
        settingsDrawer.setAttribute('aria-hidden', 'false');
        settingsOverlay.classList.add('show');
        loadMicrophones();
    }

    function closeSettingsDrawer() {
        if (!settingsDrawer || !settingsOverlay) return;
        settingsDrawer.classList.remove('open');
        settingsDrawer.setAttribute('aria-hidden', 'true');
        settingsOverlay.classList.remove('show');
    }

    function bindSettingsControls() {
        if (settingMicSelect) {
            settingMicSelect.value = settings.audio.micId || '';
            settingMicSelect.addEventListener('change', () => {
                settings.audio.micId = settingMicSelect.value;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingGain) {
            settingGain.value = settings.audio.gain;
            setRangeDisplay(settingGainValue, settings.audio.gain, value => `${Number(value).toFixed(2)}x`);
            settingGain.addEventListener('input', () => {
                settings.audio.gain = Number(settingGain.value);
                setRangeDisplay(settingGainValue, settings.audio.gain, value => `${Number(value).toFixed(2)}x`);
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingNoiseGate) {
            settingNoiseGate.value = settings.audio.noiseGate;
            setRangeDisplay(settingNoiseGateValue, settings.audio.noiseGate, value => Number(value).toFixed(2));
            settingNoiseGate.addEventListener('input', () => {
                settings.audio.noiseGate = Number(settingNoiseGate.value);
                setRangeDisplay(settingNoiseGateValue, settings.audio.noiseGate, value => Number(value).toFixed(2));
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingVadSens) {
            settingVadSens.value = settings.audio.vadSensitivity;
            setRangeDisplay(settingVadSensValue, settings.audio.vadSensitivity, value => Number(value).toFixed(2));
            settingVadSens.addEventListener('input', () => {
                settings.audio.vadSensitivity = Number(settingVadSens.value);
                setRangeDisplay(settingVadSensValue, settings.audio.vadSensitivity, value => Number(value).toFixed(2));
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingPunctuation) {
            settingPunctuation.checked = settings.transcription.punctuation;
            settingPunctuation.addEventListener('change', () => {
                settings.transcription.punctuation = settingPunctuation.checked;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingMergeStrategy) {
            settingMergeStrategy.value = settings.transcription.mergeStrategy;
            settingMergeStrategy.addEventListener('change', () => {
                settings.transcription.mergeStrategy = settingMergeStrategy.value;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingModelMode) {
            settingModelMode.value = settings.transcription.modelMode;
            settingModelMode.addEventListener('change', () => {
                settings.transcription.modelMode = settingModelMode.value;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingSummaryInterval) {
            settingSummaryInterval.value = String(settings.summary.intervalSec);
            settingSummaryInterval.addEventListener('change', () => {
                settings.summary.intervalSec = Number(settingSummaryInterval.value);
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingSummaryLive) {
            settingSummaryLive.checked = settings.summary.liveSummary;
            settingSummaryLive.addEventListener('change', () => {
                settings.summary.liveSummary = settingSummaryLive.checked;
                saveSettings();
                queueSettingsUpdate();
                applyDisplaySettings();
            });
        }

        if (settingLlmEnabled) {
            settingLlmEnabled.checked = settings.summary.llmEnabled;
            settingLlmEnabled.addEventListener('change', () => {
                settings.summary.llmEnabled = settingLlmEnabled.checked;
                saveSettings();
                queueSettingsUpdate();
                applySettings();
            });
        }

        if (settingLlmModel) {
            settingLlmModel.value = settings.summary.llmModel;
            settingLlmModel.addEventListener('change', () => {
                settings.summary.llmModel = settingLlmModel.value;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingAutoScroll) {
            settingAutoScroll.checked = settings.display.autoScroll;
            settingAutoScroll.addEventListener('change', () => {
                settings.display.autoScroll = settingAutoScroll.checked;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingShowTimestamps) {
            settingShowTimestamps.checked = settings.display.showTimestamps;
            settingShowTimestamps.addEventListener('change', () => {
                settings.display.showTimestamps = settingShowTimestamps.checked;
                saveSettings();
                queueSettingsUpdate();
                applyDisplaySettings();
            });
        }

        if (settingExportFormat) {
            settingExportFormat.value = settings.display.exportFormat;
            settingExportFormat.addEventListener('change', () => {
                settings.display.exportFormat = settingExportFormat.value;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingMongoEnabled) {
            settingMongoEnabled.checked = settings.storage.mongoEnabled;
            settingMongoEnabled.addEventListener('change', () => {
                settings.storage.mongoEnabled = settingMongoEnabled.checked;
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingRetentionDays) {
            settingRetentionDays.value = settings.storage.retentionDays;
            settingRetentionDays.addEventListener('input', () => {
                settings.storage.retentionDays = Math.max(0, Number(settingRetentionDays.value));
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingMaxSize) {
            settingMaxSize.value = settings.storage.maxSizeMb;
            settingMaxSize.addEventListener('input', () => {
                settings.storage.maxSizeMb = Math.max(0, Number(settingMaxSize.value));
                saveSettings();
                queueSettingsUpdate();
            });
        }

        if (settingAutoCleanup) {
            settingAutoCleanup.checked = settings.storage.autoCleanup;
            settingAutoCleanup.addEventListener('change', () => {
                settings.storage.autoCleanup = settingAutoCleanup.checked;
                saveSettings();
                queueSettingsUpdate();
            });
        }
    }

    function initSettingsDrawer() {
        if (settingsBtn) {
            settingsBtn.addEventListener('click', openSettingsDrawer);
        }
        if (settingsCloseBtn) {
            settingsCloseBtn.addEventListener('click', closeSettingsDrawer);
        }
        if (settingsOverlay) {
            settingsOverlay.addEventListener('click', closeSettingsDrawer);
        }
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeSettingsDrawer();
            }
        });
        bindSettingsControls();
    }

    function appendWelcomeMessage() {
        const welcome = document.createElement('div');
        welcome.className = 'qa-msg ai-welcome';
        welcome.textContent = '嗨！我是你的实时会议助手。连接成功后，你可以随时问我关于会议内容的问题。';
        qaHistory.appendChild(welcome);
    }

    function renderQA(session) {
        qaHistory.textContent = '';
        if (!session || session.qaLog.length === 0) {
            appendWelcomeMessage();
            return;
        }
        session.qaLog.forEach(entry => {
            const div = document.createElement('div');
            div.className = `qa-msg ${entry.role === 'user' ? 'user' : 'ai'}`;
            if (entry.role === 'ai') {
                div.innerHTML = renderMarkdown(entry.text);
            } else {
                div.textContent = entry.text;
            }
            qaHistory.appendChild(div);
        });
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    function updateQAAvailability() {
        const hasSession = Boolean(activeSessionId);
        const isLive = hasSession && activeSessionId === liveSessionId;
        const isTerminated = hasSession && isSessionTerminated(activeSessionId);
        const llmEnabled = settings.summary.llmEnabled;
        const enabled = hasSession && llmEnabled;
        qaInput.disabled = !enabled;
        qaSend.disabled = !enabled;
        if (!llmEnabled) {
            qaInput.placeholder = 'LLM 已关闭';
        } else if (!hasSession) {
            qaInput.placeholder = '请选择会话后提问';
        } else if (isLive) {
            qaInput.placeholder = isTerminated ? '向 AI 提问（已终止会话）' : '向 AI 提问...';
        } else {
            qaInput.placeholder = '向 AI 提问（历史会话）';
        }
    }

    function appendQABubble(text, type) {
        const div = document.createElement('div');
        div.className = `qa-msg ${type}`;
        if (type.includes('ai')) {
            div.innerHTML = renderMarkdown(text);
        } else {
            div.textContent = text;
        }
        qaHistory.appendChild(div);
        qaHistory.scrollTop = qaHistory.scrollHeight;
    }

    function removeTypingIndicator() {
        const typingBubble = qaHistory.querySelector('.qa-msg.typing');
        if (typingBubble) {
            typingBubble.remove();
        }
    }

    // ============ Export ============
    function buildExportPayload() {
        const now = new Date();
        const session = getActiveSession();
        const sessionId = activeSessionId || '';
        const transcript = [];
        const finalOrder = session ? session.finalSegmentOrder : [];
        const finalSegments = session ? session.finalSegments : new Map();
        if (finalOrder && finalOrder.length > 0) {
            finalOrder.forEach(segmentId => {
                const segment = finalSegments.get(segmentId);
                if (!segment || !segment.text) return;
                transcript.push({
                    segment_id: segmentId,
                    ts: segment.ts || null,
                    text: segment.text,
                });
            });
        }
        return {
            exported_at: now.toISOString(),
            session_id: sessionId,
            transcript,
            insights: session ? session.lastInsights : { summary: [], summary_live: [], actions: [], questions: [] },
            qa: session ? session.qaLog : [],
        };
    }

    function buildMarkdownExport() {
        const now = new Date();
        const payload = buildExportPayload();
        const lines = [];
        lines.push(`# Session Export (${now.toLocaleString('zh-CN')})`);
        if (payload.session_id) {
            lines.push(`- Session ID: ${payload.session_id}`);
        }
        lines.push('');
        lines.push('## 实时转写');

        if (!payload.transcript || payload.transcript.length === 0) {
            lines.push('- 暂无内容');
        } else {
            payload.transcript.forEach(item => {
                const timestamp = item.ts ? formatTimestamp(item.ts) : '';
                const line = `${timestamp} ${item.text || ''}`.trim();
                lines.push(`- ${line}`);
            });
        }

        lines.push('');
        lines.push('## 智能整理');
        lines.push('### 整体摘要');
        lines.push(...listToMarkdown(payload.insights.summary, text => `- ${text}`));
        lines.push('');
        lines.push('### 实时摘要');
        lines.push(...listToMarkdown(payload.insights.summary_live, text => `- ${text}`));
        lines.push('');
        lines.push('### 待办事项');
        lines.push(...listToMarkdown(
            payload.insights.actions,
            action => `- ${typeof action === 'string' ? action : (action.text || '')}`
        ));
        lines.push('');
        lines.push('### 悬而未决');
        lines.push(...listToMarkdown(
            payload.insights.questions,
            q => `- ${typeof q === 'string' ? q : (q.text || '')}`
        ));

        lines.push('');
        lines.push('## 问答记录');
        if (!payload.qa || payload.qa.length === 0) {
            lines.push('- 暂无内容');
        } else {
            payload.qa.forEach(entry => {
                const clock = formatClock(entry.ts);
                const roleLabel = entry.role === 'user' ? '用户' : 'AI';
                const prefix = clock ? `[${clock}] ${roleLabel}:` : `${roleLabel}:`;
                lines.push(`- ${prefix} ${entry.text}`);
            });
        }

        return lines.join('\n');
    }

    function buildTextExport() {
        const now = new Date();
        const payload = buildExportPayload();
        const lines = [];
        lines.push(`Session Export (${now.toLocaleString('zh-CN')})`);
        if (payload.session_id) {
            lines.push(`Session ID: ${payload.session_id}`);
        }
        lines.push('');
        lines.push('[实时转写]');
        if (!payload.transcript || payload.transcript.length === 0) {
            lines.push('暂无内容');
        } else {
            payload.transcript.forEach(item => {
                const timestamp = item.ts ? formatTimestamp(item.ts) : '';
                lines.push(`${timestamp} ${item.text || ''}`.trim());
            });
        }
        lines.push('');
        lines.push('[智能整理]');
        lines.push('整体摘要:');
        lines.push(...(payload.insights.summary || []).map(text => `- ${text}`));
        lines.push('实时摘要:');
        lines.push(...(payload.insights.summary_live || []).map(text => `- ${text}`));
        lines.push('待办事项:');
        lines.push(...(payload.insights.actions || []).map(item => `- ${typeof item === 'string' ? item : (item.text || '')}`));
        lines.push('悬而未决:');
        lines.push(...(payload.insights.questions || []).map(item => `- ${typeof item === 'string' ? item : (item.text || '')}`));
        lines.push('');
        lines.push('[问答记录]');
        if (!payload.qa || payload.qa.length === 0) {
            lines.push('暂无内容');
        } else {
            payload.qa.forEach(entry => {
                const clock = formatClock(entry.ts);
                const roleLabel = entry.role === 'user' ? '用户' : 'AI';
                const prefix = clock ? `[${clock}] ${roleLabel}:` : `${roleLabel}:`;
                lines.push(`${prefix} ${entry.text}`);
            });
        }
        return lines.join('\n');
    }

    function buildExportContent() {
        const stamp = new Date().toISOString().replace(/[:T]/g, '-').slice(0, 19);
        switch (settings.display.exportFormat) {
            case 'json':
                return {
                    content: JSON.stringify(buildExportPayload(), null, 2),
                    filename: `session_export_${stamp}.json`,
                    mime: 'application/json;charset=utf-8',
                };
            case 'text':
                return {
                    content: buildTextExport(),
                    filename: `session_export_${stamp}.txt`,
                    mime: 'text/plain;charset=utf-8',
                };
            case 'markdown':
            default:
                return {
                    content: buildMarkdownExport(),
                    filename: `session_export_${stamp}.md`,
                    mime: 'text/markdown;charset=utf-8',
                };
        }
    }

    function downloadExport() {
        const { content, filename, mime } = buildExportContent();
        const blob = new Blob([content], { type: mime });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    function initDownloads() {
        if (!downloadNotesBtn) return;
        downloadNotesBtn.addEventListener('click', downloadExport);
    }

    function suspendAutoScroll() {
        autoScrollSuspendUntil = Date.now() + 5000;
        if (autoScrollTimer) {
            clearTimeout(autoScrollTimer);
        }
        autoScrollTimer = setTimeout(() => {
            autoScrollSuspendUntil = 0;
        }, 5000);
    }

    function initTimelineAutoScrollGuard() {
        if (!timelineContainer) return;
        timelineContainer.addEventListener('scroll', () => {
            if (autoScrollInProgress) return;
            suspendAutoScroll();
        });
        timelineContainer.addEventListener('wheel', suspendAutoScroll, { passive: true });
        timelineContainer.addEventListener('touchstart', suspendAutoScroll, { passive: true });
        timelineContainer.addEventListener('touchmove', suspendAutoScroll, { passive: true });
    }

    // ============ History UI ============
    function renderHistoryList() {
        if (!historyList || !historyCount) return;
        historyList.textContent = '';
        const orderedIds = historyOrder.slice().sort((a, b) => {
            const aMeta = historyMeta.get(a) || {};
            const bMeta = historyMeta.get(b) || {};
            const aRank = (a === liveSessionId && !aMeta.terminated) ? 0 : 1;
            const bRank = (b === liveSessionId && !bMeta.terminated) ? 0 : 1;
            if (aRank !== bRank) return aRank - bRank;
            const diff = getSessionSortTs(b) - getSessionSortTs(a);
            if (diff !== 0) return diff;
            return String(a).localeCompare(String(b));
        });
        historyCount.textContent = orderedIds.length.toString();

        if (orderedIds.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'history-empty';
            empty.textContent = '暂无历史记录';
            historyList.appendChild(empty);
            return;
        }

        orderedIds.forEach(sessionId => {
            const meta = historyMeta.get(sessionId) || {};
            const isLive = sessionId === liveSessionId;
            const isTerminated = Boolean(meta.terminated);
            const item = document.createElement('button');
            item.type = 'button';
            item.className = 'history-item';
            item.dataset.sessionId = sessionId;

            if (sessionId === activeSessionId) {
                item.classList.add('active');
            }
            if (isLive && !isTerminated) {
                item.classList.add('live');
            }
            if (isTerminated) {
                item.classList.add('terminated');
            }

            const title = document.createElement('div');
            title.className = 'history-title';
            title.textContent = `会话 ${sessionId}`;

            const badges = document.createElement('div');
            badges.className = 'history-badges';
            if (isLive && !isTerminated) {
                const liveBadge = document.createElement('span');
                liveBadge.className = 'history-badge';
                liveBadge.textContent = 'LIVE';
                badges.appendChild(liveBadge);
            }
            if (isTerminated) {
                const endBadge = document.createElement('span');
                endBadge.className = 'history-badge end';
                endBadge.textContent = '已终止';
                badges.appendChild(endBadge);
            }
            if (isLive && !isTerminated) {
                const endBtn = document.createElement('button');
                endBtn.type = 'button';
                endBtn.className = 'history-action';
                endBtn.innerHTML = '<i class="ph ph-stop-circle"></i>终止';
                endBtn.addEventListener('click', (event) => {
                    event.stopPropagation();
                    requestSessionTerminate(sessionId);
                });
                badges.appendChild(endBtn);
            }

            const metaLine = document.createElement('div');
            metaLine.className = 'history-meta';
            const startedText = meta.started_at ? `开始 ${formatHistoryTime(meta.started_at)}` : '开始未知';
            const lastText = isTerminated
                ? (meta.terminated_at ? `终止 ${formatHistoryTime(meta.terminated_at)}` : '已终止')
                : (meta.last_active
                    ? `最近 ${formatHistoryTime(meta.last_active)}`
                    : (isLive ? '进行中' : '暂无更新'));
            metaLine.textContent = `${startedText} · ${lastText}`;

            const headerRow = document.createElement('div');
            headerRow.style.display = 'flex';
            headerRow.style.alignItems = 'center';
            headerRow.style.gap = '8px';
            headerRow.appendChild(title);
            if (badges.childNodes.length > 0) {
                headerRow.appendChild(badges);
            }

            item.appendChild(headerRow);
            item.appendChild(metaLine);
            historyList.appendChild(item);
        });
    }

    function requestHistoryList() {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({
            command: 'history.list',
            limit: HISTORY_LIMIT,
        }));
    }

    function requestHistoryLoad(sessionId) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({
            command: 'history.load',
            session_id: sessionId,
        }));
    }

    function requestSessionTerminate(sessionId) {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        ws.send(JSON.stringify({
            command: 'session.terminate',
            session_id: sessionId,
        }));
    }

    function initHistoryActions() {
        if (historyList) {
            historyList.addEventListener('click', (event) => {
                const item = event.target.closest('.history-item');
                if (!item) return;
                const sessionId = item.dataset.sessionId;
                if (!sessionId) return;
                setActiveSession(sessionId);
                const session = ensureSession(sessionId);
                if (!session.loaded) {
                    requestHistoryLoad(sessionId);
                }
            });
        }

        if (newSessionBtn) {
            newSessionBtn.addEventListener('click', () => {
                if (!ws || ws.readyState !== WebSocket.OPEN) return;
                pendingNewSession = true;
                ws.send(JSON.stringify({ command: 'reset' }));
            });
        }

        if (pauseBtn) {
            pauseBtn.addEventListener('click', () => {
                if (!ws || ws.readyState !== WebSocket.OPEN) return;
                const nextState = !audioPaused;
                ws.send(JSON.stringify({
                    command: 'audio.pause',
                    paused: nextState,
                }));
            });
        }
    }

    // ============ WebSocket ============
    function connect() {
        setStatus('connecting');
        draftPreview.textContent = '连接服务器中...';
        showConnectionBanner(false);
        setSettingsStatus('未连接', 'error');

        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[WS] Connected');
            setStatus('recording');
            draftPreview.textContent = '正在聆听...';
            sessionStartTime = Date.now();
            showConnectionBanner(false);
            setSettingsStatus('已连接', 'ok');
            sendSettingsUpdate();
            ws.send(JSON.stringify({ command: 'audio.devices' }));
        };

        ws.onclose = () => {
            console.log('[WS] Disconnected');
            setStatus('disconnected');
            draftPreview.textContent = '连接已断开，3秒后重连...';
            showConnectionBanner(true);
            setSettingsStatus('未连接', 'error');
            setTimeout(connect, 3000);
        };

        ws.onerror = (err) => {
            console.error('[WS] Error:', err);
            showConnectionBanner(true);
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
        const { type, payload, segment_id, ts, session_id } = event;

        switch (type) {
            case 'connection.established':
                liveSessionId = payload.session_id;
                ensureSession(liveSessionId);
                touchHistorySession(liveSessionId, { source: 'live' });
                if (!activeSessionId) {
                    setActiveSession(liveSessionId, { resetTimer: true });
                }
                requestHistoryList();
                requestHistoryLoad(liveSessionId);
                console.log('[ASR] Session:', payload.session_id);
                break;

            case 'engine.ready':
                console.log('[ASR] Engine ready', payload);
                break;

            case 'session.changed':
                handleSessionChanged(payload);
                break;

            case 'engine.reset':
                handleEngineReset(payload);
                break;

            case 'session.terminated':
                handleSessionTerminated(payload);
                break;

            case 'history.list':
                handleHistoryList(payload);
                break;

            case 'history.session':
                handleHistorySession(payload);
                break;

            case 'audio.devices':
                handleAudioDevices(payload);
                break;

            case 'audio.paused':
                handleAudioPaused(payload);
                break;

            case 'settings.applied':
                handleSettingsApplied(payload);
                break;

            case 'vad.speech.start':
                handleSpeechStart(session_id, segment_id, ts);
                break;

            case 'vad.speech.end':
                handleSpeechEnd(session_id, segment_id);
                break;

            case 'asr.partial':
                handlePartial(session_id, segment_id, payload, ts);
                break;

            case 'asr.final':
                handleFinal(session_id, segment_id, payload, ts);
                break;

            case 'insights.update':
                handleInsightsUpdate(session_id, payload);
                break;

            case 'qa.answer':
                handleQAAnswer(session_id, payload);
                break;

            case 'engine.error':
                console.error('[ASR] Error:', payload.error);
                break;
        }
    }

    function handleSessionChanged(payload) {
        const newSessionId = payload.session_id;
        const previousSessionId = payload.previous_session_id;
        liveSessionId = newSessionId;
        ensureSession(newSessionId);
        touchHistorySession(newSessionId, { source: 'live', terminated: false });

        if (!activeSessionId || activeSessionId === previousSessionId) {
            setActiveSession(newSessionId, { resetTimer: true });
        }
        pendingNewSession = false;
        requestHistoryList();
        requestHistoryLoad(newSessionId);
        applyActiveSessionTermination();
    }

    function handleEngineReset(payload) {
        if (payload && payload.session_id) {
            liveSessionId = payload.session_id;
            touchHistorySession(payload.session_id, { source: 'live', terminated: false });
            if (pendingNewSession || activeSessionId === payload.session_id) {
                setActiveSession(payload.session_id, { resetTimer: true });
            }
            pendingNewSession = false;
            applyActiveSessionTermination();
        }
    }

    function handleSessionTerminated(payload) {
        const sessionId = payload?.session_id;
        if (!sessionId) return;
        const terminatedAt = payload.terminated_at ? Number(payload.terminated_at) : null;
        const meta = historyMeta.get(sessionId) || { session_id: sessionId };
        meta.terminated = true;
        if (terminatedAt) {
            meta.terminated_at = terminatedAt;
        }
        meta.sort_ts = Math.max(Number(meta.sort_ts || 0), terminatedAt || (Date.now() / 1000));
        historyMeta.set(sessionId, meta);
        const session = ensureSession(sessionId);
        if (session) {
            session.terminated = true;
        }
        renderHistoryList();
        if (sessionId === activeSessionId) {
            applyActiveSessionTermination();
            renderActiveSession();
        } else {
            updateQAAvailability();
        }
    }

    function handleHistoryList(payload) {
        const sessionsList = payload.sessions || [];
        const previousMeta = new Map(historyMeta);
        historyMeta.clear();
        historyOrder = [];

        sessionsList.forEach(item => {
            if (!item.session_id) return;
            const merged = mergeHistoryMeta(previousMeta.get(item.session_id), item);
            if (item.session_id === payload.live_session_id && !merged.started_at) {
                merged.started_at = Date.now() / 1000;
            }
            if (item.session_id === payload.live_session_id && !merged.terminated) {
                merged.sort_ts = Math.max(Number(merged.sort_ts || 0), Date.now() / 1000);
            }
            historyMeta.set(item.session_id, merged);
            historyOrder.push(item.session_id);
            const session = ensureSession(item.session_id);
            if (session) {
                session.terminated = Boolean(merged.terminated);
            }
        });

        if (payload.live_session_id) {
            liveSessionId = payload.live_session_id;
        }

        if (!activeSessionId && liveSessionId) {
            setActiveSession(liveSessionId);
        } else {
            renderHistoryList();
        }
        updateQAAvailability();
        applyActiveSessionTermination();
    }

    function handleHistorySession(payload) {
        const sessionId = payload.session_id;
        const session = ensureSession(sessionId);
        if (!session) return;

        session.segments = new Map();
        session.segmentOrder = [];
        session.finalSegments = new Map();
        session.finalSegmentOrder = [];
        session.qaLog = [];
        session.draftPreview = '';

        const transcript = (payload.transcript || []).slice().sort((a, b) => {
            const aTs = a.ts || 0;
            const bTs = b.ts || 0;
            return aTs - bTs;
        });
        transcript.forEach(item => {
            const segmentId = item.segment_id || `seg-${Math.random().toString(16).slice(2, 8)}`;
            session.segments.set(segmentId, {
                ts: item.ts || null,
                text: item.text || '',
                isFinal: true,
            });
            session.segmentOrder.push(segmentId);
            session.finalSegments.set(segmentId, { ts: item.ts || null, text: item.text || '' });
            session.finalSegmentOrder.push(segmentId);
        });

        const insights = payload.insights || {};
        session.lastInsights = {
            summary: insights.summary || [],
            summary_live: insights.summary_live || [],
            actions: insights.actions || [],
            questions: insights.questions || [],
        };
        if (payload.terminated) {
            session.terminated = true;
            const meta = historyMeta.get(sessionId) || { session_id: sessionId };
            meta.terminated = true;
            if (payload.terminated_at) {
                meta.terminated_at = payload.terminated_at;
            }
            historyMeta.set(sessionId, meta);
        }
        if (session.lastInsights.summary.length || session.lastInsights.summary_live.length || session.lastInsights.actions.length || session.lastInsights.questions.length) {
            session.lastUpdateTs = Date.now();
        }
        setSessionSortTs(sessionId, Date.now() / 1000);

        const qa = (payload.qa || []).slice().sort((a, b) => {
            const aTs = a.ts_ms || 0;
            const bTs = b.ts_ms || 0;
            return aTs - bTs;
        });
        qa.forEach(item => {
            const tsMs = item.ts_ms || Date.now();
            if (item.question) {
                session.qaLog.push({ role: 'user', text: item.question, ts: tsMs });
            }
            if (item.answer) {
                session.qaLog.push({ role: 'ai', text: item.answer, ts: tsMs });
            }
        });

        session.loaded = true;

        if (sessionId === activeSessionId) {
            renderActiveSession();
            applyActiveSessionTermination();
        }
    }

    function handleAudioDevices(payload) {
        const devices = payload.devices || [];
        serverMics = devices;
        const selected = payload.selected || settings.audio.micId;
        updateMicOptions(devices, selected);
        if (selected && settings.audio.micId !== selected) {
            settings.audio.micId = selected;
            saveSettings();
        }
    }

    function handleAudioPaused(payload) {
        const paused = Boolean(payload.paused);
        audioPaused = paused;
        setPauseButtonState(paused);
        if (liveSessionId && isSessionTerminated(liveSessionId) && activeSessionId === liveSessionId) {
            setStatus('terminated');
            draftPreview.textContent = '会话已终止';
            setPauseButtonDisabled(true);
            return;
        }
        if (paused) {
            setStatus('paused');
            draftPreview.textContent = '录音已暂停';
        } else {
            setStatus('listening');
            draftPreview.textContent = '正在聆听...';
        }
    }

    function handleSettingsApplied(payload) {
        const ignored = payload.ignored || [];
        const requiresRestart = payload.requires_restart || [];

        if (requiresRestart.length > 0) {
            setSettingsStatus('部分需重启', 'warn');
        } else {
            setSettingsStatus('已同步', 'ok');
        }

        if (ignored.length > 0) {
            console.warn('[Settings] Ignored:', ignored);
        }
        if (requiresRestart.length > 0) {
            console.warn('[Settings] Requires restart:', requiresRestart);
        }
    }

    function handleSpeechStart(sessionId, segmentId, ts) {
        if (!sessionId || !segmentId) return;
        const session = ensureSession(sessionId);
        if (!session) return;

        if (!session.segments.has(segmentId)) {
            session.segments.set(segmentId, { ts, text: '', isFinal: false });
            session.segmentOrder.push(segmentId);
        }

        if (sessionId === liveSessionId) {
            currentSegmentId = segmentId;
            setStatus('recording');
        }

        if (sessionId === activeSessionId) {
            ensureTimelineItem(segmentId, ts);
            scrollToBottom();
        }

        console.log(`[VAD] Speech started: ${segmentId}`);
    }

    function handleSpeechEnd(sessionId, segmentId) {
        if (sessionId === liveSessionId && currentSegmentId === segmentId) {
            currentSegmentId = null;
            setStatus('listening');
            draftPreview.textContent = '正在聆听...';
        }
        console.log(`[VAD] Speech ended: ${segmentId}`);
    }

    function handlePartial(sessionId, segmentId, payload, ts) {
        if (!sessionId) return;
        const session = ensureSession(sessionId);
        if (!session) return;

        const text = payload.text || '';
        const isStreaming = payload.streaming === true || segmentId === 'live';

        if (isStreaming) {
            session.draftPreview = text;
            if (sessionId === activeSessionId) {
                draftPreview.textContent = text || '正在聆听...';
            }
            return;
        }

        if (!segmentId) return;
        let segment = session.segments.get(segmentId);
        if (!segment) {
            segment = { ts, text: '', isFinal: false };
            session.segments.set(segmentId, segment);
            session.segmentOrder.push(segmentId);
        }

        segment.text = text;
        segment.isFinal = false;
        if (!segment.ts && ts) {
            segment.ts = ts;
        }

        if (sessionId === activeSessionId) {
            ensureTimelineItem(segmentId, ts);
            const textSpan = document.getElementById(`text-${segmentId}`);
            if (textSpan) {
                textSpan.textContent = text;
                textSpan.className = 'text draft';
            }
            draftPreview.textContent = text || '正在聆听...';
            scrollToBottom();
        }
    }

    function handleFinal(sessionId, segmentId, payload, ts) {
        if (!sessionId || !segmentId) return;
        const session = ensureSession(sessionId);
        if (!session) return;

        const text = payload.text || '';
        const duration = payload.duration_ms || 0;

        let segment = session.segments.get(segmentId);
        if (!segment) {
            segment = { ts, text: '', isFinal: false };
            session.segments.set(segmentId, segment);
            session.segmentOrder.push(segmentId);
        }

        segment.text = text;
        segment.isFinal = true;
        if (!segment.ts && ts) {
            segment.ts = ts;
        }

        if (!session.finalSegments.has(segmentId)) {
            session.finalSegmentOrder.push(segmentId);
        }
        const storedTs = segment.ts || ts;
        session.finalSegments.set(segmentId, { ts: storedTs, text: text });

        markSessionActivity(sessionId, ts);

        if (sessionId === activeSessionId) {
            ensureTimelineItem(segmentId, ts);
            const textSpan = document.getElementById(`text-${segmentId}`);
            if (textSpan) {
                textSpan.textContent = text;
                textSpan.className = 'text final';
            }
            scrollToBottom();
        }

        console.log(`[ASR] Final (${duration}ms): ${text}`);
    }

    // ============ Insights Handler ============
    function handleInsightsUpdate(sessionId, payload) {
        if (!sessionId) return;
        const session = ensureSession(sessionId);
        if (!session) return;

        session.lastInsights = {
            summary: payload.summary || [],
            summary_live: payload.summary_live || [],
            actions: payload.actions || [],
            questions: payload.questions || []
        };

        markSessionActivity(sessionId, Date.now() / 1000);
        session.lastUpdateTs = Date.now();
        setSessionSortTs(sessionId, Date.now() / 1000);

        if (sessionId !== activeSessionId) {
            return;
        }

        if (updateBadge) {
            updateBadge.textContent = '刚刚更新';
            updateBadge.style.opacity = '1';
            setTimeout(() => { updateBadge.style.opacity = '0.7'; }, 2000);
        }

        renderInsights(session);
    }

    // ============ QA Handler ============
    function handleQAAnswer(sessionId, payload) {
        if (!sessionId) return;
        const session = ensureSession(sessionId);
        if (!session) return;

        const answer = payload.answer || '无法获取回答';

        session.qaLog.push({ role: 'ai', text: answer, ts: Date.now() });
        markSessionActivity(sessionId, Date.now() / 1000);
        setSessionSortTs(sessionId, Date.now() / 1000);

        if (pendingQAForSessionId && sessionId === pendingQAForSessionId) {
            waitingForAnswer = false;
            pendingQAForSessionId = null;
        }

        if (sessionId !== activeSessionId) {
            return;
        }

        removeTypingIndicator();
        appendQABubble(answer, 'ai');
        console.log('[QA] Answer received:', answer);
    }

    // ============ QA UI ============
    function sendQA() {
        const query = qaInput.value.trim();
        if (!query || waitingForAnswer) return;

        if (!settings.summary.llmEnabled) {
            appendQABubble('LLM 已关闭，无法提问。', 'ai');
            return;
        }

        if (!activeSessionId) {
            appendQABubble('请先选择一个会话再提问。', 'ai');
            return;
        }

        const session = ensureSession(activeSessionId);
        session.qaLog.push({ role: 'user', text: query, ts: Date.now() });

        appendQABubble(query, 'user');
        qaInput.value = '';

        if (ws && ws.readyState === WebSocket.OPEN) {
            waitingForAnswer = true;
            pendingQAForSessionId = activeSessionId;
            ws.send(JSON.stringify({
                command: 'ask',
                question: query,
                session_id: activeSessionId,
            }));
            appendQABubble('正在思考...', 'ai typing');
        } else {
            appendQABubble('未连接到服务器', 'ai');
        }
    }

    qaSend.addEventListener('click', sendQA);
    qaInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQA();
    });

    // ============ Init ============
    function initPlaceholders() {
        renderEmptyState();
        setPauseButtonState(false);
    }

    initPlaceholders();
    initDownloads();
    initHistoryActions();
    initSettingsDrawer();
    initTimelineAutoScrollGuard();
    applySettings();

    // Connect to WebSocket server
    connect();
});
