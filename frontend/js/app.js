/* ═══════════════════════════════════════════════════════════════════
   Application Logic — The Apple Brief
   Handles polling, message queue, and UI orchestration
   ═══════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────
let sessionId = null;
let lastSeenId = 0;
let isRunning = false;
let isComplete = false;
let displayedCount = 0;

// Message queue for sequential display with delays
const messageQueue = [];
let processingQueue = false;
let pollTimer = null;
let displayTimer = null;
let serverTypingAgent = null;

// ── DOM References ────────────────────────────────────────────────────
const chatArea = document.getElementById('chatArea');
const welcomeScreen = document.getElementById('welcomeScreen');
const messagesContainer = document.getElementById('messagesContainer');
const inputForm = document.getElementById('inputForm');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');

// ── Initialize ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderSidebar();
  renderHeader();
  renderWelcome();

  // Input handling
  questionInput.addEventListener('input', () => {
    sendBtn.disabled = !questionInput.value.trim() || isRunning;
  });

  inputForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (!q || isRunning) return;
    startRoundtable(q);
  });
});

// ── Scroll to bottom ──────────────────────────────────────────────────
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ── Remove typing indicator ───────────────────────────────────────────
function removeTypingIndicator() {
  const existing = document.getElementById('typingIndicator');
  if (existing) existing.remove();
}

// ── Show typing indicator ─────────────────────────────────────────────
function showTypingIndicator(agentId) {
  removeTypingIndicator();
  const html = createTypingIndicator(agentId);
  messagesContainer.insertAdjacentHTML('beforeend', html);
  setSidebarActive(agentId);
  scrollToBottom();
}

// ── Process message queue with 3-second delays ────────────────────────
function processQueue() {
  if (processingQueue) return;
  processingQueue = true;
  showNext();
}

function showNext() {
  if (messageQueue.length === 0) {
    processingQueue = false;
    // If server says someone is typing, show it
    if (serverTypingAgent) {
      showTypingIndicator(serverTypingAgent);
    }
    // If complete and queue drained, show end
    if (isComplete) {
      finishMeeting();
    }
    return;
  }

  const msg = messageQueue.shift();

  // For separators, show immediately without counting toward progress
  if (msg.type === 'separator') {
    removeTypingIndicator();
    messagesContainer.insertAdjacentHTML('beforeend', createSeparator(msg.text));
    scrollToBottom();
    displayTimer = setTimeout(showNext, 800);
    return;
  }

  // Show typing indicator for this agent
  showTypingIndicator(msg.agentId);

  displayTimer = setTimeout(() => {
    // Remove typing, show message
    removeTypingIndicator();
    messagesContainer.insertAdjacentHTML('beforeend', createChatBubble(msg));
    displayedCount++;
    updateProgress(displayedCount, CONFIG.TOTAL_AGENT_MESSAGES);
    setSidebarActive(msg.agentId);
    scrollToBottom();

    // Wait before showing next
    displayTimer = setTimeout(showNext, CONFIG.MESSAGE_DISPLAY_DELAY);
  }, CONFIG.TYPING_INDICATOR_DURATION);
}

function enqueueMessage(msg) {
  messageQueue.push(msg);
  processQueue();
}

// ── Polling logic ─────────────────────────────────────────────────────
async function pollSession() {
  if (!sessionId) return;

  try {
    const response = await fetch(
      `${CONFIG.API_BASE_URL}/api/roundtable/${sessionId}?after=${lastSeenId}`
    );

    if (!response.ok) {
      throw new Error(`Poll failed: ${response.status}`);
    }

    const data = await response.json();
    const startId = lastSeenId;

    // Enqueue new messages
    for (let i = 0; i < data.messages.length; i++) {
      const msg = data.messages[i];
      const msgIndex = startId + i;
      lastSeenId = msgIndex + 1;
      const frontendAgentId = resolveAgentId(msg.agentId, msg.name);
      enqueueMessage({
        id: msgIndex,
        agentId: frontendAgentId,
        text: msg.text,
        type: msg.type,
      });
    }

    // Track server typing state
    if (data.typing) {
      serverTypingAgent = resolveAgentId(data.typing.agentId, data.typing.name);
      // If queue is empty and not processing, show typing
      if (messageQueue.length === 0 && !processingQueue) {
        showTypingIndicator(serverTypingAgent);
      }
    } else {
      serverTypingAgent = null;
    }

    // Check if done
    if (data.status === 'complete') {
      isComplete = true;
      // Queue will handle showing end of meeting when drained
      if (messageQueue.length === 0 && !processingQueue) {
        finishMeeting();
      }
      return; // Stop polling
    }

    if (data.status === 'error') {
      messagesContainer.insertAdjacentHTML('beforeend',
        createErrorMessage(data.error || 'Unknown error'));
      scrollToBottom();
      isComplete = true;
      if (messageQueue.length === 0 && !processingQueue) {
        enableInput();
      }
      return;
    }

    // Continue polling
    pollTimer = setTimeout(pollSession, CONFIG.POLL_INTERVAL);

  } catch (err) {
    console.error('[Poll] Error:', err.message);
    // Retry after delay
    pollTimer = setTimeout(pollSession, CONFIG.POLL_INTERVAL * 2);
  }
}

// ── Start a new roundtable ────────────────────────────────────────────
async function startRoundtable(question) {
  // Reset state
  resetState();
  isRunning = true;
  disableInput();
  resetMessageTime();

  // Switch to messages view
  welcomeScreen.classList.add('hidden');
  messagesContainer.classList.remove('hidden');
  messagesContainer.innerHTML = '';

  setHeaderStatus('Live — 42nd floor conference room', true);

  try {
    const response = await fetch(`${CONFIG.API_BASE_URL}/api/roundtable`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.error || `Server error: ${response.status}`);
    }

    const data = await response.json();
    sessionId = data.session_id;

    // Start polling
    pollTimer = setTimeout(pollSession, 500);

  } catch (err) {
    messagesContainer.insertAdjacentHTML('beforeend',
      createErrorMessage(err.message));
    scrollToBottom();
    enableInput();
    isRunning = false;
  }
}

// ── Finish meeting ────────────────────────────────────────────────────
function finishMeeting() {
  removeTypingIndicator();
  messagesContainer.insertAdjacentHTML('beforeend', createEndOfMeeting());
  scrollToBottom();
  setHeaderStatus('Meeting concluded', false);
  updateProgress(CONFIG.TOTAL_AGENT_MESSAGES, CONFIG.TOTAL_AGENT_MESSAGES);
  setSidebarActive(null);
  enableInput();
  isRunning = false;
}

// ── Reset for new question ────────────────────────────────────────────
function resetForNewQuestion() {
  questionInput.value = '';
  questionInput.placeholder = 'Ask another question about Apple...';
  questionInput.focus();
}

function resetState() {
  sessionId = null;
  lastSeenId = 0;
  isComplete = false;
  displayedCount = 0;
  messageQueue.length = 0;
  processingQueue = false;
  serverTypingAgent = null;

  if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
  if (displayTimer) { clearTimeout(displayTimer); displayTimer = null; }

  removeTypingIndicator();
}

// ── Input state management ────────────────────────────────────────────
function disableInput() {
  questionInput.disabled = true;
  questionInput.placeholder = 'Analysts are discussing...';
  sendBtn.disabled = true;
  sendBtn.innerHTML = '<div class="spinner"></div>';
}

function enableInput() {
  questionInput.disabled = false;
  questionInput.placeholder = 'Ask another question about Apple...';
  questionInput.value = '';
  sendBtn.disabled = true;
  sendBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
         stroke="rgba(255,255,255,0.2)" stroke-width="2"
         stroke-linecap="round" stroke-linejoin="round">
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22L11 13L2 9L22 2Z" />
    </svg>
  `;
}
