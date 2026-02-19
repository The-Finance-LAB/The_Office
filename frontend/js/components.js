/* ═══════════════════════════════════════════════════════════════════
   UI Components — The Apple Brief
   Renders all chat elements as HTML strings
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Highlight financial figures in text with monospace + glow
 */
function highlightFinancials(text) {
  // Match dollar amounts, percentages, large numbers, fiscal years
  const pattern = /(\$[\d,.]+\s*(?:billion|million|trillion|B|M|T)?|[\d,.]+%|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\bFY\d{4}\b|\b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\b)/gi;
  return text.replace(pattern, (match) => {
    return `<span class="financial-figure" style="color: #D4A853; text-shadow: 0 0 8px rgba(212, 168, 83, 0.3);">${match}</span>`;
  });
}

/**
 * Generate a fake timestamp for messages
 */
let messageTimeMinutes = 0;
function getMessageTime() {
  const hours = 14; // 2:00 PM start
  const mins = messageTimeMinutes;
  messageTimeMinutes += Math.floor(Math.random() * 2) + 1;
  const h = hours + Math.floor(mins / 60);
  const m = mins % 60;
  return `${h}:${String(m).padStart(2, '0')} PM`;
}

function resetMessageTime() {
  messageTimeMinutes = 0;
}

/**
 * Render the sidebar with all participants
 */
function renderSidebar() {
  const sidebar = document.getElementById('sidebar');
  let html = '<h2 class="sidebar-title">Participants</h2>';

  ALL_AGENTS.forEach((agent) => {
    html += `
      <div class="sidebar-card glass-card" id="sidebar-${agent.id}"
           style="--card-glow: 0 0 20px ${agent.glowColor};">
        <div class="card-inner">
          <div class="sidebar-avatar" style="background: ${agent.avatarBg};">
            ${agent.initials}
          </div>
          <div class="sidebar-info">
            <div class="sidebar-name" style="color: ${agent.color};">
              ${agent.name}
              <span class="pulse-dot hidden" id="pulse-${agent.id}"></span>
            </div>
            <div class="sidebar-meta">
              ${agent.year ? `<span class="sidebar-year" style="color: ${agent.color}80;">${agent.year}</span>` : ''}
              <span class="sidebar-role">${agent.role}</span>
            </div>
          </div>
        </div>
      </div>
    `;
  });

  html += `
    <div class="sidebar-datasource glass-card">
      <p>
        <span class="label">Data Source</span><br/>
        Apple Inc. 10-K SEC Filings<br/>
        FY2020 — FY2025
      </p>
    </div>
  `;

  sidebar.innerHTML = html;
}

/**
 * Render the chat header
 */
function renderHeader() {
  const header = document.getElementById('chatHeader');
  const agents = ALL_AGENTS.slice(0, 5);

  let avatarsHtml = agents.map(a => `
    <div class="mini-avatar" style="background: ${a.avatarBg};" title="${a.name}">
      ${a.initials}
    </div>
  `).join('');
  avatarsHtml += '<div class="mini-avatar mini-avatar-more">+2</div>';

  header.innerHTML = `
    <div class="header-row">
      <div class="header-icon">
        <div class="header-apple-icon">
          <svg width="20" height="24" viewBox="0 0 814 1000" fill="#0B1120">
            <path d="M788.1 340.9c-5.8 4.5-108.2 62.2-108.2 190.5 0 148.4 130.3 200.9 134.2 202.2-.6 3.2-20.7 71.9-68.7 141.9-42.8 61.6-87.5 123.1-155.5 123.1s-85.5-39.5-164-39.5c-76.5 0-103.7 40.8-165.9 40.8s-105.6-57.8-155.5-127.4c-58.3-81.6-105.6-208.4-105.6-328.6 0-193.3 125.7-296 249.3-296 65.7 0 120.5 43.1 161.7 43.1 39.2 0 100.4-45.8 175.1-45.8 28.3 0 130 2.6 197.1 98.7zM554.1 159.4c31.1-36.9 53.1-88.1 53.1-139.4 0-7.1-.6-14.3-1.9-20.1-50.6 1.9-110.8 33.7-147.1 75.8-28.3 32.4-55.1 83.6-55.1 135.5 0 7.8.6 15.6 1.3 18.2 2.6.6 6.4 1.3 10.2 1.3 45.4 0 103-30.4 139.5-71.3z"/>
          </svg>
        </div>
        <div class="header-online-dot"></div>
      </div>
      <div class="header-info">
        <h1 class="header-title">THE APPLE BRIEF</h1>
        <p class="header-status" id="headerStatus">
          <span class="live-dot"></span>
          Live — 42nd floor conference room
        </p>
      </div>
      <div class="header-avatars">${avatarsHtml}</div>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" id="progressFill"></div>
    </div>
  `;
}

/**
 * Render the welcome screen
 */
function renderWelcome() {
  const welcome = document.getElementById('welcomeScreen');
  let agentPills = ALL_AGENTS.map(a => `
    <div class="welcome-agent-pill"
         style="background-color: ${a.color}10; border: 1px solid ${a.color}20;">
      <div class="pill-avatar" style="background: ${a.avatarBg};">${a.initials}</div>
      <span class="pill-name" style="color: ${a.color}CC;">${a.name}</span>
    </div>
  `).join('');

  welcome.innerHTML = `
    <div class="welcome-card glass-card">
      <h2 class="welcome-title">THE APPLE BRIEF</h2>
      <p class="welcome-subtitle">An Analyst Roundtable</p>
      <p class="welcome-desc">
        Conference room, 42nd floor. Six analysts, one question,
        and a lot of data between them.
      </p>
      <div class="welcome-agents">${agentPills}</div>
      <p class="welcome-footer">
        Type your question below to start the roundtable.<br/>
        <span class="mono">Powered by AI Analyst Agents — FY2020–FY2025</span>
      </p>
    </div>
  `;
}

/**
 * Create a chat bubble HTML element
 */
function createChatBubble(message) {
  const agent = AGENTS[message.agentId];
  if (!agent) return '';

  const time = getMessageTime();
  const isChief = message.agentId === 'chief';
  const yearTag = agent.year
    ? `<span class="bubble-year-tag" style="background: ${agent.color}15; color: ${agent.color};">${agent.year}</span>`
    : '';

  const bodyStyle = `
    border-color: ${agent.color}40;
    box-shadow: 0 0 20px ${agent.glowColor};
    ${isChief ? `background: linear-gradient(135deg, rgba(212,168,83,0.08), rgba(184,148,46,0.03));` : ''}
  `;

  return `
    <div class="chat-bubble" data-agent="${agent.id}">
      <div class="bubble-avatar" style="background: ${agent.avatarBg}; box-shadow: 0 0 12px ${agent.glowColor};">
        ${agent.initials}
      </div>
      <div class="bubble-content">
        <div class="bubble-header">
          <span class="bubble-name" style="color: ${agent.color};">${agent.name}</span>
          ${yearTag}
          <span class="bubble-role">${agent.role}</span>
        </div>
        <div class="bubble-body glass-card ${isChief ? 'chief-gradient' : ''}" style="${bodyStyle}">
          <p class="bubble-text">${highlightFinancials(message.text)}</p>
        </div>
        <div class="bubble-timestamp">
          <span class="bubble-time">${time}</span>
          <span class="bubble-check">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
          </span>
        </div>
      </div>
    </div>
  `;
}

/**
 * Create a separator HTML element
 */
function createSeparator(text) {
  return `
    <div class="separator">
      <div class="separator-inner">
        <div class="separator-line separator-line-left"></div>
        <span class="separator-text">${text}</span>
        <div class="separator-line separator-line-right"></div>
      </div>
    </div>
  `;
}

/**
 * Create a typing indicator HTML element
 */
function createTypingIndicator(agentId) {
  const agent = AGENTS[agentId];
  if (!agent) return '';

  return `
    <div class="typing-indicator" id="typingIndicator">
      <div class="typing-avatar" style="background: ${agent.avatarBg};">
        ${agent.initials}
      </div>
      <div class="typing-pill glass-card"
           style="border-color: ${agent.color}20; box-shadow: 0 0 12px ${agent.glowColor};">
        <span class="typing-dot" style="background-color: ${agent.color};"></span>
        <span class="typing-dot" style="background-color: ${agent.color};"></span>
        <span class="typing-dot" style="background-color: ${agent.color};"></span>
      </div>
    </div>
  `;
}

/**
 * Create end-of-meeting HTML element
 */
function createEndOfMeeting() {
  return `
    <div class="end-of-meeting">
      <div class="end-divider">
        <div class="end-divider-line separator-line-left"></div>
        <span class="end-divider-text">end of meeting</span>
        <div class="end-divider-line separator-line-right"></div>
      </div>
      <div class="end-card glass-card">
        <p class="end-title">Multi-Agent Financial Analyst System</p>
        <p class="end-subtitle">Apple 10-K Reports — FY2020 through FY2025</p>
        <div class="end-stats">
          <div class="end-stat">
            <div class="end-stat-value">7</div>
            <div class="end-stat-label">Agents</div>
          </div>
          <div class="end-stat-divider"></div>
          <div class="end-stat">
            <div class="end-stat-value">6</div>
            <div class="end-stat-label">Years</div>
          </div>
          <div class="end-stat-divider"></div>
          <div class="end-stat">
            <div class="end-stat-value">1</div>
            <div class="end-stat-label">Verdict</div>
          </div>
        </div>
      </div>
      <button class="new-question-btn" onclick="resetForNewQuestion()">
        Ask Another Question
      </button>
      <p class="end-hint">Type a new question in the input below to start another roundtable</p>
    </div>
  `;
}

/**
 * Create error message HTML
 */
function createErrorMessage(text) {
  return `
    <div class="error-message">
      <div class="error-inner glass-card">
        <p class="error-text">Error: ${text}</p>
      </div>
    </div>
  `;
}

/**
 * Update sidebar active state
 */
function setSidebarActive(agentId) {
  document.querySelectorAll('.sidebar-card').forEach(card => {
    card.classList.remove('active');
    card.style.boxShadow = '';
    card.style.borderColor = '';
  });
  document.querySelectorAll('.pulse-dot').forEach(dot => dot.classList.add('hidden'));

  if (agentId) {
    const card = document.getElementById(`sidebar-${agentId}`);
    const pulse = document.getElementById(`pulse-${agentId}`);
    const agent = AGENTS[agentId];
    if (card && agent) {
      card.classList.add('active');
      card.style.boxShadow = `0 0 20px ${agent.glowColor}`;
      card.style.borderColor = `${agent.color}40`;
    }
    if (pulse) pulse.classList.remove('hidden');
  }
}

/**
 * Update progress bar
 */
function updateProgress(current, total) {
  const fill = document.getElementById('progressFill');
  if (fill && total > 0) {
    fill.style.width = `${(current / total) * 100}%`;
  }
}

/**
 * Update header status text
 */
function setHeaderStatus(text, isLive) {
  const status = document.getElementById('headerStatus');
  if (status) {
    status.innerHTML = isLive
      ? `<span class="live-dot"></span> ${text}`
      : text;
  }
}
