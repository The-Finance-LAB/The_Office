/* ═══════════════════════════════════════════════════════════════════
   Configuration — The Apple Brief
   Change API_BASE_URL to your Render deployment URL
   ═══════════════════════════════════════════════════════════════════ */

const CONFIG = {
  // Change this to your Render deployment URL (no trailing slash)
  // e.g., "https://the-office-api.onrender.com"
  API_BASE_URL: "http://localhost:5000",

  // Timing (milliseconds)
  MESSAGE_DISPLAY_DELAY: 3000,   // 3 seconds between messages
  TYPING_INDICATOR_DURATION: 1500, // typing indicator shows for 1.5s
  POLL_INTERVAL: 2000,            // poll every 2 seconds

  // Total messages expected (chief opening + separator + 6 analysts + separator + chief closing)
  TOTAL_AGENT_MESSAGES: 8,  // Only counting non-separator messages for progress bar
};
