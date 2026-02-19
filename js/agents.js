/* ═══════════════════════════════════════════════════════════════════
   Agent Definitions — The Apple Brief
   Matches the Python backend analyst personalities
   ═══════════════════════════════════════════════════════════════════ */

const AGENTS = {
  chief: {
    id: "chief",
    name: "Chief Analyst",
    role: "Lead",
    year: null,
    color: "#D4A853",
    glowColor: "rgba(212, 168, 83, 0.3)",
    initials: "CA",
    avatarBg: "linear-gradient(135deg, #D4A853, #B8942E)",
  },
  marcus: {
    id: "marcus",
    name: "Marcus",
    role: "The Veteran",
    year: "FY2020",
    color: "#8B9DC3",
    glowColor: "rgba(139, 157, 195, 0.3)",
    initials: "M",
    avatarBg: "linear-gradient(135deg, #8B9DC3, #6B7DAA)",
  },
  priya: {
    id: "priya",
    name: "Priya",
    role: "Growth Evangelist",
    year: "FY2021",
    color: "#C4687A",
    glowColor: "rgba(196, 104, 122, 0.3)",
    initials: "P",
    avatarBg: "linear-gradient(135deg, #C4687A, #A84D5F)",
  },
  james: {
    id: "james",
    name: "James",
    role: "The Contrarian",
    year: "FY2022",
    color: "#4A9E8E",
    glowColor: "rgba(74, 158, 142, 0.3)",
    initials: "J",
    avatarBg: "linear-gradient(135deg, #4A9E8E, #3A7E72)",
  },
  sofia: {
    id: "sofia",
    name: "Sofia",
    role: "Macro Thinker",
    year: "FY2023",
    color: "#E8A87C",
    glowColor: "rgba(232, 168, 124, 0.3)",
    initials: "S",
    avatarBg: "linear-gradient(135deg, #E8A87C, #D08A5E)",
  },
  derek: {
    id: "derek",
    name: "Derek",
    role: "The Quant",
    year: "FY2024",
    color: "#6B8DB2",
    glowColor: "rgba(107, 141, 178, 0.3)",
    initials: "D",
    avatarBg: "linear-gradient(135deg, #6B8DB2, #4F7196)",
  },
  anika: {
    id: "anika",
    name: "Anika",
    role: "The Closer",
    year: "FY2025",
    color: "#8B7EC8",
    glowColor: "rgba(139, 126, 200, 0.3)",
    initials: "A",
    avatarBg: "linear-gradient(135deg, #8B7EC8, #6F62AC)",
  },
};

// Ordered list of analyst IDs (excluding chief) for iteration
const ANALYST_ORDER = ["marcus", "priya", "james", "sofia", "derek", "anika"];

// All agents including chief
const ALL_AGENTS = Object.values(AGENTS);

// Map backend year IDs to frontend agent IDs
const YEAR_TO_AGENT = {
  "2020": "marcus",
  "2021": "priya",
  "2022": "james",
  "2023": "sofia",
  "2024": "derek",
  "2025": "anika",
  "chief": "chief",
};

// Map backend agent names to frontend agent IDs (fallback)
const NAME_TO_AGENT = {
  "marcus": "marcus",
  "priya": "priya",
  "james": "james",
  "sofia": "sofia",
  "derek": "derek",
  "anika": "anika",
  "chief analyst": "chief",
  "chief": "chief",
};

/**
 * Resolve a backend agent identifier to a frontend agent ID
 */
function resolveAgentId(backendId, name) {
  if (YEAR_TO_AGENT[backendId]) return YEAR_TO_AGENT[backendId];
  if (AGENTS[backendId]) return backendId;
  if (name) {
    const lower = name.toLowerCase();
    if (NAME_TO_AGENT[lower]) return NAME_TO_AGENT[lower];
  }
  return backendId;
}
