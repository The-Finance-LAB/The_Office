# -*- coding: utf-8 -*-
"""
🍎 AAPL Analyst Office — Flask API Backend
Preserves the original LangChain multi-agent architecture from Colab.
"""

import os
import json
import uuid
import time
import threading
import glob
import re
import sqlite3
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Project Paths ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
FILINGS_DIR = BASE_DIR / "filings"
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = str(DB_DIR / "aapl_office.db")
CHROMA_DIR = str(DB_DIR / "chroma")


# ════════════════════════════════════════════════════════════════════════════
# CORE INFRASTRUCTURE (Extractors + Stores) — Preserved from original
# ════════════════════════════════════════════════════════════════════════════

STATEMENT_PATTERNS = {
    "income_statement": [
        r"consolidated\s+statements?\s+of\s+operations",
        r"consolidated\s+statements?\s+of\s+income",
        r"revenue|net\s+sales|cost\s+of\s+(goods\s+)?sales|gross\s+margin",
    ],
    "balance_sheet": [
        r"consolidated\s+balance\s+sheets?",
        r"total\s+assets|total\s+liabilities|shareholders.*equity",
    ],
    "cash_flow": [
        r"consolidated\s+statements?\s+of\s+cash\s+flows?",
        r"operating\s+activities|investing\s+activities|financing\s+activities",
    ],
    "equity_statement": [
        r"consolidated\s+statements?\s+of\s+shareholders.*equity",
    ],
    "comprehensive_income": [
        r"consolidated\s+statements?\s+of\s+comprehensive\s+income",
    ],
}


@dataclass
class ExtractedTable:
    page_number: int
    statement_type: str
    title: str
    headers: list
    data: list
    raw_dataframe: Optional[str] = None
    fiscal_year: Optional[str] = None
    company: str = "AAPL"
    metadata: dict = field(default_factory=dict)
    def to_dict(self): return asdict(self)


# ── Table Extractor ──────────────────────────────────────────────────────
class TenKTableExtractor:
    def __init__(self, pdf_path, company="AAPL"):
        self.pdf_path = Path(pdf_path)
        self.company = company
        self.doc = fitz.open(str(self.pdf_path))
        self.tables = []
        self.fiscal_year = self._detect_fiscal_year()

    def _detect_fiscal_year(self):
        text = " ".join(self.doc[i].get_text() for i in range(min(3, self.doc.page_count)))
        m = re.search(r"fiscal\s+year\s+ended\s+\w+\s+\d+,?\s+(\d{4})", text, re.I)
        return m.group(1) if m else "unknown"

    def _classify_page(self, text):
        tl = text.lower()
        scores = {}
        for st, pats in STATEMENT_PATTERNS.items():
            s = sum(1 for p in pats if re.search(p, tl))
            if s > 0: scores[st] = s
        return max(scores, key=scores.get) if scores else "note_or_other"

    def _extract_title(self, text):
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        kws = ["CONSOLIDATED","BALANCE SHEET","INCOME","CASH FLOW","OPERATIONS","EQUITY"]
        for l in lines[:10]:
            if any(kw in l.upper() for kw in kws): return l
        return lines[0] if lines else "Untitled"

    def _clean_value(self, val):
        if val is None or val == "" or val == "\u2014": return None
        val = str(val).strip().replace("$", "").strip()
        if val.startswith("(") and val.endswith(")"): val = "-" + val[1:-1]
        val = val.replace(",", "")
        try: return float(val)
        except ValueError: return val if val else None

    def _make_unique_headers(self, headers):
        unique, seen = [], {}
        for h in headers:
            h_str = str(h).strip() if h else ""
            if not h_str: h_str = "col"
            if h_str in seen:
                seen[h_str] += 1
                h_str = f"{h_str}_{seen[h_str]}"
            else:
                seen[h_str] = 0
            unique.append(h_str)
        return unique

    def extract_all(self):
        self.tables = []
        for page_num in range(self.doc.page_count):
            page = self.doc[page_num]
            page_text = page.get_text()
            for table in page.find_tables().tables:
                if table.row_count < 2: continue
                raw = table.extract()
                if not raw or len(raw) < 2: continue
                headers = self._make_unique_headers(raw[0])
                df = pd.DataFrame(raw[1:], columns=headers)
                for col in df.columns:
                    df[col] = df[col].apply(lambda x: self._clean_value(x) if isinstance(x, str) else x)
                df = df.dropna(how="all")
                if len(df.columns) > 1:
                    first_col = df.columns[0]
                    if df[first_col].apply(lambda x: isinstance(x, str)).any():
                        df = df.rename(columns={first_col: "line_item"})
                if df.empty: continue
                self.tables.append(ExtractedTable(
                    page_number=page_num+1,
                    statement_type=self._classify_page(page_text),
                    title=self._extract_title(page_text),
                    headers=list(df.columns),
                    data=df.to_dict(orient="records"),
                    raw_dataframe=df.to_csv(index=False),
                    fiscal_year=self.fiscal_year,
                    company=self.company,
                    metadata={"rows": len(df), "columns": len(df.columns)}
                ))
        return self.tables


# ── Text Extractor ────────────────────────────────────────────────────────
class TenKTextExtractor:
    SECTIONS = {
        "business": r"Item\s+1\.\s+Business",
        "risk_factors": r"Item\s+1A\.\s+Risk\s+Factors",
        "mda": r"Item\s+7\.\s+Management.s\s+Discussion",
        "financial_statements": r"Item\s+8\.\s+Financial\s+Statements",
    }

    def __init__(self, pdf_path):
        self.doc = fitz.open(pdf_path)

    def extract_sections(self):
        full_text = "\n\n".join(self.doc[i].get_text() for i in range(self.doc.page_count))
        sections, boundaries = {}, []
        for name, pattern in self.SECTIONS.items():
            match = re.search(pattern, full_text, re.I)
            if match: boundaries.append((match.start(), name))
        boundaries.sort(key=lambda x: x[0])
        for i, (start, name) in enumerate(boundaries):
            end = boundaries[i+1][0] if i+1 < len(boundaries) else len(full_text)
            sections[name] = full_text[start:end].strip()
        return sections

    def chunk_for_embedding(self, chunk_size=800, overlap=150):
        sections = self.extract_sections()
        chunks = []
        for section_name, text in sections.items():
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i:i+chunk_size]
                chunks.append({"text": " ".join(chunk_words), "section": section_name,
                               "chunk_index": len(chunks), "word_count": len(chunk_words)})
        return chunks


# ── Structured Store (SQLite) ─────────────────────────────────────────────
class StructuredFinancialStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT, fiscal_year TEXT, source_file TEXT,
                UNIQUE(company, fiscal_year)
            );
            CREATE TABLE IF NOT EXISTS financial_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id INTEGER, statement_type TEXT, title TEXT,
                page_number INTEGER, headers_json TEXT, data_json TEXT,
                csv_data TEXT, metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS line_items (
                filing_id INTEGER, table_id INTEGER, statement_type TEXT,
                line_item TEXT NOT NULL, value REAL, period TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_line_search
                ON line_items(line_item COLLATE NOCASE);
        """)
        self.conn.commit()

    def store_filing(self, company, fiscal_year, tables, source_file=""):
        self.conn.execute(
            "INSERT OR REPLACE INTO filings (company, fiscal_year, source_file) VALUES (?,?,?)",
            (company, fiscal_year, source_file))
        self.conn.commit()
        filing_id = self.conn.execute(
            "SELECT id FROM filings WHERE company=? AND fiscal_year=?",
            (company, fiscal_year)).fetchone()["id"]
        self.conn.execute("DELETE FROM financial_tables WHERE filing_id=?", (filing_id,))
        self.conn.execute("DELETE FROM line_items WHERE filing_id=?", (filing_id,))

        for table in tables:
            cursor = self.conn.execute(
                """INSERT INTO financial_tables
                   (filing_id, statement_type, title, page_number, headers_json, data_json, csv_data, metadata_json)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (filing_id, table.get("statement_type"), table.get("title"),
                 table.get("page_number",0), json.dumps(table.get("headers",[])),
                 json.dumps(table.get("data",[])), table.get("raw_dataframe",""),
                 json.dumps(table.get("metadata",{}))))
            table_id = cursor.lastrowid

            for row in table.get("data", []):
                line_item = row.get("line_item", None)
                if line_item is None:
                    for key, val in row.items():
                        if isinstance(val, str) and len(val) > 2 and not val.replace("-","").replace(".","").replace(",","").isdigit():
                            line_item = val; break
                if not line_item or not isinstance(line_item, str) or len(line_item) <= 2:
                    continue
                values = {}
                for key, val in row.items():
                    if key == "line_item": continue
                    if isinstance(val, (int, float)) and val == val:
                        values[key] = val
                for period, value in values.items():
                    self.conn.execute(
                        "INSERT INTO line_items (filing_id, table_id, statement_type, line_item, value, period) VALUES (?,?,?,?,?,?)",
                        (filing_id, table_id, table.get("statement_type"), line_item, value, period))
        self.conn.commit()
        return filing_id

    def query_line_item(self, line_item, company="AAPL", fiscal_year=None):
        q = """SELECT li.line_item, li.value, li.period, li.statement_type,
                      f.company, f.fiscal_year, ft.title, ft.page_number
               FROM line_items li
               JOIN filings f ON li.filing_id = f.id
               JOIN financial_tables ft ON li.table_id = ft.id
               WHERE li.line_item LIKE ? AND f.company = ?"""
        p = [f"%{line_item}%", company]
        if fiscal_year: q += " AND f.fiscal_year = ?"; p.append(fiscal_year)
        q += " ORDER BY f.fiscal_year DESC"
        return [dict(r) for r in self.conn.execute(q, p).fetchall()]

    def get_statement(self, statement_type, company="AAPL", fiscal_year=None):
        q = """SELECT ft.*, f.company, f.fiscal_year FROM financial_tables ft
               JOIN filings f ON ft.filing_id = f.id
               WHERE ft.statement_type = ? AND f.company = ?"""
        p = [statement_type, company]
        if fiscal_year: q += " AND f.fiscal_year = ?"; p.append(fiscal_year)
        rows = self.conn.execute(q, p).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["headers"] = json.loads(r.pop("headers_json", "[]"))
            r["data"] = json.loads(r.pop("data_json", "[]"))
            r.pop("csv_data", None)
            r.pop("metadata_json", None)
            results.append(r)
        return results

    def get_summary(self, company="AAPL"):
        filings = self.conn.execute(
            "SELECT fiscal_year, source_file FROM filings WHERE company=?", (company,)).fetchall()
        counts = self.conn.execute(
            """SELECT ft.statement_type, COUNT(*) as c FROM financial_tables ft
               JOIN filings f ON ft.filing_id = f.id WHERE f.company = ?
               GROUP BY ft.statement_type""", (company,)).fetchall()
        line_count = self.conn.execute(
            """SELECT COUNT(*) as c FROM line_items li
               JOIN filings f ON li.filing_id = f.id WHERE f.company = ?""", (company,)).fetchone()["c"]
        return {"company": company, "filings": [dict(f) for f in filings],
                "table_counts": {r["statement_type"]: r["c"] for r in counts},
                "total_line_items": line_count}


# ── Vector Store (ChromaDB) ───────────────────────────────────────────────
class VectorTextStore:
    def __init__(self, persist_dir="./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="tenk_text", metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks, company, fiscal_year):
        ids, docs, metas = [], [], []
        for i, c in enumerate(chunks):
            ids.append(f"{company}_{fiscal_year}_{c['section']}_{i}")
            docs.append(c["text"])
            metas.append({"company": company, "fiscal_year": fiscal_year,
                          "section": c["section"], "chunk_index": c["chunk_index"]})
        self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def search(self, query, company="AAPL", fiscal_year=None, n_results=5, section=None):
        where_clauses = [{"company": company}]
        if fiscal_year: where_clauses.append({"fiscal_year": fiscal_year})
        if section: where_clauses.append({"section": section})
        where_filter = {"$and": where_clauses} if len(where_clauses) > 1 else where_clauses[0]
        results = self.collection.query(
            query_texts=[query], n_results=n_results, where=where_filter)
        return [{"text": d, "metadata": m, "distance": dist}
                for d, m, dist in zip(
                    results["documents"][0], results["metadatas"][0], results["distances"][0])]


# ════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT ARCHITECTURE — Preserved from original
# ════════════════════════════════════════════════════════════════════════════

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool


def make_year_tools(fiscal_year, store, vec):
    """Factory: creates tools scoped to a single fiscal year."""

    query_metric_description = (
        f"Query a financial metric from AAPL's FY{fiscal_year} 10-K. "
        f"Examples: 'total net sales', 'net income', 'total assets', 'cash and cash equivalents'."
    )
    @tool(description=query_metric_description)
    def query_metric(metric: str) -> str:
        """Query a financial metric."""
        results = store.query_line_item(metric, "AAPL", fiscal_year)
        return json.dumps({"fiscal_year": fiscal_year, "metric": metric,
                           "results": results[:15]}, default=str)

    get_statement_description = (
        f"Get a complete financial statement from AAPL's FY{fiscal_year} 10-K. "
        f"Types: balance_sheet, income_statement, cash_flow, equity_statement."
    )
    @tool(description=get_statement_description)
    def get_statement(statement_type: str) -> str:
        """Get a complete financial statement."""
        results = store.get_statement(statement_type, "AAPL", fiscal_year)
        return json.dumps({"fiscal_year": fiscal_year, "results": results}, default=str)

    search_narrative_description = (
        f"Search narrative text (MD&A, Risk Factors, Business) from AAPL's FY{fiscal_year} 10-K."
    )
    @tool(description=search_narrative_description)
    def search_narrative(query: str) -> str:
        """Search narrative text."""
        results = vec.search(query, "AAPL", fiscal_year=fiscal_year, n_results=3)
        return json.dumps({"fiscal_year": fiscal_year, "results": [
            {"text": r["text"][:500], "section": r["metadata"]["section"]}
            for r in results
        ]}, default=str)

    query_metric.name = f"fy{fiscal_year}_query_metric"
    get_statement.name = f"fy{fiscal_year}_get_statement"
    search_narrative.name = f"fy{fiscal_year}_search_narrative"

    return [query_metric, get_statement, search_narrative]


# ── Cross-year tools ─────────────────────────────────────────────────────
structured_store = None
vector_store = None


def init_cross_year_tools(store):
    """Initialize cross-year tools with the shared store."""
    @tool
    def compare_metric_across_years(metric: str) -> str:
        """Compare a financial metric across ALL available fiscal years."""
        results = store.query_line_item(metric, "AAPL")
        by_year = {}
        for r in results:
            fy = r["fiscal_year"]
            if fy not in by_year: by_year[fy] = []
            by_year[fy].append({"value": r["value"], "period": r["period"],
                                "statement": r["statement_type"]})
        return json.dumps({"metric": metric, "years_available": sorted(by_year.keys()),
                           "data_by_year": by_year}, default=str)

    @tool
    def list_available_years() -> str:
        """List all fiscal years that have been ingested."""
        summary = store.get_summary("AAPL")
        return json.dumps(summary, default=str)

    return [compare_metric_across_years, list_available_years]


# ── Year Specialist Agent ────────────────────────────────────────────────
class YearSpecialist:
    """An agent specialized in one fiscal year of Apple's 10-K."""

    def __init__(self, fiscal_year, store, vec_store, model="claude-sonnet-4-20250514"):
        self.fiscal_year = fiscal_year
        self.tools = make_year_tools(fiscal_year, store, vec_store)

        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(model=model, temperature=0, max_tokens=4096)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.system_prompt = f"""You are the FY{fiscal_year} Financial Specialist for Apple Inc. (AAPL).

You are an expert EXCLUSIVELY on Apple's 10-K filing for fiscal year {fiscal_year}.
You have access to:
- All financial tables (balance sheet, income statement, cash flow, etc.)
- Narrative text (MD&A, risk factors, business description)

Rules:
- ONLY answer questions about FY{fiscal_year} data
- Always use your tools to look up data — never guess numbers
- Report values in millions USD
- Cite the page number when possible
- If asked about other years, say you only cover FY{fiscal_year}

Keep responses focused and data-driven. The Chief Analyst may ask you
specific questions to compare with other years — be precise."""

    def invoke(self, question):
        """Ask this specialist a question and get a response."""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=question),
        ]
        for _ in range(5):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                matching_tool = next((t for t in self.tools if t.name == tc["name"]), None)
                if matching_tool:
                    result = matching_tool.invoke(tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return response.content if isinstance(response.content, str) else str(response.content)


# ── Chief Analyst (Supervisor) ───────────────────────────────────────────
class ChiefAnalyst:
    def __init__(self, specialists, store, vec_store, model="claude-sonnet-4-20250514"):
        self.specialists = specialists
        self.available_years = sorted(specialists.keys())
        self.store = store
        self.vec_store = vec_store
        self.own_tools = init_cross_year_tools(store)

        from langchain_anthropic import ChatAnthropic
        self.llm = ChatAnthropic(model=model, temperature=0, max_tokens=8192)
        self.llm_with_tools = self.llm.bind_tools(self.own_tools)

        self.system_prompt = f"""You are the Chief Financial Analyst for the AAPL Analyst Office.

You lead a team of year-specialist agents, each expert on one fiscal year of Apple's 10-K.

Available specialists: {', '.join(f'FY{y}' for y in self.available_years)}

YOUR WORKFLOW:
1. ANALYZE the user's question to determine which year(s) are relevant
2. For single-year questions: Delegate to that specialist
3. For multi-year/trend questions: Query multiple specialists, then synthesize
4. For cross-year comparisons: Use your compare_metric_across_years tool first

ROUTING RULES:
- "What was revenue in 2020?" → FY2020 specialist only
- "Compare revenue 2020 vs 2021" → FY2020 + FY2021 specialists
- "Show the revenue trend" → compare_metric_across_years tool, then all specialists

SYNTHESIS RULES:
- When combining data from multiple years, present as a clear comparison table
- Calculate growth rates (YoY %) when comparing years
- Always cite fiscal years for every number
- Values are in millions USD unless stated otherwise"""

        self.conversation_history = []

    def _route_question(self, question):
        q_lower = question.lower()
        mentioned_years = []
        for year in self.available_years:
            if year in q_lower or f"fy{year}" in q_lower or f"fy {year}" in q_lower:
                mentioned_years.append(year)

        trend_keywords = ["trend", "over time", "growth", "compare", "comparison",
                          "year over year", "yoy", "across years", "historical",
                          "evolution", "trajectory", "all years", "each year"]
        is_trend = any(kw in q_lower for kw in trend_keywords)
        vs_pattern = any(kw in q_lower for kw in [" vs ", " versus ", " compared to "])

        if is_trend or (len(mentioned_years) == 0 and vs_pattern):
            return {"years": self.available_years, "strategy": "trend"}
        elif len(mentioned_years) >= 2 or vs_pattern:
            return {"years": mentioned_years if mentioned_years else self.available_years,
                    "strategy": "multi"}
        elif len(mentioned_years) == 1:
            return {"years": mentioned_years, "strategy": "single"}
        else:
            return {"years": [self.available_years[-1]], "strategy": "single"}

    def ask(self, question):
        routing = self._route_question(question)
        years = routing["years"]
        strategy = routing["strategy"]

        specialist_responses = {}
        cross_year_context = ""

        if strategy == "trend":
            for kw in ["revenue", "net sales", "income", "assets", "cash",
                       "debt", "equity", "margin", "earnings", "liabilities"]:
                if kw in question.lower():
                    result = self.own_tools[0].invoke({"metric": kw})
                    cross_year_context += f"\nCross-year data for '{kw}':\n{result}\n"
                    break
            for year in years:
                if year in self.specialists:
                    resp = self.specialists[year].invoke(question)
                    specialist_responses[year] = resp
        elif strategy == "multi":
            for year in years:
                if year in self.specialists:
                    resp = self.specialists[year].invoke(question)
                    specialist_responses[year] = resp
        else:
            year = years[0]
            if year in self.specialists:
                resp = self.specialists[year].invoke(question)
                specialist_responses[year] = resp

        synthesis_prompt = f"""The user asked: \"{question}\"

Routing strategy: {strategy} (years: {years})

Here are the responses from each year's specialist:
"""
        for year, resp in sorted(specialist_responses.items()):
            synthesis_prompt += f"\n--- FY{year} Specialist ---\n{resp}\n"

        if strategy == "trend" and cross_year_context:
            synthesis_prompt += f"\n--- Cross-Year Data ---\n{cross_year_context}\n"

        synthesis_prompt += """
Now synthesize a comprehensive answer:
- If single year: present the specialist's findings clearly
- If multi-year: create a comparison with YoY growth rates
- If trend: show the full trajectory with analysis
- Always cite fiscal years for every number
- Calculate ratios and growth rates where useful
"""

        messages = [
            SystemMessage(content=self.system_prompt),
            *self.conversation_history,
            HumanMessage(content=synthesis_prompt),
        ]

        for _ in range(3):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            if not response.tool_calls:
                break
            for tc in response.tool_calls:
                matching = next((t for t in self.own_tools if t.name == tc["name"]), None)
                if matching:
                    result = matching.invoke(tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        final_answer = response.content if isinstance(response.content, str) else str(response.content)

        self.conversation_history.append(HumanMessage(content=question))
        self.conversation_history.append(AIMessage(content=final_answer))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return final_answer


# ════════════════════════════════════════════════════════════════════════════
# ROUNDTABLE DIALOGUE — Preserved from original
# ════════════════════════════════════════════════════════════════════════════

PERSONALITIES = {
    "2020": {
        "name": "Marcus",
        "style": (
            "You are MARCUS — the veteran. 25 years on the Street. Dry humor, "
            "speaks in short declarative punches. Loves historical context. "
            "Sometimes starts mid-thought as if continuing a conversation in his head. "
            "Occasionally skeptical. Example tone: 'Look, the numbers don't lie. "
            "Forty billion from China and everyone's celebrating — I've seen this movie before.'"
        ),
    },
    "2021": {
        "name": "Priya",
        "style": (
            "You are PRIYA — the growth evangelist. Sharp, fast-talking, always excited "
            "about momentum. Uses vivid metaphors. Tends to interrupt or build on others "
            "with energy: 'Wait, it gets better—' or 'Okay but hold on—'. Speaks like "
            "someone who just found gold in the data. Quick with mental math."
        ),
    },
    "2022": {
        "name": "James",
        "style": (
            "You are JAMES — the contrarian realist. Former risk analyst. Always finds "
            "the crack in the narrative. Calm, measured, almost clinical. Starts sentences "
            "with things like 'Interesting, but...' or 'Nobody's talking about the fact that...' "
            "Loves ratios and margin analysis. Slightly sardonic."
        ),
    },
    "2023": {
        "name": "Sofia",
        "style": (
            "You are SOFIA — the macro thinker. Sees everything through geopolitical and "
            "currency lenses. References trade tensions, FX headwinds, regulatory shifts. "
            "Thoughtful, sometimes pauses before dropping a sharp insight. Speaks like: "
            "'You have to zoom out here...' or 'This isn't just an Apple story, this is a China story.'"
        ),
    },
    "2024": {
        "name": "Derek",
        "style": (
            "You are DEREK — the quant. Everything is a model. Thinks in CAGRs, standard "
            "deviations, and regression. Occasionally teases colleagues for being too qualitative. "
            "Speaks like: 'If you run the numbers...' or 'The three-year CAGR tells a different story.' "
            "Precise, a bit nerdy, but his math lands hard."
        ),
    },
    "2025": {
        "name": "Anika",
        "style": (
            "You are ANIKA — the closer. Youngest analyst, but fearless. Synthesizes what "
            "everyone said and flips it into a forward-looking thesis. Confident, direct, "
            "occasionally challenges a senior colleague's point respectfully. Speaks like: "
            "'Here's what none of us are saying out loud...' or 'So the real question becomes...'"
        ),
    },
}

DEFAULT_PERSONALITIES = [
    {"name": "Chen", "style": "You are CHEN — methodical, data-first, speaks in structured arguments."},
    {"name": "Rachel", "style": "You are RACHEL — the pattern-spotter. Connects dots others miss."},
]


def get_personality(year, index):
    if str(year) in PERSONALITIES:
        return PERSONALITIES[str(year)]
    fb = DEFAULT_PERSONALITIES[index % len(DEFAULT_PERSONALITIES)]
    return fb


def build_prompt(year, question, prior_dialogue, personality, is_first, is_last):
    """Build a natural-dialogue analyst prompt — preserved from original."""
    position = ""
    if is_first:
        position = (
            "You speak FIRST. No one has said anything yet. Set the baseline — "
            "drop a key number and frame the conversation."
        )
    elif is_last:
        position = (
            "You speak LAST before the Chief wraps up. Tie the threads together. "
            "Reference something specific from at least two previous speakers by name. "
            "Deliver your conclusion with conviction."
        )
    else:
        position = (
            "You are mid-conversation. You MUST directly react to what the previous "
            "speaker just said — agree, challenge, or extend their point. Use their NAME. "
            "Then add your own data point. Show how the story evolved in your year."
        )

    prompt = (
        f"{personality['style']}\n\n"
        f"SETTING: Analyst roundtable meeting. You cover FY{year} Apple data.\n\n"
        f"YOUR POSITION: {position}\n\n"
        f"STRICT DIALOGUE RULES:\n"
        f"• Write ONLY your spoken words. 2-4 sentences.\n"
        f"• NO character names, NO labels, NO narration, NO stage directions.\n"
        f"• NO bullet points, NO lists.\n"
        f"• Cite at least one hard number naturally embedded in speech.\n"
        f"• Do NOT start with 'That X percent...' or 'In my FY data...'\n"
        f"• Start differently from every other analyst — vary your opening.\n"
        f"• Sound like a real person in a real meeting. Sentence fragments are fine.\n"
        f"• If doing quick math, say it conversationally: 'which works out to roughly a 12% clip.'\n\n"
        f"CHIEF'S QUESTION: \"{question}\"\n\n"
    )

    if prior_dialogue:
        prompt += f"CONVERSATION SO FAR:\n{prior_dialogue}\n\n"

    prompt += "YOUR LINES:"
    return prompt


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE — Initialized on startup
# ════════════════════════════════════════════════════════════════════════════

specialists = {}
chief = None
sessions = {}  # session_id -> { messages: [], status: "running"|"done"|"error" }


def ingest_filings():
    """Ingest all PDF filings on startup."""
    global structured_store, vector_store, specialists, chief

    print("📁 Initializing stores...")
    structured_store = StructuredFinancialStore(DB_PATH)
    vector_store = VectorTextStore(persist_dir=CHROMA_DIR)

    pdf_files = sorted(glob.glob(str(FILINGS_DIR / "*.pdf")))
    print(f"Found {len(pdf_files)} PDF(s) to ingest")

    ingested_years = []
    for pdf_path in pdf_files:
        print(f"📄 Processing: {Path(pdf_path).name}")
        try:
            ext = TenKTableExtractor(pdf_path, company="AAPL")
            tables = ext.extract_all()
            txt = TenKTextExtractor(pdf_path)
            chunks = txt.chunk_for_embedding()
            table_dicts = [t.to_dict() for t in tables]
            structured_store.store_filing("AAPL", ext.fiscal_year, table_dicts, Path(pdf_path).name)
            vector_store.add_chunks(chunks, "AAPL", ext.fiscal_year)
            ingested_years.append(ext.fiscal_year)
            print(f"   ✅ FY{ext.fiscal_year}: {len(tables)} tables, {len(chunks)} text chunks")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\nIngested years: {sorted(ingested_years)}")

    # Create specialists
    for year in sorted(ingested_years):
        specialists[year] = YearSpecialist(year, structured_store, vector_store)
        print(f"  🤖 FY{year} Specialist created")

    # Create Chief Analyst
    chief = ChiefAnalyst(specialists, structured_store, vector_store)
    print(f"\n✅ AAPL Analyst Office is OPEN!")
    print(f"   Chief Analyst supervising {len(specialists)} year specialists")


def run_roundtable(session_id, question):
    """Run the roundtable dialogue in a background thread."""
    global sessions

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        chief_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.75,
            max_output_tokens=400,
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
        )

        sorted_years = sorted(specialists.keys())
        dialogue_so_far = ""

        # Chief opens with the question
        sessions[session_id]["messages"].append({
            "agentId": "chief",
            "name": "Chief Analyst",
            "year": None,
            "text": question,
            "type": "question",
            "timestamp": time.time()
        })

        # Separator: analyst discussion begins
        sessions[session_id]["messages"].append({
            "agentId": None,
            "name": None,
            "year": None,
            "text": "analyst discussion",
            "type": "separator",
            "timestamp": time.time()
        })

        # Each analyst speaks
        for i, year in enumerate(sorted_years):
            if sessions[session_id].get("cancelled"):
                break

            spec = specialists[year]
            is_first = (i == 0)
            is_last = (i == len(sorted_years) - 1)
            personality = get_personality(year, i)

            prompt = build_prompt(
                year, question, dialogue_so_far,
                personality, is_first, is_last
            )

            # Mark as typing
            sessions[session_id]["typing"] = {
                "agentId": year,
                "name": personality["name"]
            }

            # Call the specialist (uses Claude via LangChain)
            result = spec.invoke(prompt)

            if isinstance(result, str):
                contribution = result.strip()
            elif isinstance(result, dict):
                contribution = result.get("output", result.get("content", str(result))).strip()
            else:
                contribution = str(result).strip()

            # Clean LLM artifacts
            for prefix in [f"FY{year}", f"**FY{year}", "Analyst:", f"{personality['name']}:",
                           f"**{personality['name']}**", f"{personality['name'].upper()}:"]:
                if contribution.lower().startswith(prefix.lower()):
                    contribution = contribution[len(prefix):].strip().lstrip(":").lstrip("*").strip()

            sessions[session_id]["messages"].append({
                "agentId": year,
                "name": personality["name"],
                "year": year,
                "text": contribution,
                "type": "analyst",
                "timestamp": time.time()
            })
            sessions[session_id]["typing"] = None

            dialogue_so_far += f"\n{personality['name']} (FY{year}): {contribution}\n"
            print(f"  ✅ {personality['name']} (FY{year}) spoke.")

        # Separator: chief's conclusion
        sessions[session_id]["messages"].append({
            "agentId": None,
            "name": None,
            "year": None,
            "text": "chief analyst conclusion",
            "type": "separator",
            "timestamp": time.time()
        })

        # Chief wraps up
        sessions[session_id]["typing"] = {
            "agentId": "chief",
            "name": "Chief Analyst"
        }

        chief_prompt = (
            "You are the CHIEF ANALYST closing this meeting.\n"
            "You are authoritative, measured, and sharp. Like a senior partner delivering a verdict.\n\n"
            "RULES:\n"
            "• 3-5 sentences of pure spoken dialogue. No labels, no narration.\n"
            "• Reference at least 2 analysts BY NAME and their numbers.\n"
            "• Identify the inflection point or key trend.\n"
            "• End with one sharp, forward-looking line — the kind that ends a meeting.\n\n"
            f"QUESTION: {question}\n\n"
            f"FULL CONVERSATION:\n{dialogue_so_far}\n\n"
            "YOUR CLOSING WORDS:"
        )

        summary = chief_llm.invoke([
            SystemMessage(content="You are a senior Wall Street chief analyst. Pure dialogue only."),
            HumanMessage(content=chief_prompt),
        ])

        sessions[session_id]["messages"].append({
            "agentId": "chief",
            "name": "Chief Analyst",
            "year": None,
            "text": summary.content.strip(),
            "type": "conclusion",
            "timestamp": time.time()
        })

        sessions[session_id]["typing"] = None
        sessions[session_id]["status"] = "done"
        print(f"  ✅ Roundtable complete for session {session_id}")

    except Exception as e:
        print(f"  ❌ Roundtable error: {e}")
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        sessions[session_id]["typing"] = None


# ════════════════════════════════════════════════════════════════════════════
# FLASK API ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "specialists": list(specialists.keys()),
        "ready": len(specialists) > 0
    })


@app.route("/api/analysts", methods=["GET"])
def get_analysts():
    """Return the list of available analysts with their personalities."""
    analysts = []
    for year in sorted(specialists.keys()):
        p = get_personality(year, sorted(specialists.keys()).index(year))
        analysts.append({
            "year": year,
            "name": p["name"],
            "style": p["style"][:100] + "..."
        })
    return jsonify({"analysts": analysts})


@app.route("/api/roundtable", methods=["POST"])
def start_roundtable():
    """Start a new roundtable session with a question."""
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if not specialists:
        return jsonify({"error": "System not ready. No specialists loaded."}), 503

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "messages": [],
        "status": "running",
        "typing": None,
        "question": question,
        "created_at": time.time()
    }

    # Run in background thread
    thread = threading.Thread(target=run_roundtable, args=(session_id, question))
    thread.daemon = True
    thread.start()

    return jsonify({"session_id": session_id})


@app.route("/api/roundtable/<session_id>", methods=["GET"])
def poll_roundtable(session_id):
    """Poll for new messages in a roundtable session."""
    if session_id not in sessions:
        return jsonify({"error": "Session not found"}), 404

    session = sessions[session_id]
    after = int(request.args.get("after", 0))

    return jsonify({
        "status": session["status"],
        "typing": session.get("typing"),
        "messages": session["messages"][after:],
        "total": len(session["messages"]),
        "error": session.get("error")
    })


# ════════════════════════════════════════════════════════════════════════════
# STARTUP
# ════════════════════════════════════════════════════════════════════════════

# Ingest filings on startup
print("🚀 Starting AAPL Analyst Office...")
ingest_filings()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
