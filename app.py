"""
app.py — AI Text Summariser (Streamlit Web App)
================================================
Features:
  - 4 Summarisation modes
  - PDF Upload
  - Compare All Modes side by side
  - Keyword Extractor
  - Download summary
Powered by Google Gemini API (Free)
Run with:  streamlit run app.py
"""

import re
import time
import io
import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Text Summariser",
    page_icon="",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 14px; line-height: 1.6; }
    .summary-box {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        font-size: 14px;
        line-height: 1.75;
        color: #14532d;
        margin-top: 0.5rem;
        min-height: 120px;
    }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value { font-size: 26px; font-weight: 700; color: #1e293b; }
    .stat-label { font-size: 12px; color: #64748b; margin-top: 2px; }
    .mode-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 12px;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        margin-bottom: 6px;
    }
    .keyword-badge {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        font-size: 13px;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        margin: 3px;
    }
    .compare-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        height: 100%;
    }
    .compare-title {
        font-size: 13px;
        font-weight: 700;
        color: #1d4ed8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .compare-text {
        font-size: 13px;
        color: #334155;
        line-height: 1.65;
    }
</style>
""", unsafe_allow_html=True)

# ── Paste your Gemini API key here ───────────────────────────────────────────

# ── Utility functions ─────────────────────────────────────────────────────────

def count_words(text: str) -> int:
    return len(re.findall(r'\S+', text))

def compute_compression(original: int, summary: int) -> float:
    if original == 0:
        return 0.0
    return (1 - summary / original) * 100

def extract_pdf_text(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file."""
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def call_gemini(prompt: str) -> str:
    """Call Gemini API and return response text."""
    # Try secrets first, then session state (from sidebar input)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        api_key = ""
    if not api_key:
        api_key = st.session_state.get("manual_api_key", "")
    if not api_key:
        raise ValueError("Please enter your Gemini API key in the sidebar.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPTS = {
    "Concise": (
        "Summarise the following text in at most {max_words} words. "
        "Be concise and dense. Return ONLY the summary, no preamble.\n\n{text}"
    ),
    "Bullet Points": (
        "Extract the key points from the following text as a bullet list. "
        "Use at most {max_words} total words. Start each bullet with '• '. "
        "Return ONLY the bullets.\n\n{text}"
    ),
    "Simple (ELI5)": (
        "Explain the following text in very simple language a 12-year-old can understand. "
        "At most {max_words} words. Avoid jargon. Return ONLY the explanation.\n\n{text}"
    ),
    "Technical": (
        "Write a technical summary preserving domain-specific terminology. "
        "At most {max_words} words. Return ONLY the summary.\n\n{text}"
    ),
}

KEYWORD_PROMPT = (
    "Extract exactly 8 important keywords or key phrases from the following text. "
    "Return ONLY a comma-separated list of keywords, nothing else. "
    "Example format: artificial intelligence, machine learning, neural network\n\n{text}"
)

SAMPLE_TEXT = """Artificial intelligence (AI) is intelligence demonstrated by machines,
as opposed to the natural intelligence displayed by humans. AI research has been defined
as the field of study of intelligent agents, which refers to any system that perceives
its environment and takes actions that maximize its chance of achieving its goals.
AI applications include advanced web search engines, recommendation systems,
understanding human speech, self-driving cars, generative tools, and competing at
the highest level in strategic games. As machines become increasingly capable,
tasks considered to require intelligence are often removed from the definition of AI —
a phenomenon known as the AI effect. The general problem of simulating intelligence
has been broken down into sub-problems which consist of particular traits or
capabilities that researchers expect an intelligent system to display, such as
reasoning, knowledge representation, planning, learning, and perception."""


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("AI Text Summariser")
    st.markdown("Summarise any text instantly using **AI**. Supports text input, PDF upload, keyword extraction, and mode comparison.")
    st.divider()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Settings")

        # API key input (used if secrets not configured)
        try:
            has_secret = bool(st.secrets.get("GEMINI_API_KEY", ""))
        except Exception:
            has_secret = False

        if not has_secret:
            manual_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Enter your Gemini API key from aistudio.google.com"
            )
            st.session_state["manual_api_key"] = manual_key
        else:
            st.success("API Key loaded")

        st.divider()

        mode = st.selectbox(
            "Summarisation Mode",
            list(PROMPTS.keys()),
        )

        max_words = st.slider(
            "Max words in summary",
            min_value=30, max_value=250, value=100, step=10,
        )

        st.divider()
        st.markdown("**Mode descriptions:**")
        st.markdown("1. **Concise** — short & dense")
        st.markdown("2. **Bullet Points** — key ideas as list")
        st.markdown("3. **Simple** — no jargon, easy read")
        st.markdown("4. **Technical** — keeps terminology")

        st.divider()
        st.markdown("**Extra Features:**")
        do_keywords = st.checkbox("Extract Keywords", value=True)
        do_compare  = st.checkbox("Compare All Modes", value=False)

    # ── Input Section ──
    st.subheader("Input")

    # Tabs for text input vs PDF upload
    tab1, tab2 = st.tabs(["Type / Paste Text", "Upload PDF"])

    input_text = ""

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Load sample", use_container_width=True):
                st.session_state["input_text"] = SAMPLE_TEXT

        input_text = st.text_area(
            label="input",
            label_visibility="collapsed",
            value=st.session_state.get("input_text", ""),
            placeholder="Paste your article, paragraph, or any text here...",
            height=200,
            key="input_text",
        )

    with tab2:
        uploaded_file = st.file_uploader(
            "Upload a PDF file", type=["pdf"],
            help="The app will extract text from the PDF and summarise it"
        )
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                try:
                    input_text = extract_pdf_text(uploaded_file)
                    st.success(f"PDF text extracted! ({count_words(input_text)} words found)")
                    with st.expander("Preview extracted text"):
                        st.text(input_text[:1000] + ("..." if len(input_text) > 1000 else ""))
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

    # Word count
    word_count = count_words(input_text)
    st.caption(f"Word count: **{word_count}**")

    # ── Summarise Button ──
    clicked = st.button("Summarise", type="primary", use_container_width=True)

    if clicked:
        if word_count < 20:
            st.warning(f"Please enter at least 20 words. You have {word_count}.")
            st.stop()

        st.divider()

        # ════════════════════════════════════
        # FEATURE 1 — Normal Summary
        # ════════════════════════════════════
        st.subheader("Summary")

        with st.spinner("Generating summary..."):
            try:
                start = time.perf_counter()
                prompt = PROMPTS[mode].format(max_words=max_words, text=input_text.strip())
                summary = call_gemini(prompt)
                latency_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.markdown(f'<div class="mode-badge">{mode}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        # Stats
        st.markdown("<br>", unsafe_allow_html=True)
        summary_words = count_words(summary)
        compression   = compute_compression(word_count, summary_words)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{word_count}</div><div class="stat-label">Original words</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{summary_words}</div><div class="stat-label">Summary words</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{compression:.0f}%</div><div class="stat-label">Compression</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card"><div class="stat-value">{latency_ms:.0f}ms</div><div class="stat-label">Response time</div></div>', unsafe_allow_html=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download summary as .txt",
            data=summary,
            file_name="summary.txt",
            mime="text/plain",
        )

        # ════════════════════════════════════
        # FEATURE 2 — Keyword Extractor
        # ════════════════════════════════════
        if do_keywords:
            st.divider()
            st.subheader("Keywords Extracted")

            with st.spinner("Extracting keywords..."):
                try:
                    kw_prompt  = KEYWORD_PROMPT.format(text=input_text.strip())
                    kw_result  = call_gemini(kw_prompt)
                    keywords   = [k.strip() for k in kw_result.split(",") if k.strip()]
                except Exception as e:
                    keywords = []
                    st.error(f"Keyword error: {e}")

            if keywords:
                kw_html = " ".join(
                    f'<span class="keyword-badge">{kw}</span>' for kw in keywords
                )
                st.markdown(kw_html, unsafe_allow_html=True)

        # ════════════════════════════════════
        # FEATURE 3 — Compare All Modes
        # ════════════════════════════════════
        if do_compare:
            st.divider()
            st.subheader("⚖️ Compare All Modes")
            st.caption("Generating summaries in all 4 modes — this may take a few seconds...")

            results = {}
            progress = st.progress(0)

            for i, (mode_name, mode_prompt) in enumerate(PROMPTS.items()):
                with st.spinner(f"Generating {mode_name}..."):
                    try:
                        p = mode_prompt.format(max_words=max_words, text=input_text.strip())
                        results[mode_name] = call_gemini(p)
                    except Exception as e:
                        results[mode_name] = f"Error: {e}"
                    progress.progress((i + 1) / len(PROMPTS))
                    time.sleep(1)  # avoid rate limit

            progress.empty()

            # Display in 2x2 grid
            col_a, col_b = st.columns(2)
            mode_list = list(results.items())

            for idx, (mode_name, mode_summary) in enumerate(mode_list):
                col = col_a if idx % 2 == 0 else col_b
                with col:
                    st.markdown(f"""
                    <div class="compare-card">
                        <div class="compare-title">{mode_name}</div>
                        <div class="compare-text">{mode_summary}</div>
                    </div>
                    <br>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
