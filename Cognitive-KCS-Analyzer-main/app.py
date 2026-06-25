import streamlit as st
import pandas as pd
import io

from data_loader import load_csv_data
from llm_interactions import get_llm_completion
import prompts
from chatbot import render_chat_tab


def find_column(df, candidates):
    """Return the first matching column name from candidates, or None."""
    if df is None:
        return None
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_cols:
            return lower_cols[candidate.lower()]
    return None


# --- Page Configuration ---
st.set_page_config(page_title="Cognitive KCS Analyzer", layout="wide")

st.title("🧠 Cognitive KCS Analyzer")
st.caption("LLM-Powered KCS Article Quality & Gap Analysis")

# --- Session State ---
if "articles_df" not in st.session_state:
    st.session_state.articles_df = None
if "tickets_df" not in st.session_state:
    st.session_state.tickets_df = None

# --- Sidebar ---
st.sidebar.header("Upload Data")
uploaded_articles_file = st.sidebar.file_uploader("Upload KCS Articles CSV", type="csv")
uploaded_tickets_file  = st.sidebar.file_uploader("Upload Support Tickets CSV", type="csv")

if uploaded_articles_file:
    stringio_articles = io.StringIO(uploaded_articles_file.getvalue().decode("utf-8"))
    st.session_state.articles_df = pd.read_csv(stringio_articles)
    st.sidebar.success(f"KCS Articles loaded: {st.session_state.articles_df.shape[0]} rows")
    with st.expander("Preview KCS Articles Data"):
        st.dataframe(st.session_state.articles_df.head())

if uploaded_tickets_file:
    stringio_tickets = io.StringIO(uploaded_tickets_file.getvalue().decode("utf-8"))
    st.session_state.tickets_df = pd.read_csv(stringio_tickets)
    st.sidebar.success(f"Support Tickets loaded: {st.session_state.tickets_df.shape[0]} rows")
    with st.expander("Preview Support Tickets Data"):
        st.dataframe(st.session_state.tickets_df.head())

st.sidebar.markdown("---")
st.sidebar.caption("Cognitive KCS Analyzer")


# --- Column name candidates updated to match your actual CSVs ---
# Articles (Customer_Query_Log_with_Answers.csv): Query ID | Customer Query | Category | Answer
# Tickets  (Customer_Query_Log.csv):              Query ID | Customer Query | Category

ARTICLE_ID_CANDIDATES  = ['Query ID', 'query_id', 'Article_id', 'article_id', 'id']
TICKET_ID_CANDIDATES   = ['Query ID', 'ticket_id', 'Ticket ID', 'id']
TITLE_CANDIDATES       = ['Category', 'title', 'Title', 'article_title', 'subject']
CONTENT_CANDIDATES     = ['Answer', 'Customer Query', 'content', 'Content', 'description']
QUERY_SUMMARY_CANDIDATES = ['Customer Query', 'query_summary', 'query', 'description', 'summary']


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Article Quality",
    "🔍 Knowledge Gaps",
    "💡 Improvement Suggestions",
    "🤖 KB Chatbot",
])

# ── Tab 1: Article Quality ──────────────────────────────────────────────────
with tab1:
    st.header("KCS Article Quality Assessment")
    if st.session_state.articles_df is not None:
        article_id_col = find_column(st.session_state.articles_df, ARTICLE_ID_CANDIDATES)
        title_col      = find_column(st.session_state.articles_df, TITLE_CANDIDATES)
        content_col    = find_column(st.session_state.articles_df, CONTENT_CANDIDATES)

        if article_id_col is None:
            st.error(f"Cannot find an ID column. Detected columns: {st.session_state.articles_df.columns.tolist()}")
        elif title_col is None or content_col is None:
            st.error(f"Cannot find title/content columns. Detected columns: {st.session_state.articles_df.columns.tolist()}")
        else:
            article_ids = st.session_state.articles_df[article_id_col].astype(str).tolist()
            selected_article_id = st.selectbox("Select Article ID to Assess:", article_ids)

            if selected_article_id and st.button("Assess Quality", key="quality_btn"):
                article = st.session_state.articles_df[
                    st.session_state.articles_df[article_id_col].astype(str) == str(selected_article_id)
                ].iloc[0]
                st.subheader(f"Assessing: {article[title_col]}")
                st.markdown("**Content:**")
                st.text_area("Article Content:", article[content_col], height=150, disabled=True)

                with st.spinner("Asking LLM to assess quality..."):
                    prompt = prompts.get_article_quality_assessment_prompt(
                        article[title_col], article[content_col])
                    assessment = get_llm_completion(prompt, max_tokens=500)
                st.markdown("---")
                st.subheader("LLM Quality Assessment:")
                st.markdown(assessment)
    else:
        st.info("Upload KCS Articles CSV (the one with the Answer column) to begin quality assessment.")


# ── Tab 2: Knowledge Gaps ───────────────────────────────────────────────────
with tab2:
    st.header("Potential Knowledge Gap Identification")
    if st.session_state.tickets_df is not None and st.session_state.articles_df is not None:
        ticket_id_col    = find_column(st.session_state.tickets_df,  TICKET_ID_CANDIDATES)
        query_col        = find_column(st.session_state.tickets_df,  QUERY_SUMMARY_CANDIDATES)
        title_col        = find_column(st.session_state.articles_df, TITLE_CANDIDATES)

        if ticket_id_col is None:
            st.error(f"Cannot find ticket ID column. Detected columns: {st.session_state.tickets_df.columns.tolist()}")
        elif query_col is None:
            st.error(f"Cannot find query/summary column. Detected columns: {st.session_state.tickets_df.columns.tolist()}")
        else:
            ticket_ids = st.session_state.tickets_df[ticket_id_col].astype(str).tolist()
            selected_ticket_id = st.selectbox("Select Ticket ID to Analyze for Gaps:", ticket_ids)

            if selected_ticket_id and st.button("Analyze for Gap", key="gap_btn"):
                ticket = st.session_state.tickets_df[
                    st.session_state.tickets_df[ticket_id_col].astype(str) == str(selected_ticket_id)
                ].iloc[0]
                st.subheader(f"Analyzing Ticket: {ticket[query_col]}")

                existing_article_titles = st.session_state.articles_df[title_col].tolist()
                with st.spinner("Asking LLM to identify gaps..."):
                    prompt = prompts.get_knowledge_gap_identification_prompt(
                        ticket[query_col], existing_article_titles)
                    gap_analysis = get_llm_completion(prompt, max_tokens=300)
                st.markdown("---")
                st.subheader("LLM Gap Analysis:")
                st.markdown(gap_analysis)
    else:
        st.info("Upload both CSVs to identify knowledge gaps.")


# ── Tab 3: Improvement Suggestions ─────────────────────────────────────────
with tab3:
    st.header("Article Improvement Suggestions")
    if st.session_state.articles_df is not None and st.session_state.tickets_df is not None:
        st.write("Select an article and a related ticket to get improvement suggestions.")

        article_id_col = find_column(st.session_state.articles_df, ARTICLE_ID_CANDIDATES)
        ticket_id_col  = find_column(st.session_state.tickets_df,  TICKET_ID_CANDIDATES)
        title_col      = find_column(st.session_state.articles_df, TITLE_CANDIDATES)
        content_col    = find_column(st.session_state.articles_df, CONTENT_CANDIDATES)
        query_col      = find_column(st.session_state.tickets_df,  QUERY_SUMMARY_CANDIDATES)

        missing = []
        if article_id_col is None: missing.append("article ID column")
        if ticket_id_col  is None: missing.append("ticket ID column")
        if title_col      is None: missing.append("title/category column")
        if content_col    is None: missing.append("content/answer column")
        if query_col      is None: missing.append("query/summary column")

        if missing:
            st.error(f"Cannot find: {', '.join(missing)}. "
                     f"Articles columns: {st.session_state.articles_df.columns.tolist()} | "
                     f"Tickets columns: {st.session_state.tickets_df.columns.tolist()}")
        else:
            article_ids_improve = st.session_state.articles_df[article_id_col].astype(str).tolist()
            selected_article_id_improve = st.selectbox(
                "Select Article to Improve:", article_ids_improve, key="sel_art_imp")

            ticket_ids_improve = st.session_state.tickets_df[ticket_id_col].astype(str).tolist()
            selected_ticket_id_improve = st.selectbox(
                "Select Related Ticket:", ticket_ids_improve, key="sel_tkt_imp")

            if selected_article_id_improve and selected_ticket_id_improve and \
               st.button("Suggest Improvements", key="improve_btn"):
                article = st.session_state.articles_df[
                    st.session_state.articles_df[article_id_col].astype(str) == str(selected_article_id_improve)
                ].iloc[0]
                ticket = st.session_state.tickets_df[
                    st.session_state.tickets_df[ticket_id_col].astype(str) == str(selected_ticket_id_improve)
                ].iloc[0]

                st.subheader(f"Article: {article[title_col]}")
                st.markdown(f"**Related Ticket Query:** {ticket[query_col]}")

                with st.spinner("Asking LLM for improvement suggestions..."):
                    prompt = prompts.get_article_improvement_prompt(
                        article[title_col], article[content_col], ticket[query_col])
                    improvement_suggestion = get_llm_completion(prompt, max_tokens=400)
                st.markdown("---")
                st.subheader("LLM Improvement Suggestion:")
                st.markdown(improvement_suggestion)
    else:
        st.info("Upload both CSVs for improvement suggestions.")


# ── Tab 4: KB Chatbot ───────────────────────────────────────────────────────
with tab4:
    render_chat_tab(
        articles_df=st.session_state.articles_df,
        tickets_df=st.session_state.tickets_df,
    )