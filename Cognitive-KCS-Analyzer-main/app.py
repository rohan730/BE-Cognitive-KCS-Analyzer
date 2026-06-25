import streamlit as st
import pandas as pd
from data_loader import load_csv_data # Assuming it can take UploadedFile
from llm_interactions import get_llm_completion
import prompts
import io # Required to read uploaded file with pandas

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

# --- Global state (if needed, or use Streamlit session state) ---
if 'articles_df' not in st.session_state:
    st.session_state.articles_df = None
if 'tickets_df' not in st.session_state:
    st.session_state.tickets_df = None

# --- Sidebar for Data Upload ---
st.sidebar.header("Upload Data")
uploaded_articles_file = st.sidebar.file_uploader("Upload KCS Articles CSV", type="csv")
uploaded_tickets_file = st.sidebar.file_uploader("Upload Support Tickets CSV", type="csv")

if uploaded_articles_file:
    # To read CSV from UploadedFile object
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


# --- Main Application Tabs ---
tab1, tab2, tab3 = st.tabs(["📖 Article Quality", "🔍 Knowledge Gaps", "💡 Improvement Suggestions"])

with tab1:
    st.header("KCS Article Quality Assessment")
    if st.session_state.articles_df is not None:
        article_id_col = find_column(st.session_state.articles_df, ['Article_id', 'article_id', 'Article ID', 'article id', 'id', 'Query ID', 'query_id', 'query id'])
        title_col = find_column(st.session_state.articles_df, ['title', 'Title', 'article_title', 'Article Title', 'Category', 'subject'])
        content_col = find_column(st.session_state.articles_df, ['content', 'Content', 'article_content', 'Article Content', 'Customer Query', 'query', 'description'])
        if article_id_col is None:
            st.error("Uploaded KCS Articles CSV does not contain a recognized article ID column. Please include one of: Article_id, article_id, Article ID, or Query ID.")
        elif title_col is None or content_col is None:
            st.error("Uploaded KCS Articles CSV does not contain recognized title/content columns. Please include fields like title and content or Customer Query.")
        else:
            article_ids = st.session_state.articles_df[article_id_col].astype(str).tolist()
            selected_article_id = st.selectbox("Select Article ID to Assess:", article_ids)

            if selected_article_id and st.button("Assess Quality", key="quality_btn"):
                article = st.session_state.articles_df[st.session_state.articles_df[article_id_col].astype(str) == str(selected_article_id)].iloc[0]
                st.subheader(f"Assessing: {article[title_col]}")
                st.markdown("**Content:**")
                st.text_area("Article Content:", article[content_col], height=150, disabled=True)

                with st.spinner("Asking LLM to assess quality..."):
                    prompt = prompts.get_article_quality_assessment_prompt(article[title_col], article[content_col])
                    assessment = get_llm_completion(prompt, max_tokens=500)
                st.markdown("---")
                st.subheader("LLM Quality Assessment:")
                st.markdown(assessment) # LLM output often has markdown, so st.markdown is good
    else:
        st.info("Upload KCS articles CSV to begin quality assessment.")

with tab2:
    st.header("Potential Knowledge Gap Identification")
    if st.session_state.tickets_df is not None and st.session_state.articles_df is not None:
        ticket_id_col = find_column(st.session_state.tickets_df, ['ticket_id', 'Ticket_id', 'Ticket ID', 'ticket id', 'id'])
        title_col = find_column(st.session_state.articles_df, ['title', 'Title', 'article_title', 'Article Title', 'Category', 'subject'])
        if ticket_id_col is None:
            st.error("Uploaded support tickets CSV does not contain a recognized ticket ID column. Please include one of: ticket_id, Ticket ID, or id.")
        else:
            ticket_ids = st.session_state.tickets_df[ticket_id_col].astype(str).tolist()
            selected_ticket_id = st.selectbox("Select Ticket ID to Analyze for Gaps:", ticket_ids)

            if selected_ticket_id and st.button("Analyze for Gap", key="gap_btn"):
                ticket = st.session_state.tickets_df[st.session_state.tickets_df[ticket_id_col].astype(str) == str(selected_ticket_id)].iloc[0]
                st.subheader(f"Analyzing Ticket: {ticket['query_summary']}")
                
                existing_article_titles = st.session_state.articles_df[title_col].tolist()
                with st.spinner("Asking LLM to identify gaps..."):
                    prompt = prompts.get_knowledge_gap_identification_prompt(ticket['query_summary'], existing_article_titles)
                    gap_analysis = get_llm_completion(prompt, max_tokens=300)
                st.markdown("---")
                st.subheader("LLM Gap Analysis:")
                st.markdown(gap_analysis)
    else:
        st.info("Upload both KCS articles and Support Tickets CSVs to identify knowledge gaps.")

with tab3:
    st.header("Article Improvement Suggestions")
    if st.session_state.articles_df is not None and st.session_state.tickets_df is not None:
        st.write("Select an article and a related ticket to get improvement suggestions.")
        article_id_col = find_column(st.session_state.articles_df, ['Article_id', 'article_id', 'Article ID', 'article id', 'id', 'Query ID', 'query_id', 'query id'])
        ticket_id_col = find_column(st.session_state.tickets_df, ['ticket_id', 'Ticket_id', 'Ticket ID', 'ticket id', 'id'])
        title_col = find_column(st.session_state.articles_df, ['title', 'Title', 'article_title', 'Article Title', 'Category', 'subject'])
        content_col = find_column(st.session_state.articles_df, ['content', 'Content', 'article_content', 'Article Content', 'Customer Query', 'query', 'description'])

        if article_id_col is None or ticket_id_col is None or title_col is None or content_col is None:
            st.error("Uploaded CSVs are missing required fields for article improvement. Please ensure article and ticket files include ID, title, and content columns.")
        else:
            article_ids_improve = st.session_state.articles_df[article_id_col].astype(str).tolist()
            selected_article_id_improve = st.selectbox("Select Article to Improve:", article_ids_improve, key="sel_art_imp")
            ticket_ids_improve = st.session_state.tickets_df[ticket_id_col].astype(str).tolist()
            selected_ticket_id_improve = st.selectbox("Select Related Ticket:", ticket_ids_improve, key="sel_tkt_imp")

            if selected_article_id_improve and selected_ticket_id_improve and st.button("Suggest Improvements", key="improve_btn"):
                article = st.session_state.articles_df[st.session_state.articles_df[article_id_col].astype(str) == str(selected_article_id_improve)].iloc[0]
                ticket = st.session_state.tickets_df[st.session_state.tickets_df[ticket_id_col].astype(str) == str(selected_ticket_id_improve)].iloc[0]

                st.subheader(f"Article: {article[title_col]}")
                st.markdown(f"**Related Ticket Query:** {ticket['query_summary']}")

                with st.spinner("Asking LLM for improvement suggestions..."):
                    prompt = prompts.get_article_improvement_prompt(article[title_col], article[content_col], ticket['query_summary'])
                    improvement_suggestion = get_llm_completion(prompt, max_tokens=400)
                st.markdown("---")
                st.subheader("LLM Improvement Suggestion:")
                st.markdown(improvement_suggestion)
    else:
        st.info("Upload both KCS articles and Support Tickets CSVs for improvement suggestions.")

st.sidebar.markdown("---")
st.sidebar.caption("Project by [Your Name]")
