import streamlit as st
import pandas as pd
from llm_interactions import get_llm_completion


# ── helpers ────────────────────────────────────────────────────────────────────

def _df_summary(df: pd.DataFrame, label: str) -> str:
    """Return a compact text snapshot of a dataframe for the LLM context."""
    if df is None or df.empty:
        return f"No {label} data available."
    rows = min(len(df), 50)          # cap at 50 rows to stay within token budget
    sample = df.head(rows).to_csv(index=False)
    return (
        f"=== {label} ({len(df)} total rows, showing first {rows}) ===\n"
        f"Columns: {', '.join(df.columns.tolist())}\n\n"
        f"{sample}"
    )


def build_system_context(articles_df, tickets_df) -> str:
    """Build a rich system prompt that embeds the loaded data."""
    articles_block = _df_summary(articles_df, "KCS Articles")
    tickets_block  = _df_summary(tickets_df,  "Support Tickets")

    return f"""You are a KCS Knowledge Assistant embedded in the Cognitive KCS Analyzer tool.
You have full access to the user's uploaded knowledge-base articles and support tickets (shown below).
Answer every question by reasoning over this data. Be specific — reference article IDs, titles, ticket IDs,
and exact values from the tables when relevant. If a question cannot be answered from the data, say so clearly.

Capabilities you can demonstrate:
- List, filter, or summarise articles / tickets on request
- Identify which articles cover a given topic
- Spot articles with low quality signals (e.g. very short content, missing fields)
- Find tickets whose queries are not covered by any article (knowledge gaps)
- Compare two articles or two tickets side-by-side
- Answer "how many …" counting questions
- Suggest which article is most relevant to a customer query

{articles_block}

{tickets_block}
"""


def _history_to_messages(history: list[dict]) -> list[str]:
    """
    Convert chat history into a single formatted string for the Gemini `contents` field.
    Gemini's google-genai SDK (non-chat session path) expects a plain string or list of parts.
    We concatenate the conversation so the model sees full context.
    """
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def get_chatbot_response(user_message: str,
                         chat_history: list[dict],
                         articles_df,
                         tickets_df) -> str:
    """
    Build a prompt that fuses system context + conversation history + new user message,
    then call get_llm_completion (which already wraps the Gemini client).
    """
    system_ctx  = build_system_context(articles_df, tickets_df)
    history_str = _history_to_messages(chat_history[:-1])   # exclude the latest user turn

    full_prompt = f"""{system_ctx}

--- Conversation so far ---
{history_str}

--- New user message ---
User: {user_message}
Assistant:"""

    return get_llm_completion(full_prompt, max_tokens=800, temperature=0.3)


# ── Streamlit UI ───────────────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "List all article titles and their IDs",
    "Which articles have the shortest content?",
    "Show me all support tickets",
    "Are there any tickets not covered by existing articles?",
    "Which article is most relevant to password reset issues?",
    "How many articles and tickets are loaded?",
]


def render_chat_tab(articles_df, tickets_df):
    """
    Drop this into your app.py tab.  Call as:
        from chatbot import render_chat_tab
        with tab4:
            render_chat_tab(st.session_state.articles_df, st.session_state.tickets_df)
    """
    # ── session state init ──────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input_key" not in st.session_state:
        st.session_state.chat_input_key = 0

    # ── guard: need at least one dataset ───────────────────────────────────
    data_ready = articles_df is not None or tickets_df is not None
    if not data_ready:
        st.info("Upload at least one CSV (articles or tickets) from the sidebar to start chatting.")
        return

    # ── header row ─────────────────────────────────────────────────────────
    col_title, col_clear = st.columns([5, 1])
    with col_title:
        st.markdown("### 💬 Ask anything about your KB data")
    with col_clear:
        if st.button("🗑 Clear", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.chat_input_key += 1
            st.rerun()

    # ── data status badges ──────────────────────────────────────────────────
    badge_cols = st.columns(2)
    with badge_cols[0]:
        if articles_df is not None:
            st.success(f"✅ {len(articles_df)} articles loaded")
        else:
            st.warning("⚠️ No articles loaded")
    with badge_cols[1]:
        if tickets_df is not None:
            st.success(f"✅ {len(tickets_df)} tickets loaded")
        else:
            st.warning("⚠️ No tickets loaded")

    st.markdown("---")

    # ── suggested questions (only when history is empty) ───────────────────
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        cols = st.columns(3)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 3]:
                if st.button(q, key=f"sugg_{i}"):
                    _submit_message(q, articles_df, tickets_df)
                    st.rerun()

    # ── chat history ────────────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ── input box ───────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask about articles, tickets, gaps, quality…",
        key=f"chat_input_{st.session_state.chat_input_key}"
    )
    if user_input:
        _submit_message(user_input, articles_df, tickets_df)
        st.rerun()


def _submit_message(user_text: str, articles_df, tickets_df):
    """Append user msg, call LLM, append assistant response."""
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    with st.spinner("Thinking…"):
        reply = get_chatbot_response(
            user_message=user_text,
            chat_history=st.session_state.chat_history,
            articles_df=articles_df,
            tickets_df=tickets_df,
        )

    st.session_state.chat_history.append({"role": "assistant", "content": reply})