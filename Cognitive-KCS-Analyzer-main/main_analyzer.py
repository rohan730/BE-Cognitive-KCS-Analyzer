from data_loader import load_csv_data
from llm_interactions import get_llm_completion
import prompts # To access the prompt generation functions

def analyze_article_quality(article_df):
    """Analyzes the quality of each KCS article."""
    if article_df is None or article_df.empty:
        print("No articles to analyze.")
        return []

    results = []
    print("\n--- Analyzing KCS Article Quality ---")
    # Let's analyze only the first 2 articles for brevity in this example
    for index, row in article_df.head(2).iterrows():
        article_id = row['article_id']
        title = row['title']
        content = row['content']

        print(f"\nAssessing Article ID: {article_id} - \"{title}\"")
        assessment_prompt = prompts.get_article_quality_assessment_prompt(title, content)
        quality_assessment = get_llm_completion(assessment_prompt, max_tokens=500) # Increased max_tokens for detailed assessment

        print("LLM Quality Assessment:")
        print(quality_assessment)
        results.append({
            "article_id": article_id,
            "title": title,
            "quality_assessment": quality_assessment
        })
    return results

def identify_knowledge_gaps(tickets_df, articles_df):
    """Identifies potential knowledge gaps based on new tickets."""
    if tickets_df is None or tickets_df.empty:
        print("No tickets to analyze for gaps.")
        return []
    if articles_df is None or articles_df.empty:
        print("No articles provided to compare against for gap analysis.")
        return []

    existing_article_titles = articles_df['title'].tolist()
    results = []
    print("\n--- Identifying Potential Knowledge Gaps ---")
    # Let's analyze only the first 2 tickets for brevity
    for index, row in tickets_df.head(2).iterrows():
        ticket_id = row['ticket_id']
        query_summary = row['query_summary']

        print(f"\nAnalyzing Ticket ID: {ticket_id} - \"{query_summary}\"")
        gap_prompt = prompts.get_knowledge_gap_identification_prompt(query_summary, existing_article_titles)
        gap_analysis = get_llm_completion(gap_prompt, max_tokens=300)

        print("LLM Gap Analysis:")
        print(gap_analysis)
        results.append({
            "ticket_id": ticket_id,
            "query_summary": query_summary,
            "gap_analysis": gap_analysis
        })
    return results

def suggest_article_improvements(ticket_query, articles_df):
    """
    For a given ticket query, find the most relevant article (e.g., via simple keyword match or embedding similarity - simplified here)
    and then ask LLM to suggest improvements.
    For this example, we'll manually pick an article and ticket for demonstration.
    """
    if articles_df is None or articles_df.empty:
        print("No articles available for improvement suggestions.")
        return None

    print("\n--- Suggesting Article Improvements ---")

    # Example: Ticket T006 ("Password reset email not arriving") vs Article KCS001 ("How to Reset Password")
    # In a real app, you'd have a mechanism to match tickets to relevant articles.
    # For this demo, we hardcode one example.
    target_article_id = "KCS001"
    related_ticket_summary = "Password reset email not arriving in inbox" # From T006

    article_row = articles_df[articles_df['article_id'] == target_article_id].iloc[0]
    article_title = article_row['title']
    article_content = article_row['content']

    print(f"\nSuggesting improvements for Article ID: {target_article_id} - \"{article_title}\"")
    print(f"Based on related query: \"{related_ticket_summary}\"")

    improvement_prompt = prompts.get_article_improvement_prompt(article_title, article_content, related_ticket_summary)
    improvement_suggestion = get_llm_completion(improvement_prompt, max_tokens=400)

    print("LLM Improvement Suggestion:")
    print(improvement_suggestion)
    return {
        "article_id": target_article_id,
        "related_query": related_ticket_summary,
        "improvement_suggestion": improvement_suggestion
    }


if __name__ == "__main__":
    articles_df = load_csv_data('sample_data/kcs_articles.csv')
    tickets_df = load_csv_data('sample_data/support_tickets.csv')

    if articles_df is not None:
        quality_results = analyze_article_quality(articles_df)
        # print("\nFull Quality Results (raw):", quality_results) # For debugging

    if tickets_df is not None and articles_df is not None:
        gap_results = identify_knowledge_gaps(tickets_df, articles_df)
        # print("\nFull Gap Results (raw):", gap_results) # For debugging

    if articles_df is not None:
        # For improvement suggestion, we are using a specific example.
        # In a real app, you might iterate through tickets that were 'partially matched' by the gap analysis.
        improvement_result = suggest_article_improvements(
            "Password reset email not arriving in inbox", # Example query
            articles_df
        )
        # print("\nFull Improvement Suggestion (raw):", improvement_result) # For debugging
    
    print("\n--- Analysis Complete ---")
    print("Review the printed outputs above for LLM responses.")
    print("Consider adding more structured parsing of LLM responses for production use.")
