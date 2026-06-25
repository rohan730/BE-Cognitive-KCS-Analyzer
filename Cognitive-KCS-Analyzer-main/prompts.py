KCS_QUALITY_CRITERIA = """
Assess the KCS article based on the following criteria. For each criterion, provide a score from 1 (Poor) to 5 (Excellent) and a brief justification.
Criteria:
1.  **Clarity:** Is the language clear, concise, and easy to understand? Is it free of jargon where possible, or is jargon explained?
2.  **Completeness:** Does the article seem to cover the issue comprehensively? Does it address potential follow-up questions or common variations of the problem?
3.  **Actionability:** Does the article provide clear, step-by-step instructions that a user can follow to resolve the issue? Are the steps logical?
4.  **Structure & Readability:** Is the article well-structured with headings, bullet points, or numbered lists where appropriate? Is it easy to scan and find information?
5.  **Problem/Solution Statement:** Is the problem clearly stated? Is the solution directly addressing the stated problem?

Output Format:
Clarity: [Score]/5 - [Justification]
Completeness: [Score]/5 - [Justification]
Actionability: [Score]/5 - [Justification]
Structure & Readability: [Score]/5 - [Justification]
Problem/Solution Statement: [Score]/5 - [Justification]
Overall Score: [Average Score]/5
Summary of Strengths: [Brief summary]
Summary of Weaknesses/Areas for Improvement: [Brief summary]
"""

def get_article_quality_assessment_prompt(article_title, article_content):
    return f"""You are an KCS Quality Assessor.
Your task is to evaluate the following KCS knowledge base article.

Article Title: "{article_title}"
Article Content:
---
{article_content}
---

Evaluation Task:
{KCS_QUALITY_CRITERIA}
"""

def get_knowledge_gap_identification_prompt(ticket_summary, existing_article_titles):
    article_titles_str = "\n".join([f"- {title}" for title in existing_article_titles])
    return f"""You are a Knowledge Management Analyst.
A new support ticket has come in with the following summary:
Ticket Summary: "{ticket_summary}"

Here is a list of existing KCS article titles in our knowledge base:
---
{article_titles_str}
---

Task:
1.  Determine if the ticket query is likely already covered by one of the existing articles. If yes, state which one(s) seem most relevant.
2.  If it's NOT clearly covered, or only partially covered, state that this represents a potential knowledge gap.
3.  Suggest a concise title for a new KCS article that would address this ticket query if it's a gap. If it's partially covered, suggest how an existing article could be improved or if a new, more specific article is needed.

Output your analysis clearly. For example:
Relevance to existing articles: [Covered by KCSXXX / Partially covered by KCSXXX / Potential knowledge gap]
Suggestion: [Title for new article / How to improve existing article KCSXXX / No action needed if fully covered]
"""

def get_article_improvement_prompt(article_title, article_content, related_ticket_summary):
    return f"""You are a KCS Article Improvement Specialist.
You are reviewing an existing KCS article in light of a new, related support ticket.

Existing Article Title: "{article_title}"
Existing Article Content:
---
{article_content}
---

Related Support Ticket Summary: "{related_ticket_summary}"

Task:
1. Analyze if the information in the ticket summary reveals a shortcoming or a missing piece of information in the existing article.
2. If yes, suggest specific improvements to the article content to better address the scenario described in the ticket. Be precise.
3. If the article already covers it well, state that no immediate improvement is needed based on this ticket.

Output Format:
Analysis: [Detailed analysis of how the ticket relates to the article's current content]
Suggested Improvements: [Specific changes, additions, or clarifications for the article. If none, state "No improvements needed based on this ticket."]
"""
