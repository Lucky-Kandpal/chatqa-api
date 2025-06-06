# import json
# from langchain_community.chat_models import ChatOllama
# from datetime import datetime
# import re
# from collections import Counter

# # ========= CONFIG ===========
# CHAT_FILE = "whatsapp_flat.json"
# OLLAMA_MODEL = "llama3:instruct"
# # ============================

# # Step 1: Load Chat Data
# with open(CHAT_FILE, "r", encoding="utf-8") as f:
#     chat_data = json.load(f)

# # Step 2: Initialize LLM
# llm = ChatOllama(model=OLLAMA_MODEL)

# # Step 3: Get All Unique Senders
# def get_all_senders(data):
#     return sorted(set(msg["sender"] for msg in data))

# # Step 4: Query Classification
# def classify_query_prompt(query):
#     return f"""
# Classify the following query into one of these categories:

# 1. FILTER - Queries asking for specific messages (first/last message by someone, messages on a date, etc.)
# 2. SUMMARIZE - Queries asking for summary, overview, or general information about conversations
# 3. CONTEXT_QA - Queries asking questions that need context from multiple messages to answer
# 4. TOPIC_QA - Queries asking about specific topics, events, or discussions mentioned in chat
# 5. STATS - Queries asking for statistics, counts, or analytical information

# Query: "{query}"

# Respond with only the category name (FILTER, SUMMARIZE, CONTEXT_QA, TOPIC_QA, or STATS).
# """

# # Step 5: Prompt to Resolve Fuzzy Sender Name
# def resolve_sender_prompt(user_query, all_senders):
#     sender_list = "\n".join(f"- {name}" for name in all_senders)
#     return f"""
# You are given a user query and a list of real WhatsApp sender names.

# Sender list:
# {sender_list}

# Query:
# "{user_query}"

# If the query refers to a person, return the most likely matching sender name **exactly as it appears** in the list.
# If no sender is mentioned, return `null`.

# Respond only with the exact name string or `null`.
# """

# # Step 6: Prompt LLM to Convert Query to JSON Filters
# def extract_filter_prompt(query):
#     return f"""
# You are given a user query about a WhatsApp chat log. Extract a JSON object containing filters:
# - "sender": if the query mentions a person (optional)
# - "date": if the query specifies a date (format: dd/mm/yy) (optional)
# - "order": "asc" or "desc" if it says "first", "last", "latest", etc. (optional)
# - "limit": how many messages to return (default 1 if asking for first/last)

# QUERY:
# "{query}"

# Respond with only a JSON object. No explanations.
# """

# # Step 7: Apply Filters to Chat Data
# def apply_filters(chat_data, filters):
#     results = chat_data

#     sender = filters.get("sender")
#     if isinstance(sender, str) and sender.strip():
#         results = [m for m in results if m["sender"].lower() == sender.lower()]

#     date = filters.get("date")
#     if isinstance(date, str) and date.strip():
#         results = [m for m in results if m["date"] == date]

#     if "order" in filters:
#         reverse = filters["order"] == "desc"
#         results.sort(key=lambda m: parse_datetime(m), reverse=reverse)

#     if "limit" in filters:
#         results = results[:filters["limit"]]

#     return results

# def parse_datetime(msg):
#     dt_str = f"{msg['date']} {msg['time']}"
#     try:
#         return datetime.strptime(dt_str, "%d/%m/%y %I:%M:%S %p")
#     except:
#         return datetime.strptime(dt_str, "%d/%m/%y %I:%M:%S %p")

# # Step 8: Build Prompt for Final Answer (Filter queries)
# def build_prompt(messages, query):
#     if not messages:
#         return f"No messages found for your query: '{query}'"
#     context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in messages])
#     return f"""
# You are a WhatsApp chat assistant. Given the following messages:

# {context}

# Answer the question: "{query}"

# Be concise and refer to the message sender and time if needed.
# """

# # Step 9: RAG Functions for Different Query Types

# def retrieve_relevant_messages(query, chat_data, top_k=20):
#     """Simple keyword-based retrieval for RAG"""
#     query_words = set(query.lower().split())
#     scored_messages = []
    
#     for msg in chat_data:
#         msg_words = set(msg["message"].lower().split())
#         # Simple scoring based on word overlap
#         score = len(query_words.intersection(msg_words))
#         if score > 0:
#             scored_messages.append((score, msg))
    
#     # Sort by score and return top_k
#     scored_messages.sort(key=lambda x: x[0], reverse=True)
#     return [msg for _, msg in scored_messages[:top_k]]

# def handle_summarize_query(query, chat_data):
#     """Handle summarization queries using RAG"""
#     # Get a sample of messages across time periods
#     sorted_msgs = sorted(chat_data, key=lambda m: parse_datetime(m))
    
#     # Sample messages from different time periods
#     total_msgs = len(sorted_msgs)
#     sample_indices = [0, total_msgs//4, total_msgs//2, 3*total_msgs//4, total_msgs-1]
#     sample_msgs = [sorted_msgs[i] for i in sample_indices if i < total_msgs]
    
#     # Add some recent messages
#     recent_msgs = sorted_msgs[-10:]
#     all_msgs = sample_msgs + recent_msgs
    
#     # Remove duplicates while preserving order
#     seen = set()
#     unique_msgs = []
#     for msg in all_msgs:
#         msg_id = f"{msg['date']}_{msg['time']}_{msg['sender']}"
#         if msg_id not in seen:
#             seen.add(msg_id)
#             unique_msgs.append(msg)
    
#     context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in unique_msgs[-30:]])
    
#     prompt = f"""
# You are analyzing a WhatsApp chat log. Based on the following sample messages:

# {context}

# Total messages in chat: {len(chat_data)}
# Participants: {', '.join(get_all_senders(chat_data))}

# Answer this query: "{query}"

# Provide a comprehensive summary covering the main topics, participants' communication patterns, and key insights.
# """
#     return prompt

# def handle_context_qa(query, chat_data):
#     """Handle context-based Q&A using RAG"""
#     relevant_msgs = retrieve_relevant_messages(query, chat_data, top_k=30)
    
#     if not relevant_msgs:
#         # Fallback to recent messages
#         sorted_msgs = sorted(chat_data, key=lambda m: parse_datetime(m))
#         relevant_msgs = sorted_msgs[-20:]
    
#     context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in relevant_msgs])
    
#     prompt = f"""
# You are a WhatsApp chat assistant. Based on the following relevant messages from the conversation:

# {context}

# Answer this question: "{query}"

# Use the context from the messages to provide a comprehensive answer. Reference specific messages or participants when relevant.
# """
#     return prompt

# def handle_topic_qa(query, chat_data):
#     """Handle topic-specific Q&A using RAG"""
#     relevant_msgs = retrieve_relevant_messages(query, chat_data, top_k=25)
    
#     if not relevant_msgs:
#         return f"No relevant messages found for the topic in your query: '{query}'"
    
#     context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in relevant_msgs])
    
#     prompt = f"""
# You are analyzing a WhatsApp chat for specific topics. Here are the most relevant messages:

# {context}

# Question about the topic: "{query}"

# Analyze the messages and provide detailed insights about this topic, including who discussed it, when, and what was said.
# """
#     return prompt

# def handle_stats_query(query, chat_data):
#     """Handle statistical queries"""
#     # Generate statistics
#     total_messages = len(chat_data)
#     senders = get_all_senders(chat_data)
#     sender_counts = Counter(msg["sender"] for msg in chat_data)
    
#     # Date range
#     sorted_msgs = sorted(chat_data, key=lambda m: parse_datetime(m))
#     first_msg_date = sorted_msgs[0]["date"] if sorted_msgs else "N/A"
#     last_msg_date = sorted_msgs[-1]["date"] if sorted_msgs else "N/A"
    
#     # Daily message counts
#     daily_counts = Counter(msg["date"] for msg in chat_data)
#     most_active_day = daily_counts.most_common(1)[0] if daily_counts else ("N/A", 0)
    
#     # Get top sender
#     top_sender = sender_counts.most_common(1)[0] if sender_counts else ("N/A", 0)
    
#     # Create a comprehensive stats summary
#     stats_summary = f"""
# Chat Overview:
# - Total messages: {total_messages}
# - Total participants: {len(senders)}
# - Date range: {first_msg_date} to {last_msg_date}
# - Most active day: {most_active_day[0]} with {most_active_day[1]} messages

# Sender who sent MOST messages: {top_sender[0]} with {top_sender[1]} messages

# All participants ranked by message count:
# {chr(10).join([f"{i+1}. {sender}: {count} messages" for i, (sender, count) in enumerate(sender_counts.most_common())])}

# Most active days:
# {chr(10).join([f"{i+1}. {date}: {count} messages" for i, (date, count) in enumerate(daily_counts.most_common(5))])}

# Participants list: {', '.join(senders)}
# """
    
#     prompt = f"""
# You are a WhatsApp chat statistics assistant. Here are the complete and accurate statistics:

# {stats_summary}

# User question: "{query}"

# Answer the user's question using the exact information provided above. Be direct and specific.
# """
#     return prompt

# # Step 10: Enhanced CLI Chat
# print("💬 Enhanced WhatsApp Query Bot with RAG\nType 'exit' to quit.")
# print("Supports: Filtering, Summarization, Context Q&A, Topic Analysis, and Statistics")

# while True:
#     try:
#         query = input("\n🧠 Ask something: ").strip()
#         if query.lower() in ["exit", "quit"]:
#             print("👋 Goodbye!")
#             break

#         # Classify the query type
#         query_type = llm.invoke(classify_query_prompt(query)).content.strip()
#         print(f"🔍 Query type: {query_type}")

#         if query_type == "FILTER":
#             # Original filtering logic
#             all_senders = get_all_senders(chat_data)
#             resolved_sender = llm.invoke(resolve_sender_prompt(query, all_senders)).content.strip()

#             filter_response = llm.invoke(extract_filter_prompt(query))
#             filter_content = filter_response.content.strip()
#             try:
#                 filters = json.loads(filter_content)
#             except json.JSONDecodeError as e:
#                 print(f"❌ Error parsing filters: {str(e)}")
#                 continue

#             if resolved_sender.lower() != "null" and "sender" not in filters:
#                 filters["sender"] = resolved_sender

#             results = apply_filters(chat_data, filters)
#             final_prompt = build_prompt(results, query)
            
#         elif query_type == "SUMMARIZE":
#             final_prompt = handle_summarize_query(query, chat_data)
            
#         elif query_type == "CONTEXT_QA":
#             final_prompt = handle_context_qa(query, chat_data)
            
#         elif query_type == "TOPIC_QA":
#             final_prompt = handle_topic_qa(query, chat_data)
            
#         elif query_type == "STATS":
#             final_prompt = handle_stats_query(query, chat_data)
            
#         else:
#             # Fallback to context QA
#             final_prompt = handle_context_qa(query, chat_data)

#         # Get final answer
#         answer = llm.invoke(final_prompt)
#         print("\n🤖", answer.content.strip())

#     except Exception as e:
#         print(f"❌ Error: {str(e)}")


import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from datetime import datetime
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware

# ========= CONFIG ===========
CHAT_FILE = "whatsapp_flat.json"
OLLAMA_MODEL = "llama3:instruct"
# ============================

# Load chat data
with open(CHAT_FILE, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

# Initialize LLM
llm = ChatOllama(model=OLLAMA_MODEL)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryInput(BaseModel):
    query: str


# ===== Helper Functions =====

def get_all_senders(data):
    return sorted(set(msg["sender"] for msg in data))

def classify_query_prompt(query):
    return f"""
Classify the following query into one of these categories:

1. FILTER
2. SUMMARIZE
3. CONTEXT_QA
4. TOPIC_QA
5. STATS

Query: "{query}"
Respond with only the category name.
"""

def resolve_sender_prompt(user_query, all_senders):
    sender_list = "\n".join(f"- {name}" for name in all_senders)
    return f"""
You are given a user query and a list of real WhatsApp sender names.

Sender list:
{sender_list}

Query:
"{user_query}"

If the query refers to a person, return the matching sender name from the list.
If no sender is mentioned, return `null`.
"""

def extract_filter_prompt(query):
    return f"""
Extract a JSON object containing filters:
- "sender" (optional)
- "date" (format: dd/mm/yy) (optional)
- "order": "asc" or "desc" (optional)
- "limit": int (optional)

QUERY:
"{query}"

Only return a JSON object.
"""

def apply_filters(chat_data, filters):
    results = chat_data
    sender = filters.get("sender")
    if sender:
        results = [m for m in results if m["sender"].lower() == sender.lower()]
    date = filters.get("date")
    if date:
        results = [m for m in results if m["date"] == date]
    if "order" in filters:
        reverse = filters["order"] == "desc"
        results.sort(key=lambda m: parse_datetime(m), reverse=reverse)
    if "limit" in filters:
        results = results[:filters["limit"]]
    return results

def parse_datetime(msg):
    dt_str = f"{msg['date']} {msg['time']}"
    return datetime.strptime(dt_str, "%d/%m/%y %I:%M:%S %p")

def build_prompt(messages, query):
    if not messages:
        return f"No messages found for your query: '{query}'"
    context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in messages])
    return f"""
You are a WhatsApp chat assistant. Given the following messages:

{context}

Answer the question: "{query}"
"""

def retrieve_relevant_messages(query, chat_data, top_k=20):
    query_words = set(query.lower().split())
    scored_messages = []
    for msg in chat_data:
        msg_words = set(msg["message"].lower().split())
        score = len(query_words.intersection(msg_words))
        if score > 0:
            scored_messages.append((score, msg))
    scored_messages.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in scored_messages[:top_k]]

def handle_summarize_query(query, chat_data):
    sorted_msgs = sorted(chat_data, key=parse_datetime)
    total_msgs = len(sorted_msgs)
    sample_indices = [0, total_msgs//4, total_msgs//2, 3*total_msgs//4, total_msgs-1]
    sample_msgs = [sorted_msgs[i] for i in sample_indices if i < total_msgs]
    recent_msgs = sorted_msgs[-10:]
    all_msgs = sample_msgs + recent_msgs

    seen = set()
    unique_msgs = []
    for msg in all_msgs:
        msg_id = f"{msg['date']}_{msg['time']}_{msg['sender']}"
        if msg_id not in seen:
            seen.add(msg_id)
            unique_msgs.append(msg)

    context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in unique_msgs[-30:]])
    prompt = f"""
You are analyzing a WhatsApp chat log. Based on the following sample messages:

{context}

Total messages: {len(chat_data)}
Participants: {', '.join(get_all_senders(chat_data))}

Query: "{query}"
Provide a summary with main topics and participant patterns.
"""
    return prompt

def handle_context_qa(query, chat_data):
    relevant_msgs = retrieve_relevant_messages(query, chat_data, top_k=30)
    if not relevant_msgs:
        relevant_msgs = sorted(chat_data, key=parse_datetime)[-20:]
    context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in relevant_msgs])
    return f"""
You are a WhatsApp chat assistant. Based on relevant messages:

{context}

Answer: "{query}"
"""

def handle_topic_qa(query, chat_data):
    relevant_msgs = retrieve_relevant_messages(query, chat_data, top_k=25)
    if not relevant_msgs:
        return f"No messages found for topic in: '{query}'"
    context = "\n".join([f"[{m['date']} {m['time']}] {m['sender']}: {m['message']}" for m in relevant_msgs])
    return f"""
Topic Analysis from WhatsApp:

{context}

Topic: "{query}"
Discuss who talked, what they said, and when.
"""

def handle_stats_query(query, chat_data):
    total_messages = len(chat_data)
    senders = get_all_senders(chat_data)
    sender_counts = Counter(msg["sender"] for msg in chat_data)
    sorted_msgs = sorted(chat_data, key=parse_datetime)
    first_msg_date = sorted_msgs[0]["date"] if sorted_msgs else "N/A"
    last_msg_date = sorted_msgs[-1]["date"] if sorted_msgs else "N/A"
    daily_counts = Counter(msg["date"] for msg in chat_data)
    top_sender = sender_counts.most_common(1)[0] if sender_counts else ("N/A", 0)
    stats_summary = f"""
Total messages: {total_messages}
Date range: {first_msg_date} to {last_msg_date}
Top sender: {top_sender[0]} ({top_sender[1]} messages)
Top days:
{chr(10).join([f"{i+1}. {d}: {c}" for i, (d, c) in enumerate(daily_counts.most_common(5))])}
"""
    return f"""
Chat Statistics:

{stats_summary}

Query: "{query}"
Answer based only on above data.
"""

# ===== Endpoint =====

@app.post("/query")
async def ask_question(payload: QueryInput):
    query = payload.query.strip()
    query_type = llm.invoke(classify_query_prompt(query)).content.strip()

    if query_type == "FILTER":
        all_senders = get_all_senders(chat_data)
        resolved_sender = llm.invoke(resolve_sender_prompt(query, all_senders)).content.strip()
        filter_response = llm.invoke(extract_filter_prompt(query)).content.strip()
        try:
            filters = json.loads(filter_response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse filters from LLM"}

        if resolved_sender.lower() != "null" and "sender" not in filters:
            filters["sender"] = resolved_sender

        results = apply_filters(chat_data, filters)
        final_prompt = build_prompt(results, query)

    elif query_type == "SUMMARIZE":
        final_prompt = handle_summarize_query(query, chat_data)
    elif query_type == "CONTEXT_QA":
        final_prompt = handle_context_qa(query, chat_data)
    elif query_type == "TOPIC_QA":
        final_prompt = handle_topic_qa(query, chat_data)
    elif query_type == "STATS":
        final_prompt = handle_stats_query(query, chat_data)
    else:
        final_prompt = handle_context_qa(query, chat_data)

    answer = llm.invoke(final_prompt)
    return {"query_type": query_type, "answer": answer.content.strip()}
