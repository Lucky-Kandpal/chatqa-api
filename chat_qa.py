
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from datetime import datetime
from collections import Counter
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
You are a WhatsApp chat query classifier. Classify the following query into one of these categories:

1. FILTER - Used when the query asks to find specific messages
   Examples:
   - "Show messages from John"
   - "Find conversations between 1st and 15th May"
   - "Get the last 10 messages"
   - "Sort messages by date"
   - "Search for discussions about X"

2. SUMMARIZE - Used when the query asks for an overview or summary
   Examples:
   - "Summarize our conversation about project X"
   - "What was discussed in May?"
   - "Give me a summary of the chat"
   - "What topics were discussed?"
   - "Summarize the main points"

3. CONTEXT_QA - Used when the query asks about specific context in the chat
   Examples:
   - "What did John say about the meeting?"
   - "When was the last time we discussed this?"
   - "What was the decision about X?"
   - "Tell me what was said about Y"

4. TOPIC_QA - Used when the query asks about specific topics or conversations
   Examples:
   - "What was the discussion about the project?"
   - "Who talked about the deadline?"
   - "What was the plan for X?"
   - "Tell me about the conversation on Y"

5. STATS - Used when the query asks for statistics or counts
   Examples:
   - "How many messages were sent?"
   - "Who sent the most messages?"
   - "What was the busiest day?"
   - "How many times did we discuss X?"
   - "Show me message statistics"

6. GREETING - Used for any friendly or casual opening
   Examples:
   - Any variation of hello or greeting
   - "Hi"
   - "Hello"
   - "Hey"
   - "Hi there"
   - "Hi, I'm new here"
   - "Hi, how are you?"
   - "Hi, I have a question"

7. CONVERSATIONAL - Used for casual questions about capabilities
   Examples:
   - "What can you do?"
   - "How do you work?"
   - "What kind of questions can you answer?"
   - "Help me"
   - "Can you help me with X?"

Query: "{query}"

Important Rules:
1. If the query starts with any greeting or friendly opening, classify it as GREETING
2. If the query contains a question about capabilities or asking for help, classify it as CONVERSATIONAL
3. If the query is asking to find specific messages, classify it as FILTER
4. If the query is asking for statistics or counts, classify it as STATS
5. If the query is asking about topics or discussions, classify it as TOPIC_QA
6. If the query is asking for a summary or overview, classify it as SUMMARIZE
7. If the query is asking about specific context in the chat, classify it as CONTEXT_QA

Respond with only the category name from the list above. If you're unsure, respond with "UNKNOWN".
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
def count_tokens(text: str) -> int:
    """Count tokens in text using a simple whitespace-based approach."""
    return len(text.split())
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
{stats_summary}
"""

# ===== Helper Functions for Greetings and Conversations =====

def handle_greeting_query(query):
    greetings = [
        "Hi! I'm here to help with your WhatsApp chat data. How can I assist you today?",
        "Hello! Ready to help with your chat analysis. What would you like to know?",
        "Hi there! I can help you analyze your WhatsApp chat data. What can I do for you?"
    ]
    return greetings[0]  # Can be randomized later if needed

def handle_conversational_query(query):
    responses = {
        "how are you": "I'm doing great, thanks for asking! Ready to help with your WhatsApp chat data.",
        "how are you doing": "I'm doing great, thanks for asking! Ready to help with your WhatsApp chat data.",
        "what can you do": "I can help analyze your WhatsApp chat data. I can filter messages, provide summaries, answer context-based questions, analyze topics, and provide statistics.",
        "help": "I can help with: filtering messages, summarizing chats, answering context-based questions, analyzing topics, and providing statistics."
    }
    query = query.lower()
    for key in responses:
        if key in query:
            return responses[key]
    return "I'm here to help with your WhatsApp chat data. What would you like to know?"


SYSTEM_PROMPT = f"""
You are a helpful assistant answering questions about a WhatsApp chat dataset.

Data Summary:
- Total messages: {len(chat_data)}
- Participants: {', '.join(get_all_senders(chat_data))}
- Date Range: {parse_datetime(chat_data[0]).strftime("%d %b %Y")} to {parse_datetime(chat_data[-1]).strftime("%d %b %Y")}

Instructions:
- Answer queries naturally using the chat data.
- Understand and extract date ranges or sender names if mentioned.
- If user asks to summarize messages between two dates, provide a meaningful topic summary.
- If the query is ambiguous or too vague, ask a clarifying question.
- Do not assume facts not present in data.
"""

# ===== Endpoint =====

@app.post("/query")
async def ask_question(payload: QueryInput):
    query = payload.query
    logger.info(f"Received query: {query}")
    # First, classify the query
    try:
        # Check for greetings anywhere in the query
        query_lower = query.lower()
        if any(greeting in query_lower for greeting in ["hi", "hello", "hey", "greetings", "hey there"]):
            category = "GREETING"
        else:
            # Try to classify using LLM
            try:
                category = llm.invoke(classify_query_prompt(query)).content.strip()
                valid_categories = ["FILTER", "SUMMARIZE", "CONTEXT_QA", "TOPIC_QA", "STATS", "GREETING", "CONVERSATIONAL", "UNKNOWN"]
                if category not in valid_categories:
                    category = "UNKNOWN"
            except Exception as e:
                logger.error(f"Error getting LLM classification: {str(e)}")
                category = "UNKNOWN"
        
        logger.info(f"Query classified as: {category}")
    except Exception as e:
        logger.error(f"Error classifying query: {str(e)}")
        category = "UNKNOWN"
    
    # Handle different query types
    if category == "FILTER":
        try:
            filters = json.loads(llm.invoke(extract_filter_prompt(query)).content)
            filtered_messages = apply_filters(chat_data, filters)
            prompt = build_prompt(filtered_messages, query)
        except json.JSONDecodeError:
            logger.error("Invalid JSON received from LLM")
            return "I couldn't understand the filters in your query. Please try again with a clearer format."
        except Exception as e:
            logger.error(f"Error processing filter query: {str(e)}")
            return "I encountered an error processing your filter query. Please try again."
    elif category == "SUMMARIZE":
        prompt = handle_summarize_query(query, chat_data)
    elif category == "CONTEXT_QA":
        prompt = handle_context_qa(query, chat_data)
    elif category == "TOPIC_QA":
        prompt = handle_topic_qa(query, chat_data)
    elif category == "STATS":
        prompt = handle_stats_query(query, chat_data)
    elif category == "GREETING":
        return handle_greeting_query(query)
    elif category == "CONVERSATIONAL":
        return handle_conversational_query(query)
    else:
        return "I'm not sure how to handle that query. Please try asking about your WhatsApp chat data."
    
    # Generate response for non-greeting/conversational queries
    start_time = time.time()
    response = llm.invoke(prompt, system_prompt=SYSTEM_PROMPT)
    end_time = time.time()
    
    # Log token emission rate
    tokens = count_tokens(response.content)
    duration = end_time - start_time
    logger.info(f"LLM Response - Tokens: {tokens}, Duration: {duration:.2f}s")
    if duration > 0:
        tokens_per_second = tokens / duration
        logger.info(f"LLM Response - Tokens: {tokens}, Duration: {duration:.2f}s, Tokens/Sec: {tokens_per_second:.2f}")
    
    return response.content.strip()
