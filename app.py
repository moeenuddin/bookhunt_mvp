import streamlit as st

import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import openai

#os.getenv("OPENAI_API_KEY")
api_key = st.secrets["openai"]["api_key"]

INDEX_DIR = "faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding and vectorstore only once
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
vectordb = FAISS.load_local(INDEX_DIR, embedding, allow_dangerous_deserialization=True)

import openai
from openai import OpenAI
import os

client = OpenAI(api_key=api_key)

def generate_openai_response(prompt, model="gpt-3.5-turbo"):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing children's books."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ OpenAI API call failed: {e}"


def search_by_bookname(query: str, bookname: str, top_k: int = 5):
    """
    Search the FAISS index for a query and return results filtered by bookname.
    
    Args:
        query (str): The search query.
        bookname (str): The book name to filter by (from metadata).
        top_k (int): Number of final results to return after filtering.

    Returns:
        List[Document]: A list of filtered LangChain Document objects.
    """
    try:
        # Search top 20 documents
        results = vectordb.similarity_search(query, k=20)

        # Filter results based on bookname metadata
        filtered = [
            doc for doc in results
            if doc.metadata.get("bookname") == bookname
        ][:top_k]

        return filtered

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


BOOKS_DIR = "books"

@st.cache_data
def get_books_from_directory():
    if not os.path.exists(BOOKS_DIR):
        return []
    return [
        os.path.splitext(f)[0]
        for f in os.listdir(BOOKS_DIR)
        if f.endswith(".txt")
    ]

books = get_books_from_directory()



templates = [
    {
        "id": "parent_filter",
        "name": "Parental Sensitivity Review",
        "role": '''  You are a parental content advisor AI reviewing parts of a book to detect content that may not be appropriate for a 9-year-old child.

Below is a passage from a book, along with some metadata.

''',
        "context": "Ensure it's age-appropriate for under 10.",
        "output_old": "Highlight sensitive terms (rating > 1)...",
        "output":'''
        Return your findings in **structured numbered bullet points**, not in JSON. The output should be easy to read for parents and educators.

---

### FORMAT:

**Sensitive Topics Detected:**

1. **Topic:** [Type of sensitivity, e.g., "Emotional Distress"]  
   **Quote:** "[Short excerpt triggering concern]"  
   **Location:** Chapter [X], Page [Y], Chunk [Z]  
   **Reason:** [Brief explanation of why this is considered sensitive for a 9-year-old]

2. **Topic:** ...

---

**Summary of Findings:**  
[2–3 sentence high-level summary of the content review — e.g., "This section contains mild emotional distress that may affect sensitive readers. No explicit or violent content was found."]

---
'''
    },
    {
        "id": "teacher_summary",
        "name": "Teacher Content Summary",
        "role": "As a teacher, provide a high-level summary...",
        "context": "Used by teachers aged 10–12.",
        "output": "Summarize themes and vocabulary..."
    },
    {
        "id": "AI_Reviewer",
        "name": "AI Reviewer",
        "role": '''
            You are an expert AI trained in parental content moderation. Analyze the following text carefully, including context, implications, and tone.

    Please determine whether the content contains any direct or indirect references to the following categories:

    1. Violence & Gore
    2. Strong or Inappropriate Language
    3. Sexual Content or Nudity

    
        ''',
        "context": "Used by teachers aged 10–12.",
        "output": ''' 
        For each category, if relevant, provide:

        [1] Relevance Judgment: Whether this category is present (Yes/No)

        [2] Severity (1–10): Where 1 = mild and 10 = extremely explicit or disturbing

        [3] Exact Quote(s): Short excerpt(s) from the text that triggered your judgment

        [4] Explanation: Why the content is sensitive, with consideration for a 9-year-old reader

        [5] Subtle Signals (Optional): Any indirect references (e.g. metaphors, tone shifts, implications) that may also suggest sensitivity

    🔄 Repeat this structure for each category, even if not all are triggered.

    ⛔ Important: Be thorough, neutral, and base your reasoning on both surface meaning and deeper emotional or psychological implications in the text.

    🚸 This review is intended to assess content appropriateness for children under 10 years old.

        
        '''
    },
    {
        "id": "bookreviewer",
        "name": "Book Critic",
        "role": "You are a book reviewer, very critically reading the book content to estimate with evidence (WHY) the book has content in categories 'violence and gore','language' and 'sex and nudity' ",
        "context": "Used by teachers aged 10–12.",
        "output": '''return your answer in structured format:
        ------
        Violence and Gore:<rate 1 to 10>,
        Why_this rate for Violence and Gore:<summary of evidences>,
        Evidence:"[Short excerpt triggering concern]",
        Language:<rate 1 to 10>,
        Why_this rate for Language:<summary of evidences>,
        Evidence:"[Short excerpt triggering concern]",
        Sex and nudity:<rate 1 to 10>,
        Why_this rate for Sex and nudity:<summary of evidences>,
        Evidence:"[Short excerpt triggering concern]"

where 1 is least and 10 is highest
        ------
        '''
    }
]

standard_prompts = {
    "Violence": [
        "Does this book contain scenes of physical violence?",
        "Is there any mention of weapons, war, or fighting?",
        "Are there instances of bullying or abuse?"
    ],
    "Romance": [
        "Is there romantic content in this book?",
        "Are there any kissing, dating, or sexual references?"
    ],
    "Anxiety": [
        "Are there sad or emotionally intense scenes?",
        "Could this book be disturbing for sensitive children?"
    ],
    "Critics":[
        "Are there topics related directly or indirectly about 'violence & gore' or 'language' or 'sex and nudity'?",
        ""
    ]
}

# --- Initialize Session State ---
st.set_page_config(page_title="📚 ChatGPT UI MVP", layout="wide")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_books' not in st.session_state:
    st.session_state.selected_books = []

if 'selected_template' not in st.session_state:
    st.session_state.selected_template = None

if 'selected_prompt' not in st.session_state:
    st.session_state.selected_prompt = ""

# --- Sidebar Overview ---
st.sidebar.title("📋 Controls")
st.sidebar.success("Use the tabs to interact with the app.")
st.sidebar.info("Chat is simulated. No HuggingFace/OpenAI call made yet.")

# --- Tabs as Modal Replacements ---
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📚 Books", "🧠 Templates", "💡 Prompts"])

# --- 📚 Book Selection ---
with tab2:
    st.header("📚 Select Books")
    selected_books = st.multiselect("Choose from the list:", books, default=st.session_state.selected_books)
    st.session_state.selected_books = selected_books
    if selected_books:
        st.info(f"Selected: {', '.join(selected_books)}")

# --- 🧠 Template Selection ---
with tab3:
    st.header("🧠 Select Template")
    selected_name = st.selectbox("Choose a template:", [""] + [t["name"] for t in templates])
    selected_template = next((t for t in templates if t["name"] == selected_name), None)
    st.session_state.selected_template = selected_template

    if selected_template:
        with st.expander("🔍 Template Details", expanded=True):
            st.markdown(f"**Role:** {selected_template['role']}")
            st.markdown(f"**Context:** {selected_template['context']}")
            st.markdown(f"**Output Format:** {selected_template['output']}")

# --- 💡 Prompt Selection ---
with tab4:
    st.header("💡 Choose a Standard Query")
    category = st.selectbox("Prompt Category", [""] + list(standard_prompts.keys()))
    if category:
        prompt = st.selectbox("Pick a prompt:", standard_prompts[category])
        if prompt:
            st.session_state.selected_prompt = prompt
            st.success("Prompt selected. You can edit in Chat tab.")

# --- 💬 Chat UI ---
with tab1:
    st.header("💬 Chat with Assistant")
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your Message:", value=st.session_state.selected_prompt)
        submit = st.form_submit_button("Send")

    if submit and user_input.strip():
        full_query = user_input.strip()
        query=user_input
        book =None 
        if len(st.session_state.selected_books):
            book=st.session_state.selected_books[0] # 1 book
        matches = search_by_bookname(query, book)
        context=[]

        if matches:
            st.subheader("🔎 Top Matching Chunks")
            for i, doc in enumerate(matches, 1):
                st.markdown(f"### 🔹 Result {i}")
                st.markdown(f"📘 **Book:** `{doc.metadata.get('bookname')}`")
                st.markdown(f"📄 **Page:** `{doc.metadata.get('page')}`, **Chunk:** `{doc.metadata.get('chunk_number')}`")
                st.markdown(f"📝 {doc.page_content[:500]}{'...' if len(doc.page_content) > 500 else ''}")
                st.markdown("---")
                context.append(f" **Page:** `{doc.metadata.get('page')}`, **Chunk:** `{doc.metadata.get('chunk_number')}`"+doc.page_content+"\n")
        else:
            st.warning("No matching results found for the selected book.")
        
        #print md
        st.markdown(" ".join(context))


        # Append book and template info
        if st.session_state.selected_books:
            full_query += f"\n\n📚 Books: {', '.join(st.session_state.selected_books)}"
        if st.session_state.selected_template:
            full_query = (
                f"🧠 Role: {st.session_state.selected_template['role']}\n"
                + "   ".join(context)
                +f"🎯 Output in format: {st.session_state.selected_template['output']}\n\n"
                #+ full_query
                + user_input.strip()
            )

        # Simulated response
        st.session_state.chat_history.append(("user", user_input.strip()))
        #st.session_state.chat_history.append(("bot", f"📝 **Prepared Query (Not sent)**\n\n{full_query}"))
        final_prompt = full_query
        # 🔥 OpenAI Response
        response = generate_openai_response(final_prompt)
        st.session_state.chat_history.append(("bot", f"📝 **Response 3.5 turbo **\n\n{response}"))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"<div style='background:#d1e7dd;padding:10px;border-radius:10px;margin-bottom:5px'><b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#f8d7da;padding:10px;border-radius:10px;margin-bottom:5px'><b>Assistant:</b><br>{message}</div>", unsafe_allow_html=True)
