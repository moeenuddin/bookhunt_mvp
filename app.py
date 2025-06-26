import streamlit as st

# --- Dummy Data Sets ---
books = [
    "The Great Gatsby", "1984", "To Kill a Mockingbird",
    "Pride and Prejudice", "Moby Dick", "War and Peace",
    "The Odyssey", "A Tale of Two Cities", "The Catcher in the Rye"
]

templates = [
    {
        "id": "parent_filter",
        "name": "Parental Sensitivity Review",
        "role": "As a parent, you have to analyze the following book(s)...",
        "context": "Ensure it's age-appropriate for under 10.",
        "output": "Highlight sensitive terms (rating > 1)..."
    },
    {
        "id": "teacher_summary",
        "name": "Teacher Content Summary",
        "role": "As a teacher, provide a high-level summary...",
        "context": "Used by teachers aged 10–12.",
        "output": "Summarize themes and vocabulary..."
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

        # Append book and template info
        if st.session_state.selected_books:
            full_query += f"\n\n📚 Books: {', '.join(st.session_state.selected_books)}"
        if st.session_state.selected_template:
            full_query = (
                f"🧠 Role: {st.session_state.selected_template['role']}\n"
                f"🎯 Output: {st.session_state.selected_template['output']}\n\n"
                + full_query
            )

        # Simulated response
        st.session_state.chat_history.append(("user", user_input.strip()))
        st.session_state.chat_history.append(("bot", f"📝 **Prepared Query (Not sent)**\n\n{full_query}"))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "user":
            st.markdown(f"<div style='background:#d1e7dd;padding:10px;border-radius:10px;margin-bottom:5px'><b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#f8d7da;padding:10px;border-radius:10px;margin-bottom:5px'><b>Assistant:</b><br>{message}</div>", unsafe_allow_html=True)
