import streamlit as st
from transformers import pipeline
import PyPDF2
import docx
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import faiss

# ---------- Extract Text From Uploaded File ----------
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.name.split('.')[-1].lower()

    # TXT files
    if file_type == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    # PDF files
    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

    # DOCX files
    elif file_type == "docx":
        doc_file = docx.Document(uploaded_file)
        return "\n".join([p.text for p in doc_file.paragraphs])

    return ""

# ---------- Parse FAQ Text ----------
def parse_faq_text(text):
    questions = []
    answers = []

    blocks = text.split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 2:
            q = lines[0].replace("Q:", "").strip()
            a = lines[1].replace("A:", "").strip()
            questions.append(q)
            answers.append(a)

    return questions, answers

# ---------- Load Local FAQ File ----------
def load_faq(file_path="faq.txt"):
    questions = []
    answers = []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Split on blank lines
    blocks = content.split("\n\n")
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
            q = lines[0][2:].strip()
            a = lines[1][2:].strip()
            questions.append(q)
            answers.append(a)

    return questions, answers

# ---------- Embedding model (cached) ----------
@st.cache_resource
def get_embedder():
    # Small, fast, good for semantic search
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ---------- FAQ Chatbot with Hybrid Search (Embeddings + TF-IDF + FAISS) ----------
class FAQChatbot:
    def __init__(self, questions, answers, alpha=0.65):
        """
        alpha = weight for embeddings vs TF-IDF in hybrid score.
        hybrid_score = alpha * embedding_sim + (1 - alpha) * tfidf_sim
        """
        self.questions = questions
        self.answers = answers
        self.alpha = alpha

        # --- Embeddings + FAISS index ---
        self.embedder = get_embedder()
        question_embeddings = self.embedder.encode(
            self.questions,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        dim = question_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # inner product = cosine if normalized
        self.faiss_index.add(question_embeddings)

        # Keep embeddings too (optional, but useful)
        self.question_embeddings = question_embeddings

        # --- TF-IDF matrix ---
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def get_best_answer(self, user_query, threshold=0.3):
        if not user_query.strip():
            return "Please ask a question."

        # 1) Embedding similarity via FAISS
        query_emb = self.embedder.encode(
            [user_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        k = len(self.questions)
        D, I = self.faiss_index.search(query_emb, k)  # D: scores, I: indices

        emb_sims = np.zeros(len(self.questions), dtype="float32")
        for rank, idx in enumerate(I[0]):
            emb_sims[idx] = D[0][rank]

        # 2) TF-IDF similarity
        query_vec = self.vectorizer.transform([user_query])
        tfidf_sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 3) Hybrid score
        hybrid_scores = self.alpha * emb_sims + (1.0 - self.alpha) * tfidf_sims

        best_idx = int(np.argmax(hybrid_scores))
        best_score = float(hybrid_scores[best_idx])

        if best_score < threshold:
            return "I'm not sure about that. Please contact support."

        return self.answers[best_idx], self.questions[best_idx], best_score

# ---------- LLM Rewriter ----------
@st.cache_resource
def get_rewriter():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=120
    )

def rewrite_answer(user_question, faq_answer, rewriter):
    prompt = (
        f"Rewrite the following answer so it clearly and politely responds to the user's question. "
        f"Do not add any new information. Keep it short and helpful.\n\n"
        f"User question: {user_question}\n"
        f"Answer: {faq_answer}"
    )

    output = rewriter(prompt)[0]["generated_text"]
    return output.strip()

# ---------- NEW: Memory-Based Question Rewriter ----------
def rewrite_with_memory(user_question, last_question, rewriter):
    """
    Use the previous question to fill in missing context in the new one.
    Helps with vague questions like 'How long does it take?'.
    """
    if not last_question:
        return user_question

    prompt = (
        "Rewrite the new question so it includes missing context from the previous question. "
        "Do NOT invent anything new. Simply fill in pronouns and vague wording.\n\n"
        f"Previous question: {last_question}\n"
        f"New question: {user_question}\n\n"
        "Rewritten question:"
    )

    result = rewriter(prompt)[0]["generated_text"].strip()
    return result

# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="FAQ Chatbot - Henry", page_icon="💬", layout="centered")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []  # List of {"role":..., "content":...}

    # MULTI-FILE UPLOAD
    uploaded_files = st.file_uploader(
        "Upload one or more FAQ files (txt, pdf, docx)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )
    st.caption("If no files are uploaded, the default faq.txt will be used.")

    all_questions = []
    all_answers = []

    # LOAD MULTIPLE FAQ FILES
    if uploaded_files:
        for file in uploaded_files:
            raw_text = extract_text_from_file(file)
            qs, ans = parse_faq_text(raw_text)

            if qs:
                all_questions.extend(qs)
                all_answers.extend(ans)

        if not all_questions:
            st.error("Uploaded files did not contain valid FAQ entries.")
            return

        st.success(f"Loaded {len(all_questions)} FAQ entries from {len(uploaded_files)} file(s).")

    else:
        try:
            all_questions, all_answers = load_faq("faq.txt")
        except FileNotFoundError:
            st.error("faq.txt missing!")
            return

        if not all_questions:
            st.error("faq.txt is empty or incorrectly formatted.")
            return

        st.info(f"Loaded {len(all_questions)} FAQ entries from default faq.txt.")

    # Initialize chatbot with COMBINED FAQ data
    chatbot = FAQChatbot(all_questions, all_answers)
    rewriter = get_rewriter()

    st.write("---")

    # DISPLAY CHAT HISTORY
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"You: {msg['content']}")
        else:
            st.markdown(f"Henry: {msg['content']}")
            if "raw" in msg:
                with st.expander("📄 Raw FAQ Answer"):
                    st.write(msg["raw"])
                with st.expander("🔍 Matched FAQ & Confidence"):
                    st.write(f"**Matched question:** {msg['matched']}")
                    st.write(f"**Similarity score:** {msg['score']:.2f}")

    # INPUT AREA (form prevents infinite loop)
    with st.form("chat_input_form", clear_on_submit=True):
        user_query = st.text_input(
            "Ask a question:",
            placeholder="e.g., 'What's your return policy?'",
        )
        submitted = st.form_submit_button("Send")

    if submitted and user_query:

        # ---- MEMORY: get last user question ----
        last_user_question = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_question = msg["content"]
                break

        # ---- Use memory to rewrite the new question ----
        rewritten_query = rewrite_with_memory(user_query, last_user_question, rewriter)

        # 1. Save original user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_query,
        })

        # 2. Process answer using the rewritten contextual question
        result = chatbot.get_best_answer(rewritten_query)

        if isinstance(result, str):
            final_answer = result
            raw = ""
            matched_q = ""
            score = 0
        else:
            answer, matched_q, score = result
            raw = answer
            final_answer = rewrite_answer(rewritten_query, answer, rewriter)

        # 3. Add bot message
        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer,
            "raw": raw,
            "matched": matched_q,
            "score": score
        })

        # 4. Refresh UI
        st.rerun()

if __name__ == "__main__":
    main()


















