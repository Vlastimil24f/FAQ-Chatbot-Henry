import io
import PyPDF2
import docx
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

app = FastAPI()

# ---------------- GLOBAL STATE (in-memory DB) ----------------
FAQ_QUESTIONS = []
FAQ_ANSWERS = []
VECTORIZER = None
FAQ_VECTORS = None
MODEL = None


# ---------------- UTILITIES ----------------

def extract_text(uploaded_file: UploadFile):
    ext = uploaded_file.filename.split(".")[-1].lower()

    if ext == "txt":
        return uploaded_file.file.read().decode("utf-8", errors="ignore")

    elif ext == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file.file)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text

    elif ext == "docx":
        doc_file = docx.Document(uploaded_file.file)
        return "\n".join([p.text for p in doc_file.paragraphs])

    return ""


def parse_faq_text(text: str):
    questions, answers = [], []
    blocks = text.split("\n\n")

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) >= 2:
            q = lines[0].replace("Q:", "").strip()
            a = lines[1].replace("A:", "").strip()
            questions.append(q)
            answers.append(a)

    return questions, answers


def build_faq_index():
    global VECTORIZER, FAQ_VECTORS
    VECTORIZER = TfidfVectorizer()
    FAQ_VECTORS = VECTORIZER.fit_transform(FAQ_QUESTIONS)


def load_model():
    global MODEL
    if MODEL is None:
        MODEL = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=120
        )


def rewrite_answer(question, answer):
    load_model()

    prompt = (
        f"Rewrite this answer clearly and politely. Do not add new information.\n\n"
        f"User question: {question}\n"
        f"Answer: {answer}"
    )
    output = MODEL(prompt)[0]["generated_text"]
    return output.strip()


# ---------------- REQUEST MODELS ----------------

class AskRequest(BaseModel):
    message: str


# ---------------- API ENDPOINTS ----------------

@app.post("/load_faqs")
async def load_faqs(files: List[UploadFile] = File(...)):
    global FAQ_QUESTIONS, FAQ_ANSWERS

    combined_q = []
    combined_a = []

    for file in files:
        text = extract_text(file)
        qs, ans = parse_faq_text(text)

        if qs:
            combined_q.extend(qs)
            combined_a.extend(ans)

    if not combined_q:
        return {"error": "No valid FAQs detected."}

    FAQ_QUESTIONS = combined_q
    FAQ_ANSWERS = combined_a
    build_faq_index()

    return {
        "status": "OK",
        "faq_entries": len(FAQ_QUESTIONS)
    }


@app.get("/status")
async def status():
    if len(FAQ_QUESTIONS) == 0:
        return {"loaded": False, "faq_entries": 0}
    return {"loaded": True, "faq_entries": len(FAQ_QUESTIONS)}


@app.post("/ask")
async def ask_question(request: AskRequest):
    if len(FAQ_QUESTIONS) == 0:
        return {"error": "No FAQ data loaded. Upload files using /load_faqs first."}

    user_q = request.message

    query_vec = VECTORIZER.transform([user_q])
    sims = cosine_similarity(query_vec, FAQ_VECTORS).flatten()

    idx = sims.argmax()
    score = sims[idx]

    if score < 0.2:
        return {
            "answer": "I'm not sure about that. Please contact support.",
            "matched_question": None,
            "similarity": float(score)
        }

    raw_answer = FAQ_ANSWERS[idx]
    polished = rewrite_answer(user_q, raw_answer)

    return {
        "answer": polished,
        "matched_question": FAQ_QUESTIONS[idx],
        "similarity": float(score),
        "raw_answer": raw_answer
    }
