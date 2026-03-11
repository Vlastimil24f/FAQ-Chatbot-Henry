# FAQ Chatbot - Henry

Henry is an AI-powered FAQ chatbot that answers user questions by searching through FAQ documents using hybrid semantic search.

The system supports TXT, PDF, and DOCX files and retrieves the most relevant FAQ entry using embeddings, TF-IDF similarity, and FAISS vector indexing. The answer is then rewritten using a language model for clearer responses.

Live demo:
https://henry-faq.streamlit.app

---

## Features

- Chat-based FAQ assistant
- Upload FAQ files (TXT, PDF, DOCX)
- Hybrid semantic search
- Embeddings using SentenceTransformers
- Vector search with FAISS
- TF-IDF similarity scoring
- AI answer rewriting using FLAN-T5
- Conversation memory for follow-up questions
- Multi-file FAQ support

---

## Tech Stack

Python  
Streamlit  
SentenceTransformers  
FAISS  
Scikit-learn  
HuggingFace Transformers  
PyPDF2  
python-docx

---

## Run Locally

Navigate to project folder

cd "FAQ Chatbot - Henry"

Activate virtual environment

source venv/bin/activate

Run the app

streamlit run app.py

---

## Example FAQ Format

Q: What is your return policy?
A: Items can be returned within 30 days.

Q: How long does shipping take?
A: Shipping typically takes 3-5 business days.
