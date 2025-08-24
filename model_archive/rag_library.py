"""
Lightweight retrieval: TF-IDF + cosine similarity (scikit-learn).
Replace with embeddings + FAISS in production.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import os
import requests

SAMPLE_DOCS = [
    {"id": "doc1", "source": "OralHealth_Guide_2022", "text": "Leukoplakia is a white patch on the oral mucosa. Biopsy recommended if persists >2 weeks."},
    {"id": "doc2", "source": "WHO_Oral_Cancer_2019", "text": "Risk factors include tobacco, alcohol, betel nut. Early biopsy indicated for suspicious lesions."},
    {"id": "doc3", "source": "ClinicSOP_v1", "text": "Follow-up interval for benign-appearing leukoplakia: 3 months. If dysplasia suspected, refer to specialist."},
    {"id": "doc4", "source": "Review_OralLesions_2021", "text": "High-risk features: ulceration, induration, rapid growth. Consider excisional biopsy."}
]

class RetrievalService:
    def __init__(self, docs=None):
        self.docs = docs or SAMPLE_DOCS
        texts = [d["text"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english").fit(texts)
        self.tfidf_matrix = self.vectorizer.transform(texts)

    def retrieve(self, query: str, top_k: int = 5):
        qv = self.vectorizer.transform([query])
        scores = linear_kernel(qv, self.tfidf_matrix).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] <= 0:
                continue
            d = dict(self.docs[i])
            d["score"] = float(scores[i])
            results.append(d)
        return results

USE_TGI = os.getenv("USE_TGI", "0") == "1"
TGI_URL = os.getenv("TGI_URL", "http://localhost:8080/generate")

class Generator:
    def __init__(self):
        self.use_tgi = USE_TGI

    def _mock_generate(self, prompt, contexts):
        if not contexts:
            return "I could not find relevant information in the knowledge base."
        top = contexts[0]
        answer = (
            f"Based on the provided references (e.g., {top['source']}), "
            f"{top['text']} \n\n[Note] This is a mock answer. "
            "Replace with LLM output in production."
        )
        return answer

    def _call_tgi(self, prompt):
        payload = {
            "inputs": prompt,
            "parameters": {"temperature": 0.1, "max_new_tokens": 300}
        }
        try:
            r = requests.post(TGI_URL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data[0].get("generated_text", "")
            return data.get("generated_text", "")
        except Exception as e:
            return f"[TGI ERROR] {str(e)}"

    def generate(self, prompt: str, contexts: list):
        if self.use_tgi:
            return self._call_tgi(prompt)
        return self._mock_generate(prompt, contexts)