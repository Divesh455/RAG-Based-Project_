from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_GENERATION_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass(frozen=True)
class SourceChunk:
    number: str
    title: str
    start: float
    end: float
    text: str
    score: float

    @property
    def start_label(self) -> str:
        return format_timestamp(self.start)

    @property
    def end_label(self) -> str:
        return format_timestamp(self.end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "title": self.title,
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "start_label": self.start_label,
            "end_label": self.end_label,
            "text": self.text,
            "score": round(float(self.score), 4),
        }


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class CourseAssistant:
    def __init__(
        self,
        embeddings_path: str | Path | None = None,
        ollama_base_url: str = DEFAULT_OLLAMA_URL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        generation_model: str = DEFAULT_GENERATION_MODEL,
    ) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.embeddings_path = self._resolve_embeddings_path(embeddings_path)
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embedding_model = embedding_model
        self.generation_model = generation_model

        self.df = joblib.load(self.embeddings_path)
        self.embedding_matrix = np.vstack(self.df["embedding"].values)

    def _resolve_embeddings_path(self, provided_path: str | Path | None) -> Path:
        candidates: list[Path] = []
        if provided_path is not None:
            candidates.append(Path(provided_path))

        candidates.extend(
            [
                self.base_dir / "Preprocessing" / "embeddings.joblib",
                self.base_dir / "embeddings.joblib",
            ]
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        checked = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "Could not find embeddings.joblib. Checked:\n"
            f"{checked}\nRun the preprocessing step before starting the UI."
        )

    def get_stats(self) -> dict[str, Any]:
        lessons = (
            self.df[["number", "title"]]
            .drop_duplicates()
            .sort_values(["number", "title"], kind="stable")
        )
        lesson_records = lessons.to_dict(orient="records")
        return {
            "chunk_count": int(len(self.df)),
            "lesson_count": int(len(lesson_records)),
            "embeddings_path": str(self.embeddings_path),
            "lesson_preview": lesson_records[:6],
        }

    def create_embedding(self, text_list: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/embed",
                json={"model": self.embedding_model, "input": text_list},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["embeddings"]
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not create embeddings from Ollama at {self.ollama_base_url}."
            ) from exc
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Unexpected embedding response returned by Ollama.") from exc

    def generate_response(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.generation_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["response"].strip()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not generate an answer from Ollama at {self.ollama_base_url}."
            ) from exc
        except (KeyError, ValueError) as exc:
            raise RuntimeError("Unexpected generation response returned by Ollama.") from exc

    def find_relevant_chunks(self, question: str, top_k: int = 3) -> list[SourceChunk]:
        if not question.strip():
            raise ValueError("Please enter a question.")

        top_k = max(1, min(int(top_k), 8))
        question_embedding = self.create_embedding([question])[0]
        similarities = cosine_similarity(
            self.embedding_matrix, [question_embedding]
        ).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        chunks: list[SourceChunk] = []
        for row_index in top_indices:
            row = self.df.iloc[int(row_index)]
            chunks.append(
                SourceChunk(
                    number=str(row["number"]),
                    title=str(row["title"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    text=str(row["text"]).strip(),
                    score=float(similarities[int(row_index)]),
                )
            )
        return chunks

    def build_prompt(self, question: str, sources: list[SourceChunk]) -> str:
        source_payload = [
            {
                "title": source.title,
                "number": source.number,
                "start": round(source.start, 2),
                "end": round(source.end, 2),
                "text": source.text,
            }
            for source in sources
        ]

        return f"""
I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time :

{json.dumps(source_payload, ensure_ascii=True)}
---------------------------------------------------------
{question.strip()}
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much cantent is taught in which video (in which video and at what timestamp) and guide the user to go to the particular video.also mention the starting time and ending time and don't say to user i would share chunks to you. If user asks unrelated question, tell him that you can only answer questions related to the course.
""".strip()

    def answer_question(self, question: str, top_k: int = 3) -> dict[str, Any]:
        sources = self.find_relevant_chunks(question=question, top_k=top_k)
        answer = self.generate_response(self.build_prompt(question, sources))
        return {
            "question": question.strip(),
            "answer": answer,
            "sources": [source.to_dict() for source in sources],
        }
        
        
'''
I am teaching web development in my Sigma web development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time :

{json.dumps(source_payload, ensure_ascii=True)}
---------------------------------------------------------
{question.strip()}
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much cantent is taught in which video (in which video and at what timestamp) and guide the user to go to the particular video.also mention the starting time and ending time and don't say to user i would share chunks to you. If user asks unrelated question, tell him that you can only answer questions related to the course.
'''
