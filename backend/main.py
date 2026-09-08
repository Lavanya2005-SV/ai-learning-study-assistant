from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.documents import Document as LCDocument
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def seed_state() -> dict[str, Any]:
    return {
        "documents": [
            {
                "id": "doc-rag-foundations",
                "name": "RAG Foundations — Week 1.md",
                "type": "text/markdown",
                "size": 4860,
                "status": "indexed",
                "uploadedAt": "2026-09-07T09:20:00.000Z",
                "chunkCount": 6,
                "objectPath": None,
                "content": (
                    "Retrieval-Augmented Generation (RAG) combines a retrieval system "
                    "with a language model. Instead of asking a model to answer from "
                    "parameters alone, the system first searches a trusted knowledge "
                    "base and then provides the most relevant passages as context.\n\n"
                    "A typical RAG pipeline has five stages: document ingestion, "
                    "chunking, embedding, retrieval, and grounded generation. Chunking "
                    "breaks long documents into passages that preserve enough meaning "
                    "to stand on their own. Embeddings represent the semantic meaning "
                    "of each passage as a vector. Retrieval compares a question with "
                    "passage vectors and returns the highest-scoring context.\n\n"
                    "RAG is useful because it reduces hallucinations, keeps answers "
                    "tied to current or private material, and makes citations possible. "
                    "A strong system still needs evaluation: measure retrieval precision, "
                    "answer faithfulness, and whether the response actually answers the "
                    "learner's question.\n\n"
                    "Agentic RAG adds a controller that can decide when to retrieve, "
                    "which sources to use, whether memory is relevant, and what action "
                    "to take next. Tools allow the controller to call retrieval, scoring, "
                    "or scheduling functions rather than doing every task in one prompt."
                ),
            }
        ],
        "studyPlans": [
            {
                "id": "plan-agentic-ai",
                "title": "Agentic AI Foundations Sprint",
                "goal": "Prepare for an Agentic AI internship evaluation",
                "durationWeeks": 4,
                "weeklyHours": 5,
                "progress": 38,
                "createdAt": "2026-09-05T07:00:00.000Z",
                "tasks": [
                    {
                        "id": "task-1",
                        "title": "Review RAG pipeline stages",
                        "week": 1,
                        "minutes": 45,
                        "type": "learn",
                        "completed": True,
                    },
                    {
                        "id": "task-2",
                        "title": "Explain embeddings in your own words",
                        "week": 1,
                        "minutes": 30,
                        "type": "practice",
                        "completed": True,
                    },
                    {
                        "id": "task-3",
                        "title": "Build a retrieval evaluation checklist",
                        "week": 1,
                        "minutes": 40,
                        "type": "build",
                        "completed": False,
                    },
                    {
                        "id": "task-4",
                        "title": "Compare tool calling and fixed workflows",
                        "week": 2,
                        "minutes": 45,
                        "type": "learn",
                        "completed": False,
                    },
                ],
            }
        ],
        "quizzes": [],
        "memory": [
            {
                "id": "memory-1",
                "category": "Goal",
                "content": "Prepare for an Agentic AI internship evaluation.",
                "source": "seed",
                "createdAt": "2026-09-05T07:00:00.000Z",
            }
        ],
        "activities": [
            {
                "id": "activity-1",
                "title": "RAG Foundations indexed",
                "detail": "6 chunks are ready for grounded questions",
                "time": "Yesterday",
                "type": "material",
            }
        ],
        "progress": {
            "studyMinutes": 115,
            "questionsAnswered": 0,
            "quizAttempts": 0,
            "averageScore": 0,
            "weeklyMinutes": [
                {"day": "Mon", "minutes": 20},
                {"day": "Tue", "minutes": 35},
                {"day": "Wed", "minutes": 0},
                {"day": "Thu", "minutes": 0},
                {"day": "Fri", "minutes": 0},
                {"day": "Sat", "minutes": 0},
                {"day": "Sun", "minutes": 0},
            ],
            "milestones": ["First course material indexed"],
        },
    }


class JsonStore:
    """Small persistent store that keeps the original API usable without a database."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self.data = seed_state()
        else:
            self.data = seed_state()
            self.save()

    def save(self) -> None:
        with self.lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def mutate(self, callback: Callable[[dict[str, Any]], Any]) -> Any:
        with self.lock:
            result = callback(self.data)
            self.save()
            return result


class DocumentInput(BaseModel):
    name: str
    type: str = "document"
    size: int = Field(default=1, ge=1)
    content: str = ""
    objectPath: str | None = None


class QuestionInput(BaseModel):
    question: str = Field(min_length=2)
    documentIds: list[str] = Field(default_factory=list)


class MemoryInput(BaseModel):
    category: str
    content: str


class StudyPlanInput(BaseModel):
    goal: str = Field(min_length=2)
    durationWeeks: int = Field(default=4, ge=1, le=52)
    weeklyHours: int = Field(default=5, ge=1, le=80)


class TaskUpdate(BaseModel):
    completed: bool


class QuizInput(BaseModel):
    topic: str | None = None
    difficulty: str = "mixed"
    questionCount: int = Field(default=5, ge=1, le=20)
    documentIds: list[str] = Field(default_factory=list)


class QuizSubmission(BaseModel):
    answers: list[int]


class UploadUrlRequest(BaseModel):
    name: str
    size: int = Field(ge=1)
    contentType: str = "application/octet-stream"


class LearningEngine:
    """LangChain-powered RAG, memory, and tool-calling layer."""

    def __init__(self, store: JsonStore):
        self.store = store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", ". ", " "]
        )
        self.llm = (
            ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.2,
            )
            if os.getenv("OPENAI_API_KEY")
            else None
        )

    def documents(self, selected_ids: list[str] | None = None) -> list[dict[str, Any]]:
        documents = self.store.data["documents"]
        if not selected_ids:
            return documents
        return [document for document in documents if document["id"] in selected_ids]

    def retrieve(self, query: str, selected_ids: list[str] | None = None) -> list[dict[str, Any]]:
        query_terms = {
            term.lower()
            for term in re.findall(r"[a-zA-Z0-9]{3,}", query)
        }
        candidates: list[dict[str, Any]] = []
        for document in self.documents(selected_ids):
            chunks = self.splitter.create_documents(
                [document.get("content", "")],
                metadatas=[{"document": document["name"], "documentId": document["id"]}],
            )
            for chunk in chunks:
                words = set(re.findall(r"[a-zA-Z0-9]{3,}", chunk.page_content.lower()))
                overlap = len(query_terms & words)
                score = overlap / max(len(query_terms), 1)
                if score > 0 or not query_terms:
                    candidates.append(
                        {
                            "document": document["name"],
                            "documentId": document["id"],
                            "excerpt": chunk.page_content.strip(),
                            "relevance": min(1.0, round(score + 0.05, 3)),
                        }
                    )
        candidates.sort(key=lambda item: item["relevance"], reverse=True)
        return candidates[:4] or [
            {
                "document": document["name"],
                "documentId": document["id"],
                "excerpt": document.get("content", "")[:600],
                "relevance": 0.05,
            }
            for document in self.documents(selected_ids)[:1]
        ]

    def memory_text(self) -> str:
        return "\n".join(
            f"- {item['category']}: {item['content']}"
            for item in self.store.data.get("memory", [])
        )

    def tools(self, selected_ids: list[str]) -> list[StructuredTool]:
        def retrieve_tool(query: str) -> str:
            return json.dumps(self.retrieve(query, selected_ids), ensure_ascii=False)

        def memory_tool(_: str = "") -> str:
            return self.memory_text() or "No learner memory has been saved."

        def progress_tool(_: str = "") -> str:
            return json.dumps(self.store.data["progress"])

        return [
            StructuredTool.from_function(
                retrieve_tool,
                name="retrieve_course_material",
                description="Retrieve relevant passages from the learner's uploaded course materials.",
            ),
            StructuredTool.from_function(
                memory_tool,
                name="read_learner_memory",
                description="Read preferences, constraints, strengths, and goals saved for this learner.",
            ),
            StructuredTool.from_function(
                progress_tool,
                name="read_learning_progress",
                description="Read the learner's current study minutes, quiz scores, and milestones.",
            ),
        ]

    def _fallback_answer(self, question: str, citations: list[dict[str, Any]]) -> str:
        if not citations:
            return (
                "I could not find a matching passage in the uploaded course material. "
                "Add a source document or rephrase the question."
            )
        context = " ".join(item["excerpt"] for item in citations[:2])
        return (
            f"Based on your course material: {context[:900]}\n\n"
            f"For your question, focus on the terms that overlap with “{question}”. "
            "A useful next step is to explain the idea in your own words and test it with a short example."
        )

    def answer(self, request: QuestionInput) -> dict[str, Any]:
        citations = self.retrieve(request.question, request.documentIds)
        steps = [
            "Retrieve relevant passages from selected course materials",
            "Read learner memory and progress signals",
            "Compose a grounded explanation with source citations",
        ]
        answer = self._fallback_answer(request.question, citations)

        if self.llm:
            tools = self.tools(request.documentIds)
            tool_map = {tool.name: tool for tool in tools}
            prompt = (
                "You are Lumen, a careful learning companion. Answer the learner's "
                "question using retrieved course context. Never invent facts that are "
                "not in the context. Mention when the sources are insufficient. Keep "
                "the explanation practical and concise.\n\n"
                f"Course context:\n{json.dumps(citations, ensure_ascii=False)}\n\n"
                f"Learner memory:\n{self.memory_text() or 'None'}"
            )
            messages: list[Any] = [
                SystemMessage(content=prompt),
                HumanMessage(content=request.question),
            ]
            try:
                model = self.llm.bind_tools(tools)
                for _ in range(3):
                    response = model.invoke(messages)
                    messages.append(response)
                    calls = getattr(response, "tool_calls", [])
                    if not calls:
                        answer = response.content if isinstance(response.content, str) else str(response.content)
                        break
                    for call in calls:
                        tool = tool_map.get(call["name"])
                        if not tool:
                            continue
                        output = tool.invoke(call.get("args", {}))
                        messages.append(
                            ToolMessage(content=str(output), tool_call_id=call["id"])
                        )
                steps = [
                    "Agent decided which learning tools were relevant",
                    "Retrieved and compared grounded course passages",
                    "Checked learner memory before composing the answer",
                    "Returned a cited explanation",
                ]
            except Exception:
                # A missing, unavailable, or rate-limited model should not take down
                # the learning workspace; the deterministic RAG fallback still works.
                steps.append("LLM unavailable; returned a deterministic grounded fallback")

        self.store.mutate(
            lambda data: data["progress"].update(
                {"questionsAnswered": data["progress"]["questionsAnswered"] + 1}
            )
        )
        return {
            "answer": answer,
            "confidence": "high" if citations and citations[0]["relevance"] >= 0.25 else "medium",
            "citations": citations,
            "agentSteps": steps,
        }

    def generate_plan(self, request: StudyPlanInput) -> dict[str, Any]:
        plan_id = new_id("plan")
        tasks = [
            {
                "id": new_id("task"),
                "title": f"Study {request.goal}",
                "week": week,
                "minutes": max(25, round(request.weeklyHours * 60 / 5)),
                "type": "learn" if week % 2 else "practice",
                "completed": False,
            }
            for week in range(1, request.durationWeeks + 1)
        ]
        plan = {
            "id": plan_id,
            "title": f"{request.goal} study plan",
            "goal": request.goal,
            "durationWeeks": request.durationWeeks,
            "weeklyHours": request.weeklyHours,
            "progress": 0,
            "createdAt": now_iso(),
            "tasks": tasks,
        }
        self.store.mutate(lambda data: data["studyPlans"].insert(0, plan))
        return plan

    def generate_quiz(self, request: QuizInput) -> dict[str, Any]:
        source = self.retrieve(request.topic or "key concepts", request.documentIds)
        topic = request.topic or "the uploaded course material"
        questions = [
            {
                "id": new_id("question"),
                "prompt": f"Which statement best describes {topic}?",
                "options": [
                    source[0]["excerpt"][:180] if source else "A grounded learning workflow",
                    "A system that never uses source material",
                    "A way to avoid checking understanding",
                    "A replacement for all learner judgment",
                ],
                "answer": 0,
                "explanation": "The first option is the one grounded in the selected course material.",
            }
        ]
        while len(questions) < request.questionCount:
            index = len(questions) + 1
            questions.append(
                {
                    "id": new_id("question"),
                    "prompt": f"Which learning practice supports question {index} about {topic}?",
                    "options": [
                        "Retrieve context, explain it, and check the evidence",
                        "Skip the source and guess",
                        "Memorize an answer without understanding",
                        "Avoid reviewing incorrect answers",
                    ],
                    "answer": 0,
                    "explanation": "Active retrieval with evidence creates a stronger learning loop.",
                }
            )
        quiz = {
            "id": new_id("quiz"),
            "title": f"{topic} practice quiz",
            "difficulty": request.difficulty,
            "createdAt": now_iso(),
            "questions": questions,
        }
        self.store.mutate(lambda data: data["quizzes"].insert(0, quiz))
        return quiz


STATE_FILE = os.getenv("LEARNING_STATE_FILE", "./data/learning_state.json")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
store = JsonStore(STATE_FILE)
engine = LearningEngine(store)

app = FastAPI(title="AI Learning & Study Assistant", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    return store.data["documents"]


@app.post("/api/documents", status_code=201)
def create_document(document: DocumentInput) -> dict[str, Any]:
    chunks = engine.splitter.split_text(document.content or document.name)
    item = {
        "id": new_id("doc"),
        **document.model_dump(),
        "status": "indexed",
        "uploadedAt": now_iso(),
        "chunkCount": max(1, len(chunks)),
    }
    store.mutate(lambda data: data["documents"].insert(0, item))
    store.mutate(
        lambda data: data["activities"].insert(
            0,
            {
                "id": new_id("activity"),
                "title": f"{document.name} indexed",
                "detail": f"{item['chunkCount']} chunks are ready for grounded questions",
                "time": "Just now",
                "type": "material",
            },
        )
    )
    return item


@app.delete("/api/documents/{document_id}", status_code=204)
def delete_document(document_id: str) -> None:
    def remove(data: dict[str, Any]) -> None:
        data["documents"] = [
            document for document in data["documents"] if document["id"] != document_id
        ]

    store.mutate(remove)


@app.post("/api/qa")
def ask_question(question: QuestionInput) -> dict[str, Any]:
    return engine.answer(question)


@app.get("/api/study-plans")
def list_study_plans() -> list[dict[str, Any]]:
    return store.data["studyPlans"]


@app.post("/api/study-plans", status_code=201)
def create_study_plan(plan: StudyPlanInput) -> dict[str, Any]:
    return engine.generate_plan(plan)


@app.patch("/api/study-plans/{plan_id}/tasks/{task_id}")
def update_task(plan_id: str, task_id: str, update: TaskUpdate) -> dict[str, Any]:
    result: dict[str, Any] | None = None

    def mutate(data: dict[str, Any]) -> None:
        nonlocal result
        plan = next((item for item in data["studyPlans"] if item["id"] == plan_id), None)
        if not plan:
            return
        for task in plan["tasks"]:
            if task["id"] == task_id:
                task["completed"] = update.completed
        total = len(plan["tasks"])
        completed = sum(task["completed"] for task in plan["tasks"])
        plan["progress"] = round(completed / total * 100) if total else 0
        result = plan

    store.mutate(mutate)
    if result is None:
        raise HTTPException(status_code=404, detail="Study plan not found")
    return result


@app.get("/api/quizzes")
def list_quizzes() -> list[dict[str, Any]]:
    return store.data["quizzes"]


@app.post("/api/quizzes", status_code=201)
def create_quiz(quiz: QuizInput) -> dict[str, Any]:
    return engine.generate_quiz(quiz)


@app.post("/api/quizzes/{quiz_id}/submit")
def submit_quiz(quiz_id: str, submission: QuizSubmission) -> dict[str, Any]:
    quiz = next((item for item in store.data["quizzes"] if item["id"] == quiz_id), None)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    score = sum(
        index < len(submission.answers) and submission.answers[index] == question["answer"]
        for index, question in enumerate(quiz["questions"])
    )
    total = len(quiz["questions"])
    percentage = round(score / total * 100) if total else 0

    def mutate(data: dict[str, Any]) -> None:
        progress = data["progress"]
        attempts = progress["quizAttempts"]
        progress["quizAttempts"] = attempts + 1
        progress["averageScore"] = round(
            ((progress["averageScore"] * attempts) + percentage) / (attempts + 1)
        )
        progress["questionsAnswered"] += total

    store.mutate(mutate)
    return {
        "quizId": quiz_id,
        "score": score,
        "total": total,
        "percentage": percentage,
        "feedback": "Strong recall—review the cited material once more." if percentage >= 70 else "Review the explanations and try the quiz again.",
    }


@app.get("/api/progress")
def get_progress() -> dict[str, Any]:
    return store.data["progress"]


@app.get("/api/memory")
def list_memory() -> list[dict[str, Any]]:
    return store.data["memory"]


@app.post("/api/memory", status_code=201)
def create_memory(memory: MemoryInput) -> dict[str, Any]:
    item = {
        "id": new_id("memory"),
        **memory.model_dump(),
        "source": "added by you",
        "createdAt": now_iso(),
    }
    store.mutate(lambda data: data["memory"].insert(0, item))
    return item


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    progress = store.data["progress"]
    plans = store.data["studyPlans"]
    active_plan = plans[0] if plans else None
    return {
        "learner": {
            "name": "Ari",
            "goal": active_plan["goal"] if active_plan else "Turn today's curiosity into exam confidence.",
        },
        "stats": {
            "studyMinutes": progress["studyMinutes"],
            "questionsAnswered": progress["questionsAnswered"],
            "materials": len(store.data["documents"]),
            "currentStreak": 2 if progress["studyMinutes"] else 0,
        },
        "planProgress": {
            "title": active_plan["title"] if active_plan else "Start a study plan",
            "percentage": active_plan["progress"] if active_plan else 0,
            "nextTask": next(
                (
                    task["title"]
                    for task in active_plan["tasks"]
                    if not task["completed"]
                ),
                "Create your first learning task",
            )
            if active_plan
            else "Create your first learning task",
        },
        "recentActivity": store.data["activities"][:5],
    }


@app.get("/api/report")
def report() -> dict[str, Any]:
    progress = store.data["progress"]
    return {
        "title": "AI Learning & Study Assistant",
        "problemStatement": "Learners often study across disconnected notes without grounded explanations, adaptive plans, or visible progress.",
        "description": "A personalized workspace that combines document-grounded Q&A, learner memory, planning, quizzes, and progress tracking.",
        "objectives": [
            "Ground answers in uploaded course material",
            "Use memory to personalize future learning support",
            "Turn goals into actionable study plans and quizzes",
            "Make learning progress visible",
        ],
        "solution": [
            "FastAPI provides a Python API compatible with the existing dashboard",
            "LangChain coordinates chunking, retrieval, memory, and tool calling",
            "A persistent JSON store keeps the demo easy to run locally",
        ],
        "technologies": ["Python", "FastAPI", "LangChain", "RAG", "Tool calling", "Pydantic"],
        "workflow": [
            "Ingest and split course material into retrievable chunks",
            "Select relevant passages for the learner's question",
            "Let the LangChain model call retrieval, memory, and progress tools",
            "Return a grounded answer with citations and an agent trace",
        ],
        "results": [
            f"{len(store.data['documents'])} course materials indexed",
            f"{progress['questionsAnswered']} grounded questions answered",
            f"{progress['quizAttempts']} quiz attempts recorded",
        ],
        "conclusion": "The Python backend turns the original interface into a working LangChain learning assistant while retaining a safe fallback when no model key is configured.",
        "challenges": [
            "Keeping the original frontend contract stable during the backend migration",
            "Returning useful answers when an external LLM is unavailable",
        ],
        "futureScope": [
            "Replace lexical retrieval with a production vector database",
            "Add PDF parsing and multimodal document understanding",
            "Add authentication and per-user state",
        ],
        "references": [
            "LangChain documentation",
            "FastAPI documentation",
            "Retrieval-Augmented Generation research literature",
        ],
    }


@app.post("/api/storage/uploads/request-url")
def request_upload_url(request: Request, upload: UploadUrlRequest) -> dict[str, Any]:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "-", upload.name)[-80:]
    object_name = f"{uuid.uuid4().hex}-{safe_name}"
    base_url = str(request.base_url).rstrip("/")
    object_path = f"/objects/{object_name}"
    return {
        "uploadURL": f"{base_url}/api/storage/objects/{object_name}",
        "objectPath": object_path,
        "metadata": upload.model_dump(),
    }


@app.put("/api/storage/objects/{object_name:path}")
async def upload_object(request: Request, object_name: str) -> dict[str, str]:
    target = UPLOAD_DIR / Path(object_name).name
    # The existing React client sends the browser File as a raw PUT body,
    # not as multipart/form-data.
    target.write_bytes(await request.body())
    return {"status": "uploaded", "objectPath": f"/objects/{target.name}"}


@app.get("/api/storage/objects/{object_name:path}")
def get_object(object_name: str) -> FileResponse:
    target = UPLOAD_DIR / Path(object_name).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Object not found")
    return FileResponse(target)


@app.get("/")
def root() -> JSONResponse:
    return JSONResponse(
        {
            "name": "AI Learning & Study Assistant",
            "backend": "FastAPI + LangChain",
            "docs": "/docs",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )