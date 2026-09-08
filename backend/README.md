# Python + LangChain backend

This backend replaces the original Express/TypeScript API while keeping the
same `/api/*` routes used by the existing React dashboard.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cp backend/.env.example .env
uvicorn backend.main:app --reload --port 8000
```

The interactive API docs are available at `http://localhost:8000/docs`.

## AI configuration

Without `OPENAI_API_KEY`, the application still runs and uses deterministic
lexical retrieval plus grounded fallback responses. With a key configured,
LangChain uses `ChatOpenAI.bind_tools()` to let the model call:

- `retrieve_course_material`
- `read_learner_memory`
- `read_learning_progress`

The response includes citations and an agent trace for the dashboard.

## Storage

State is persisted to `data/learning_state.json` by default and uploaded
objects are stored in `data/uploads`. Set `LEARNING_STATE_FILE` and
`UPLOAD_DIR` to move them to another location.