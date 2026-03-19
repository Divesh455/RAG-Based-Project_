# How to use this RAG AI Teaching assistant on your own data

## Step 1 -> Collect files to the videos folder
Move all your videos files to the videos folder.

## Step 2 -> Convert to mp3
Convert all the videos files to mp3 by running video_to_mp3.

## Step 3 -> Convert mp3 to json
Convert all the mp3 files to json by running mp3_to_json.

## Step 4 -> Convert the json files to vectors
Use the preprocess_json to convert the json files to a datafram with Embeddings and save it as joblib pickel.

## Step 5 -> Prompt generation and feeding to LLM
Read the joblib file and load it into the memory.Them create a relevant prompt as per the user query and feed it to the LLM

## FastAPI UI
Run the web UI with:

```bash
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.

### Requirements
- Make sure `Preprocessing/embeddings.joblib` exists.
- Make sure Ollama is running locally on `http://localhost:11434`.
- Default models used by the UI:
  - Embedding model: `bge-m3`
  - Generation model: `llama3.2`

You can change the model names with these environment variables:

```bash
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_CHAT_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```
