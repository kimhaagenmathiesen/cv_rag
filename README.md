# CV RAG Application

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline using [LlamaIndex](https://www.llamaindex.ai/), [ChromaDB](https://www.trychroma.com/), Hugging Face embeddings, and [Ollama](https://ollama.com/) for local LLM serving.

The goal is to load CVs and job postings from PDF files, build a vector database, and generate job applications based on the retrieved context.

## Features

- Parses CVs and job postings from the `./data/` directory (PDF or text)
- Builds or reuses a ChromaDB vector store
- Uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Uses Ollama (`gemma2:9b` model) for LLM generation
- Retrieves and displays relevant context nodes
- Generates a job application draft based on the retrieved context

## Project structure

```
cv_rag.py            # Main script
requirements.txt     # Python dependencies
./data/              # Folder containing your CV and job posting files (PDFs or text)
./chroma_db/         # Generated ChromaDB vector store (auto-created)
```

## Requirements

You can install the necessary packages with:
```
pip install -r requirements.txt
```

Example `requirements.txt`:
```
llama-index-core
llama-index-llms-ollama
llama-index-embeddings-huggingface
llama-index-vector-stores-chroma
llama-index-readers-file
chromadb
ollama
sentence-transformers
PyMuPDF
```

## How to run

1. Place your CV and job posting files (PDFs or text) in the `./data/` folder.
2. Run the script:
```
python3 cv_rag.py
```
3. The script will:
- Print embedding info
- Build or reuse the vector store
- Retrieve relevant context
- Generate a job application draft

## Example query

The system queries:
```
Write a job application based on my CV for the AI Software Engineer position.
```
You can modify this in the code as needed.

## Notes

- Ensure you have an Ollama model (`gemma2:9b`) available locally.
- Clear the `chroma_db/` folder if you want to rebuild the database:
```
rm -rf chroma_db/
```

## License

MIT or specify your preferred license.
