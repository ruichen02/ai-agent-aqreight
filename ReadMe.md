# AI Policy & Product Helper

A local-first RAG (Retrieval-Augmented Generation) system built with FastAPI, Next.js, and Qdrant.

## Setup
Running docker
```bash
- copy env
cp .env.example .env

- run all services
docker compose up --build
```

## Overview

This project implements a document-grounded AI assistant that answers policy and product questions using Retrieval-Augmented Generation (RAG).
It supports Markdown and text file ingestion, vector-based retrieval, and LLM-powered summarization with citations and chunk-level context.

## System Architecture
### Components
| **Layer**          | **Technology**              | **Description** |
|--------------------|-----------------------------|-----------------|
| **Frontend**       | Next.js (React + TypeScript) | Interactive chat UI with auto-scroll, timestamps, and citation display |
| **Backend API**    | FastAPI                     | Handles ingestion, retrieval, and LLM query orchestration |
| **Vector Database**| Qdrant                      | Stores and retrieves document embeddings |
| **LLM Interface**  | OpenAI / Stub LLM           | Generates final responses using retrieved context |


## Data Flow

### Ingestion

- Markdown and text files are parsed into structured sections (title, subsection, body).
- Chunks are embedded and stored in Qdrant with unique, compatible IDs.

### Retrieval

- When a user asks a question, the system computes embeddings and queries Qdrant.
- A hierarchical scoring system evaluates relevance across titles, sections, and body texts.
    - Assists the rag in retrieving more relevant documents to improve the accuracy

### Generation

- The most relevant chunks are passed to the LLM (OpenAI or stub).
- The model generates an answer and cites source files and sections.

### Chat Interface

- Responses are displayed with timestamps, supporting citations, and collapsible chunk details.
- The chat auto-scrolls to the latest message for improved user experience.

## Key Improvements & Features
### ingest.py

- Structured Markdown and text parsing (_md_sections, _txt_sections)
- Updated load_documents to support .md and .txt only
- Automatic file-type detection for correct parsing
- Section the markdown files into three parts: title, section, body for better chunking, accuracy and speed.

### rag.py

- QdrantStore: Ensures Qdrant-compatible IDs on upsert
- StubLLM: Adds relevance scoring and displays section-level scores
- OpenAILLM: Optimized prompt structure, context truncation, and token control
- Metric System: Extended metrics for monitoring retrieval and backlog performance

RAGEngine Enhancements:
- auto_ingest automatically loads all documents in dir/data
- _score_context implements hierarchical weighted matching (title, section, body)

### Chat.tsx

- Initial system greeting message
- Auto-scroll to latest message
- Local timestamps for each message
- Bubble-style layout for user and assistant
- Collapsible sections for citations and supporting chunks

### page.tsx

- Added collapsible Admin Panel for a cleaner and more focused UI

### .env

- Tuned CHUNK_SIZE and CHUNK_OVERLAP for better retrieval quality

## Design Trade-offs
| **Decision** | **Rationale** | **Trade-off** |
|---------------|---------------|---------------|
| **No payload filtering in Qdrant** | Simplifies retrieval logic, relies on hierarchical scoring | Less efficient for large-scale corpora |
| **Compact context string with truncation** | Prevents token overflow and keeps prompt relevant | May omit some lower-ranked sections |
| **Single-directory auto-ingest (`dir/data`)** | Streamlines setup and reduces manual work | Less flexible for multi-source ingestion |

## What I’d Ship Next

- Add evaluation dashboard to monitor retrieval precision and LLM performance
- Implement caching for repeated queries
- Multi-file upload UI for direct ingestion from frontend
- Add a memory for the model to be more interactive with user

