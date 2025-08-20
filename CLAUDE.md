# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
uv sync

# Create .env file with API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Development Utilities
```bash
# Access API documentation
# Navigate to http://localhost:8000/docs after starting

# Add new course documents
# Place files in docs/ directory - server auto-loads on startup
```

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot** with a tool-based search architecture:

### Core Request Flow
1. **Frontend** (`frontend/script.js`) → POST `/api/query`
2. **FastAPI** (`backend/app.py`) → `RAGSystem.query()`
3. **RAG Orchestrator** (`backend/rag_system.py`) → `AIGenerator.generate_response()`
4. **Claude API** (`backend/ai_generator.py`) → Decides whether to use search tools
5. **Search Tools** (`backend/search_tools.py`) → Queries vector store if needed
6. **Vector Store** (`backend/vector_store.py`) → ChromaDB semantic search
7. Response flows back through the chain with sources tracked

### Key Architectural Patterns

**Tool-Based Search**: Claude decides when to search based on query type. Use `CourseSearchTool` for course-specific content, otherwise answers from general knowledge.

**Dual Collection System**: 
- `course_metadata` collection: Course/lesson structure for semantic course name matching
- `course_content` collection: Text chunks for content retrieval

**Session Management**: Conversation context maintained via `SessionManager` with configurable history limits.

**Component Separation**:
- `DocumentProcessor`: Handles file parsing and chunking (800 chars, 100 overlap)
- `VectorStore`: ChromaDB interface with unified search API
- `AIGenerator`: Claude API integration with tool execution logic
- `ToolManager`: Extensible tool registration system

### Configuration
All settings centralized in `backend/config.py` using environment variables and dataclass pattern.

### Data Models
- `Course` → `Lesson` hierarchy for metadata
- `CourseChunk` for vector storage with course/lesson references
- `SearchResults` dataclass for unified query response handling

### Frontend Integration
Single-page application with session persistence, markdown rendering, and collapsible source display. Uses relative URLs for proxy compatibility.
- always use uv to run the server do not use pip directly