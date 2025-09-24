# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Server
```bash
# Start the FastAPI server with auto-reload
./run.sh
# OR manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install/sync dependencies (always use uv, never pip directly)
uv sync

# Add new dependencies
uv add <package-name>

# Remove dependencies
uv remove <package-name>
```

### Environment Setup
```bash
# Create .env file with required API key
cp .env.example .env
# Then edit .env to add your ANTHROPIC_API_KEY
```

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a multi-layered architecture:

### Core RAG Pipeline
1. **Document Processing** (`document_processor.py`) - Parses structured course documents with metadata extraction and intelligent text chunking
2. **Vector Storage** (`vector_store.py`) - ChromaDB with dual collections: `course_catalog` (metadata) and `course_content` (chunks)
3. **AI Generation** (`ai_generator.py`) - Anthropic Claude integration with tool-calling capabilities
4. **Search Tools** (`search_tools.py`) - Tool-based search system that Claude can invoke during conversations

### Key Architectural Decisions
- **Dual Vector Collections**: Separate storage for course metadata vs content enables both course discovery and detailed content search
- **Tool-Based Search**: Instead of simple RAG retrieval, uses Claude's tool-calling to dynamically search and filter content during conversation
- **Session Management**: Conversation history tracking with configurable limits
- **Structured Document Format**: Expects specific course document format with lessons, links, and metadata

### Configuration System
All settings centralized in `config.py`:
- Chunk size/overlap for text processing
- ChromaDB path and embedding model
- Anthropic model and API settings
- Search result limits and conversation history

### Document Format Requirements
Documents must follow this structure:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
Lesson Link: [url]
[content...]

Lesson 1: [title]
[content...]
```

### Data Flow
1. Documents in `docs/` folder auto-load on server startup
2. `RAGSystem.add_course_folder()` processes documents → chunks → vector storage
3. User queries via `/api/query` trigger tool-based search through Claude
4. `CourseSearchTool.search()` queries ChromaDB collections
5. Results contextualize Claude's response generation

### API Endpoints
- `POST /api/query` - Main chat interface with session management
- `GET /api/courses` - Course analytics and statistics
- `GET /` - Serves frontend static files

## Development Notes

### Adding New Courses
Place properly formatted course documents in the `docs/` folder. They will be automatically processed on server restart or when explicitly calling `add_course_folder()`.

### Extending Search Capabilities
New search tools should inherit from the base tool class in `search_tools.py` and register with `ToolManager`. Claude can then invoke them during conversations.

### Vector Store Customization
The dual-collection approach in `vector_store.py` allows separate optimization for course discovery vs content search. Modify collection strategies there for different use cases.
- Always use uv to run the server, do not run pip directly