"""
Shared pytest fixtures for RAG system tests
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys
import os

# Add backend to path so tests can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://example.com/course1",
        instructor="Colt Steele",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Overview", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Working With The API", lesson_link="https://example.com/lesson2"),
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 0 content: Welcome to the course. This lesson introduces Anthropic and Claude.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="You will learn about computer use, tool calling, and prompt engineering.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 1 content: This lesson covers the overview of Anthropic's models.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=1,
            chunk_index=2
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for testing"""
    mock_store = Mock()

    # Default behavior: return empty results
    mock_store.search.return_value = SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )

    # Mock other methods
    mock_store.get_lesson_link.return_value = "https://example.com/lesson0"
    mock_store.get_course_outline.return_value = None

    return mock_store


@pytest.fixture
def mock_vector_store_with_results():
    """Create a mock VectorStore that returns sample results"""
    mock_store = Mock()

    # Return sample search results
    mock_store.search.return_value = SearchResults(
        documents=[
            "Welcome to the course. This lesson introduces Anthropic and Claude.",
            "You will learn about computer use, tool calling, and prompt engineering."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.15],
        error=None
    )

    mock_store.get_lesson_link.return_value = "https://example.com/lesson0"

    return mock_store


@pytest.fixture
def mock_ai_client():
    """Create a mock Azure OpenAI client"""
    mock_client = Mock()

    # Create a mock response structure
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "This is a test response from the AI."
    mock_response.choices[0].message.tool_calls = None

    mock_client.chat.completions.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_ai_client_with_tool_call():
    """Create a mock Azure OpenAI client that returns a tool call"""
    mock_client = Mock()

    # First response with tool call
    initial_response = Mock()
    initial_response.choices = [Mock()]
    initial_response.choices[0].message.content = None

    # Create mock tool call
    tool_call = Mock()
    tool_call.id = "call_123"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "What is Anthropic?", "course_name": "Computer Use"}'
    initial_response.choices[0].message.tool_calls = [tool_call]

    # Final response after tool execution
    final_response = Mock()
    final_response.choices = [Mock()]
    final_response.choices[0].message.content = "Based on the course material, Anthropic is an AI safety company."
    final_response.choices[0].message.tool_calls = None

    # Return initial response first, then final response
    mock_client.chat.completions.create.side_effect = [initial_response, final_response]

    return mock_client


@pytest.fixture
def sample_search_results():
    """Create sample search results"""
    return SearchResults(
        documents=[
            "Anthropic is an AI safety company focused on building reliable, interpretable, and steerable AI systems.",
            "Claude is Anthropic's AI assistant that can help with various tasks."
        ],
        metadata=[
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 0,
                "chunk_index": 0
            },
            {
                "course_title": "Building Towards Computer Use with Anthropic",
                "lesson_number": 1,
                "chunk_index": 2
            }
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def sample_empty_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def sample_error_results():
    """Create search results with an error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="No course found matching 'NonExistentCourse'"
    )


@pytest.fixture
def test_config():
    """Create a test configuration"""
    from dataclasses import dataclass

    @dataclass
    class TestConfig:
        AZURE_OPENAI_ENDPOINT: str = "https://test.openai.azure.com"
        AZURE_OPENAI_API_KEY: str = "test-key"
        AZURE_OPENAI_API_VERSION: str = "2024-02-01"
        AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        CHUNK_SIZE: int = 800
        CHUNK_OVERLAP: int = 100
        MAX_RESULTS: int = 5  # Correct value for testing
        MAX_HISTORY: int = 2
        CHROMA_PATH: str = "./test_chroma_db"

    return TestConfig()


@pytest.fixture
def broken_config():
    """Create a configuration with the MAX_RESULTS bug"""
    from dataclasses import dataclass

    @dataclass
    class BrokenConfig:
        AZURE_OPENAI_ENDPOINT: str = "https://test.openai.azure.com"
        AZURE_OPENAI_API_KEY: str = "test-key"
        AZURE_OPENAI_API_VERSION: str = "2024-02-01"
        AZURE_OPENAI_DEPLOYMENT: str = "gpt-4"
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        CHUNK_SIZE: int = 800
        CHUNK_OVERLAP: int = 100
        MAX_RESULTS: int = 0  # BUG: This causes 0 search results!
        MAX_HISTORY: int = 2
        CHROMA_PATH: str = "./test_chroma_db"

    return BrokenConfig()
