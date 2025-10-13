"""
Shared pytest fixtures and helpers for RAG system tests
"""
import pytest
from unittest.mock import Mock
import sys
import os
import json

# Add backend to path so tests can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


# ============================================================================
# Test Data Fixtures
# ============================================================================

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
            content="Course Building Towards Computer Use with Anthropic Lesson 0 content: Welcome to the course.",
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
    ]


@pytest.fixture
def sample_search_results():
    """Create sample search results"""
    return SearchResults(
        documents=[
            "Anthropic is an AI safety company focused on building reliable AI systems.",
            "Claude is Anthropic's AI assistant that can help with various tasks."
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0, "chunk_index": 0},
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1, "chunk_index": 2}
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[], error=None)


@pytest.fixture
def error_search_results():
    """Create search results with an error"""
    return SearchResults(
        documents=[], metadata=[], distances=[],
        error="No course found matching 'NonExistentCourse'"
    )


# ============================================================================
# Vector Store Mocks
# ============================================================================

@pytest.fixture
def mock_vector_store():
    """Create a basic mock VectorStore"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[], error=None)
    mock_store.get_lesson_link.return_value = "https://example.com/lesson0"
    mock_store.get_course_outline.return_value = None
    return mock_store


@pytest.fixture
def mock_vector_store_with_results(sample_search_results):
    """Create a mock VectorStore that returns sample results"""
    mock_store = Mock()
    mock_store.search.return_value = sample_search_results
    mock_store.get_lesson_link.return_value = "https://example.com/lesson0"
    return mock_store


# ============================================================================
# Azure OpenAI Mock Helpers
# ============================================================================

def create_mock_response(content, tool_calls=None):
    """Helper to create a mock Azure OpenAI response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = content
    mock_response.choices[0].message.tool_calls = tool_calls
    return mock_response


def create_mock_tool_call(tool_id, function_name, arguments):
    """Helper to create a mock tool call"""
    tool_call = Mock()
    tool_call.id = tool_id
    tool_call.function.name = function_name
    tool_call.function.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
    return tool_call


@pytest.fixture
def mock_ai_client():
    """Create a basic mock Azure OpenAI client"""
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = create_mock_response("Test response from AI.")
    return mock_client


@pytest.fixture
def mock_ai_client_with_tool_call():
    """Create a mock Azure OpenAI client that uses a tool"""
    mock_client = Mock()

    # First response: AI wants to use tool
    tool_call = create_mock_tool_call(
        "call_123",
        "search_course_content",
        {"query": "What is Anthropic?", "course_name": "Computer Use"}
    )
    initial_response = create_mock_response(None, [tool_call])

    # Final response: AI provides answer after tool execution
    final_response = create_mock_response("Based on the course material, Anthropic is an AI safety company.")

    mock_client.chat.completions.create.side_effect = [initial_response, final_response]
    return mock_client


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def test_config():
    """Create a test configuration with correct values"""
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
        MAX_RESULTS: int = 5  # Correct value
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


# ============================================================================
# Helper Functions
# ============================================================================

def assert_tool_called_with(mock_client, tool_name, **expected_params):
    """Helper to assert a tool was called with specific parameters"""
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "tools" in call_kwargs
    tools = call_kwargs["tools"]
    tool_names = [t["function"]["name"] for t in tools]
    assert tool_name in tool_names, f"Tool {tool_name} not found in {tool_names}"


def get_messages_from_call(mock_client, call_index=0):
    """Helper to extract messages from a mock client call"""
    calls = mock_client.chat.completions.create.call_args_list
    if call_index >= len(calls):
        raise IndexError(f"Call index {call_index} out of range (only {len(calls)} calls)")
    return calls[call_index][1]["messages"]
