"""Pytest configuration and shared fixtures for RAG chatbot tests"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any
import sys

# Add backend directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def temp_config():
    """Create a temporary config for testing"""
    return Config(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="claude-sonnet-4-20250514",
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=50,
        MAX_RESULTS=5,  # Fixed from 0 to 5 for testing
        MAX_HISTORY=2,
        CHROMA_PATH="./test_chroma_db"
    )


@pytest.fixture
def temp_chroma_dir():
    """Create temporary ChromaDB directory"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(lesson_number=1, title="Basic Concepts", lesson_link="https://example.com/ml-course/lesson1"),
            Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml-course/lesson2"),
            Lesson(lesson_number=3, title="Unsupervised Learning", lesson_link="https://example.com/ml-course/lesson3")
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn a mapping function.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Unsupervised learning finds patterns in data without labeled examples.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    
    # Mock search method
    def mock_search(query, course_name=None, lesson_number=None, limit=None):
        # Return different results based on query
        if "machine learning" in query.lower():
            return SearchResults(
                documents=["Machine learning is a subset of artificial intelligence."],
                metadata=[{"course_title": "Introduction to Machine Learning", "lesson_number": 1}],
                distances=[0.1]
            )
        elif "error" in query.lower():
            return SearchResults.empty("Search error occurred")
        else:
            return SearchResults(documents=[], metadata=[], distances=[])
    
    mock_store.search = mock_search
    mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson1")
    
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()
    
    # Mock successful response
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock Anthropic response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.input = {"query": "machine learning"}
    mock_tool_block.id = "tool_123"
    
    mock_response.content = [mock_tool_block]
    
    return mock_response


@pytest.fixture
def sample_search_results():
    """Create sample search results"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are inspired by biological neural networks."
        ],
        metadata=[
            {"course_title": "ML Course", "lesson_number": 1},
            {"course_title": "ML Course", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create search results with error"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Test search results"
    mock_manager.get_last_sources.return_value = []
    mock_manager.reset_sources.return_value = None
    
    return mock_manager