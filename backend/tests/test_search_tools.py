"""Tests for CourseSearchTool and related search functionality"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults, VectorStore
from models import Course, Lesson


class TestCourseSearchTool:
    """Test CourseSearchTool functionality"""

    def test_get_tool_definition(self):
        """Test that tool definition is properly formatted"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)
        
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_successful_search(self, mock_vector_store, sample_search_results):
        """Test successful search execution"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute("machine learning", course_name="ML Course")
        
        # Verify search was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="machine learning",
            course_name="ML Course",
            lesson_number=None
        )
        
        # Verify result format
        assert "[ML Course - Lesson 1]" in result
        assert "[ML Course - Lesson 2]" in result
        assert "Machine learning is a subset" in result
        assert "Neural networks are inspired" in result

    def test_execute_with_lesson_filter(self, mock_vector_store, sample_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = sample_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute("neural networks", course_name="ML Course", lesson_number=2)
        
        mock_vector_store.search.assert_called_once_with(
            query="neural networks",
            course_name="ML Course",
            lesson_number=2
        )

    def test_execute_search_error(self, mock_vector_store, error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = error_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute("test query")
        
        assert result == "Database connection failed"

    def test_execute_empty_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute("nonexistent topic")
        
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, mock_vector_store, empty_search_results):
        """Test empty results message includes filter information"""
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute("test", course_name="Missing Course", lesson_number=5)
        
        assert "No relevant content found in course 'Missing Course' in lesson 5" in result

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting includes lesson links when available"""
        # Setup search results
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called_once_with("Test Course", 1)
        
        # Verify sources include link
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"

    def test_format_results_without_lesson_links(self, mock_vector_store):
        """Test result formatting when lesson links are not available"""
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify sources include None link
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["link"] is None

    def test_format_results_no_lesson_number(self, mock_vector_store):
        """Test result formatting when lesson number is not available"""
        search_results = SearchResults(
            documents=["Test content"],
            metadata=[{"course_title": "Test Course"}],  # No lesson_number
            distances=[0.1]
        )
        mock_vector_store.search.return_value = search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should not include lesson info in header or sources
        assert "[Test Course]" in result
        assert "- Lesson" not in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"


class TestCourseOutlineTool:
    """Test CourseOutlineTool functionality"""

    def test_get_tool_definition(self):
        """Test that outline tool definition is properly formatted"""
        mock_store = Mock()
        tool = CourseOutlineTool(mock_store)
        
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert definition["input_schema"]["required"] == ["course_title"]

    def test_execute_successful_outline(self, mock_vector_store):
        """Test successful course outline retrieval"""
        # Mock course resolution
        mock_vector_store._resolve_course_name.return_value = "Full Course Title"
        
        # Mock course catalog response
        mock_results = {
            'metadatas': [{
                'course_link': 'https://example.com/course',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "https://example.com/lesson1"}, {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "https://example.com/lesson2"}]'
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Course")
        
        # Verify course resolution was called
        mock_vector_store._resolve_course_name.assert_called_once_with("Course")
        
        # Verify outline content
        assert "Course: Full Course Title" in result
        assert "Course Link: https://example.com/course" in result
        assert "Total Lessons: 2" in result
        assert "Lesson 1: Intro" in result
        assert "Lesson 2: Advanced" in result

    def test_execute_course_not_found(self, mock_vector_store):
        """Test handling when course is not found"""
        mock_vector_store._resolve_course_name.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Nonexistent Course")
        
        assert "No course found matching 'Nonexistent Course'" in result

    def test_execute_metadata_not_found(self, mock_vector_store):
        """Test handling when course metadata is not found"""
        mock_vector_store._resolve_course_name.return_value = "Course Title"
        mock_vector_store.course_catalog.get.return_value = {}
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Course")
        
        assert "Course metadata not found for 'Course Title'" in result

    def test_execute_no_lessons_data(self, mock_vector_store):
        """Test handling when lessons data is missing"""
        mock_vector_store._resolve_course_name.return_value = "Course Title"
        mock_results = {'metadatas': [{}]}  # No lessons_json
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Course")
        
        assert "No lesson information found for 'Course Title'" in result

    def test_execute_json_parse_error(self, mock_vector_store):
        """Test handling of JSON parsing errors"""
        mock_vector_store._resolve_course_name.return_value = "Course Title"
        mock_results = {
            'metadatas': [{
                'lessons_json': 'invalid json'
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Course")
        
        assert "Error retrieving course outline:" in result

    def test_execute_sources_tracking(self, mock_vector_store):
        """Test that sources are properly tracked"""
        mock_vector_store._resolve_course_name.return_value = "Test Course"
        mock_results = {
            'metadatas': [{
                'course_link': 'https://example.com/course',
                'lessons_json': '[{"lesson_number": 1, "lesson_title": "Lesson 1", "lesson_link": "https://example.com/lesson1"}]'
            }]
        }
        mock_vector_store.course_catalog.get.return_value = mock_results
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute("Course")
        
        # Verify sources were tracked
        assert len(tool.last_sources) == 2  # Course + 1 lesson
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] == "https://example.com/course"
        assert tool.last_sources[1]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[1]["link"] == "https://example.com/lesson1"


class TestToolManager:
    """Test ToolManager functionality"""

    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools

    def test_register_tool_without_name(self, mock_vector_store):
        """Test error handling for tool without name"""
        manager = ToolManager()
        
        # Create a mock tool with invalid definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {}  # No name
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        tool_names = [d["name"] for d in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    def test_execute_tool(self, mock_vector_store):
        """Test tool execution"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        # Mock the execute method
        tool.execute = Mock(return_value="Mock result")
        
        manager.register_tool(tool)
        result = manager.execute_tool("search_course_content", query="test")
        
        assert result == "Mock result"
        tool.execute.assert_called_once_with(query="test")

    def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool")
        
        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test getting sources from tools"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = [{"text": "Test Source", "link": "http://example.com"}]
        
        manager.register_tool(tool)
        sources = manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "Test Source"

    def test_get_last_sources_empty(self, mock_vector_store):
        """Test getting sources when none exist"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        tool.last_sources = []
        
        manager.register_tool(tool)
        sources = manager.get_last_sources()
        
        assert sources == []

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        tool1 = CourseSearchTool(mock_vector_store)
        tool2 = CourseOutlineTool(mock_vector_store)
        tool1.last_sources = [{"text": "Source 1"}]
        tool2.last_sources = [{"text": "Source 2"}]
        
        manager.register_tool(tool1)
        manager.register_tool(tool2)
        
        manager.reset_sources()
        
        assert tool1.last_sources == []
        assert tool2.last_sources == []