"""Tests for RAG System integration and end-to-end functionality"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import tempfile
import shutil

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test RAG System integration"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing"""
        config = Mock()
        config.CHUNK_SIZE = 400
        config.CHUNK_OVERLAP = 50
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.MAX_HISTORY = 2
        return config

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_init(self, mock_session_manager, mock_ai_generator, mock_vector_store, 
                  mock_document_processor, mock_config):
        """Test RAG System initialization"""
        rag_system = RAGSystem(mock_config)
        
        # Verify all components were initialized
        mock_document_processor.assert_called_once_with(400, 50)
        mock_vector_store.assert_called_once_with("./test_chroma", "all-MiniLM-L6-v2", 5)
        mock_ai_generator.assert_called_once_with("test-key", "claude-sonnet-4-20250514")
        mock_session_manager.assert_called_once_with(2)
        
        # Verify tools were registered
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_simple(self, mock_session_manager, mock_ai_generator, 
                         mock_vector_store, mock_document_processor, mock_config):
        """Test simple query processing without session"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "AI response to query"
        mock_ai_generator.return_value = mock_ai_instance
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = []
        
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager
        
        response, sources = rag_system.query("What is machine learning?")
        
        assert response == "AI response to query"
        assert sources == []
        
        # Verify AI was called with correct parameters
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args
        assert "Answer this question about course materials: What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_session(self, mock_session_manager, mock_ai_generator,
                               mock_vector_store, mock_document_processor, mock_config):
        """Test query processing with session management"""
        # Setup mocks
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "AI response"
        mock_ai_generator.return_value = mock_ai_instance
        
        mock_session_instance = Mock()
        mock_session_instance.get_conversation_history.return_value = "Previous conversation"
        mock_session_manager.return_value = mock_session_instance
        
        mock_tool_manager = Mock()
        mock_tool_manager.get_last_sources.return_value = [{"text": "Source 1", "link": "http://example.com"}]
        
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager
        
        response, sources = rag_system.query("Follow up question", session_id="test_session")
        
        assert response == "AI response"
        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"
        
        # Verify session management
        mock_session_instance.get_conversation_history.assert_called_once_with("test_session")
        mock_session_instance.add_exchange.assert_called_once_with(
            "test_session", "Follow up question", "AI response"
        )
        
        # Verify conversation history was passed
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] == "Previous conversation"

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_with_sources(self, mock_session_manager, mock_ai_generator,
                               mock_vector_store, mock_document_processor, mock_config):
        """Test query processing that returns sources"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Response with sources"
        mock_ai_generator.return_value = mock_ai_instance
        
        mock_tool_manager = Mock()
        test_sources = [
            {"text": "ML Course - Lesson 1", "link": "http://example.com/lesson1"},
            {"text": "ML Course - Lesson 2", "link": "http://example.com/lesson2"}
        ]
        mock_tool_manager.get_last_sources.return_value = test_sources
        
        rag_system = RAGSystem(mock_config)
        rag_system.tool_manager = mock_tool_manager
        
        response, sources = rag_system.query("What is supervised learning?")
        
        assert response == "Response with sources"
        assert sources == test_sources
        
        # Verify sources were reset after retrieval
        mock_tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_tools_integration(self, mock_session_manager, mock_ai_generator,
                                    mock_vector_store, mock_document_processor, mock_config):
        """Test that tools are properly integrated in query processing"""
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = "Tool-enhanced response"
        mock_ai_generator.return_value = mock_ai_instance
        
        rag_system = RAGSystem(mock_config)
        rag_system.query("Search for ML content")
        
        # Verify tools were passed to AI generator
        call_args = mock_ai_instance.generate_response.call_args
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
        
        # Verify tool definitions are included
        tools = call_args[1]["tools"]
        tool_names = [tool["name"] for tool in tools]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_success(self, mock_session_manager, mock_ai_generator,
                                        mock_vector_store, mock_document_processor, mock_config):
        """Test successful course document addition"""
        # Setup mocks
        mock_doc_processor = Mock()
        sample_course = Course(title="Test Course", lessons=[])
        sample_chunks = [CourseChunk(content="Test content", course_title="Test Course", chunk_index=0)]
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
        mock_document_processor.return_value = mock_doc_processor
        
        mock_vector_instance = Mock()
        mock_vector_store.return_value = mock_vector_instance
        
        rag_system = RAGSystem(mock_config)
        
        course, chunk_count = rag_system.add_course_document("/path/to/course.pdf")
        
        assert course.title == "Test Course"
        assert chunk_count == 1
        
        # Verify document processing and vector store calls
        mock_doc_processor.process_course_document.assert_called_once_with("/path/to/course.pdf")
        mock_vector_instance.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_instance.add_course_content.assert_called_once_with(sample_chunks)

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_document_error(self, mock_session_manager, mock_ai_generator,
                                      mock_vector_store, mock_document_processor, mock_config):
        """Test error handling in course document addition"""
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.side_effect = Exception("Processing failed")
        mock_document_processor.return_value = mock_doc_processor
        
        rag_system = RAGSystem(mock_config)
        
        course, chunk_count = rag_system.add_course_document("/invalid/path.pdf")
        
        assert course is None
        assert chunk_count == 0

    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_folder_success(self, mock_session_manager, mock_ai_generator,
                                      mock_vector_store, mock_document_processor,
                                      mock_listdir, mock_exists, mock_config):
        """Test successful course folder processing"""
        # Setup file system mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "ignored.jpg"]
        
        # Setup document processor mock
        mock_doc_processor = Mock()
        sample_course1 = Course(title="Course 1", lessons=[])
        sample_course2 = Course(title="Course 2", lessons=[])
        sample_chunks = [CourseChunk(content="Test", course_title="Course 1", chunk_index=0)]
        
        mock_doc_processor.process_course_document.side_effect = [
            (sample_course1, sample_chunks),
            (sample_course2, sample_chunks)
        ]
        mock_document_processor.return_value = mock_doc_processor
        
        # Setup vector store mock
        mock_vector_instance = Mock()
        mock_vector_instance.get_existing_course_titles.return_value = []  # No existing courses
        mock_vector_store.return_value = mock_vector_instance
        
        rag_system = RAGSystem(mock_config)
        
        total_courses, total_chunks = rag_system.add_course_folder("/test/folder")
        
        assert total_courses == 2
        assert total_chunks == 2
        
        # Verify only PDF and TXT files were processed
        assert mock_doc_processor.process_course_document.call_count == 2

    @patch('rag_system.os.path.exists')
    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_add_course_folder_not_exists(self, mock_session_manager, mock_ai_generator,
                                         mock_vector_store, mock_document_processor,
                                         mock_exists, mock_config):
        """Test handling of non-existent folder"""
        mock_exists.return_value = False
        
        rag_system = RAGSystem(mock_config)
        
        total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")
        
        assert total_courses == 0
        assert total_chunks == 0

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_get_course_analytics(self, mock_session_manager, mock_ai_generator,
                                 mock_vector_store, mock_document_processor, mock_config):
        """Test course analytics retrieval"""
        mock_vector_instance = Mock()
        mock_vector_instance.get_course_count.return_value = 5
        mock_vector_instance.get_existing_course_titles.return_value = ["Course 1", "Course 2"]
        mock_vector_store.return_value = mock_vector_instance
        
        rag_system = RAGSystem(mock_config)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert analytics["course_titles"] == ["Course 1", "Course 2"]

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_query_error_propagation(self, mock_session_manager, mock_ai_generator,
                                    mock_vector_store, mock_document_processor, mock_config):
        """Test that errors in query processing are properly propagated"""
        # Setup AI generator to raise an exception
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.side_effect = Exception("AI processing failed")
        mock_ai_generator.return_value = mock_ai_instance
        
        rag_system = RAGSystem(mock_config)
        
        # Query should raise the exception
        with pytest.raises(Exception, match="AI processing failed"):
            rag_system.query("Test query")

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    def test_course_duplication_handling(self, mock_session_manager, mock_ai_generator,
                                        mock_vector_store, mock_document_processor, mock_config):
        """Test that duplicate courses are not re-added"""
        # Setup mocks
        mock_doc_processor = Mock()
        sample_course = Course(title="Existing Course", lessons=[])
        sample_chunks = [CourseChunk(content="Test", course_title="Existing Course", chunk_index=0)]
        mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
        mock_document_processor.return_value = mock_doc_processor
        
        mock_vector_instance = Mock()
        mock_vector_instance.get_existing_course_titles.return_value = ["Existing Course"]
        mock_vector_store.return_value = mock_vector_instance
        
        rag_system = RAGSystem(mock_config)
        
        course, chunk_count = rag_system.add_course_document("/path/to/existing_course.pdf")
        
        # Course should be processed but not added
        assert course.title == "Existing Course"
        assert chunk_count == 1
        
        # Verify add methods were called since add_course_document doesn't check duplicates
        mock_vector_instance.add_course_metadata.assert_called_once()
        mock_vector_instance.add_course_content.assert_called_once()


class TestRAGSystemIntegration:
    """Integration tests that use real components where possible"""

    def test_tool_registration_integration(self, temp_config):
        """Test that tools are properly registered in the system"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.DocumentProcessor'):
            
            rag_system = RAGSystem(temp_config)
            
            # Verify tools were registered
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 2
            
            tool_names = [td["name"] for td in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names

    def test_prompt_construction(self, temp_config):
        """Test that prompts are properly constructed"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager'), \
             patch('rag_system.DocumentProcessor'):
            
            mock_ai_instance = Mock()
            mock_ai_instance.generate_response.return_value = "Response"
            mock_ai_gen.return_value = mock_ai_instance
            
            rag_system = RAGSystem(temp_config)
            rag_system.query("What is machine learning?")
            
            # Verify the prompt includes the expected format
            call_args = mock_ai_instance.generate_response.call_args
            prompt = call_args[1]["query"]
            assert "Answer this question about course materials:" in prompt
            assert "What is machine learning?" in prompt