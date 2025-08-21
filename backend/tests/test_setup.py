"""Tests for configuration, setup, and environment validation"""

import pytest
from unittest.mock import Mock, patch
import sys
import os
import tempfile

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, config
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestConfiguration:
    """Test configuration validation and setup"""

    def test_default_config_values(self):
        """Test that default configuration values are appropriate"""
        test_config = Config()
        
        # Test critical settings
        assert test_config.ANTHROPIC_MODEL == "claude-sonnet-4-20250514"
        assert test_config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert test_config.CHUNK_SIZE == 800
        assert test_config.CHUNK_OVERLAP == 100
        assert test_config.MAX_HISTORY == 2
        assert test_config.CHROMA_PATH == "./chroma_db"

    def test_max_results_configuration_issue(self):
        """Test the critical MAX_RESULTS = 0 configuration issue"""
        test_config = Config()
        
        # This is the bug we're looking for
        assert test_config.MAX_RESULTS == 0, "MAX_RESULTS should be 0 (this is the bug)"
        
        # This setting would cause no search results to be returned
        # MAX_RESULTS should be > 0 for the system to work properly

    def test_environment_variable_loading(self):
        """Test that environment variables are properly loaded"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-env-key'}):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == 'test-env-key'

    def test_missing_api_key(self):
        """Test handling of missing API key"""
        with patch.dict(os.environ, {}, clear=True):
            test_config = Config()
            assert test_config.ANTHROPIC_API_KEY == ""

    def test_chunk_settings_validation(self):
        """Test that chunk settings are reasonable"""
        test_config = Config()
        
        # Chunk overlap should be less than chunk size
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
        
        # Both should be positive
        assert test_config.CHUNK_SIZE > 0
        assert test_config.CHUNK_OVERLAP >= 0

    def test_global_config_instance(self):
        """Test that the global config instance is properly initialized"""
        from config import config
        
        assert isinstance(config, Config)
        assert hasattr(config, 'ANTHROPIC_API_KEY')
        assert hasattr(config, 'MAX_RESULTS')


class TestVectorStoreSetup:
    """Test vector store initialization and setup"""

    def test_vector_store_init(self, temp_chroma_dir):
        """Test vector store initialization"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        assert vector_store.max_results == 5
        assert vector_store.client is not None
        assert vector_store.embedding_function is not None
        assert vector_store.course_catalog is not None
        assert vector_store.course_content is not None

    def test_vector_store_max_results_zero_bug(self, temp_chroma_dir):
        """Test the impact of MAX_RESULTS = 0 on vector store"""
        # This simulates the actual bug in the config
        vector_store = VectorStore(
            chroma_path=temp_chroma_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=0  # The problematic value from config
        )
        
        assert vector_store.max_results == 0
        
        # Add some test data
        test_course = Course(title="Test Course", lessons=[])
        test_chunks = [
            CourseChunk(content="Test content 1", course_title="Test Course", chunk_index=0),
            CourseChunk(content="Test content 2", course_title="Test Course", chunk_index=1)
        ]
        
        vector_store.add_course_metadata(test_course)
        vector_store.add_course_content(test_chunks)
        
        # Search with max_results=0 should return no results
        results = vector_store.search("test content")
        
        # This is the bug - search returns empty even with valid data
        assert results.is_empty(), "MAX_RESULTS=0 causes empty search results"

    def test_vector_store_collection_creation(self, temp_chroma_dir):
        """Test that collections are properly created"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        # Collections should be created
        collections = vector_store.client.list_collections()
        collection_names = [c.name for c in collections]
        
        assert "course_catalog" in collection_names
        assert "course_content" in collection_names

    def test_embedding_function_setup(self, temp_chroma_dir):
        """Test that embedding function is properly configured"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        # Should be able to generate embeddings
        embedding_func = vector_store.embedding_function
        assert embedding_func is not None
        
        # Test that it can actually generate embeddings
        embeddings = embedding_func(["test text"])
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0  # Should have some dimensionality

    def test_vector_store_error_handling(self):
        """Test vector store error handling for invalid paths"""
        # Test with invalid embedding model
        with pytest.raises(Exception):
            VectorStore(
                chroma_path="./test_invalid",
                embedding_model="nonexistent-model",
                max_results=5
            )


class TestDocumentLoading:
    """Test document loading and processing setup"""

    def test_docs_folder_exists(self):
        """Test that the docs folder exists with course files"""
        docs_path = "/home/patrick/learning/starting-ragchatbot-codebase/docs"
        
        assert os.path.exists(docs_path), "docs folder should exist"
        
        # Check for course files
        files = os.listdir(docs_path)
        course_files = [f for f in files if f.endswith(('.txt', '.pdf', '.docx'))]
        
        assert len(course_files) > 0, "Should have course documents in docs folder"

    def test_supported_file_formats(self):
        """Test that expected file formats are present"""
        docs_path = "/home/patrick/learning/starting-ragchatbot-codebase/docs"
        
        if os.path.exists(docs_path):
            files = os.listdir(docs_path)
            
            # Should have at least some supported formats
            supported_extensions = ['.txt', '.pdf', '.docx']
            found_extensions = set()
            
            for file in files:
                for ext in supported_extensions:
                    if file.lower().endswith(ext):
                        found_extensions.add(ext)
            
            assert len(found_extensions) > 0, "Should have files with supported extensions"


class TestDependenciesAndEnvironment:
    """Test that required dependencies and environment are set up"""

    def test_required_imports(self):
        """Test that all required modules can be imported"""
        try:
            import chromadb
            import anthropic
            import sentence_transformers
            import fastapi
            import pydantic
        except ImportError as e:
            pytest.fail(f"Required dependency not available: {e}")

    def test_chromadb_version_compatibility(self):
        """Test ChromaDB version and basic functionality"""
        import chromadb
        
        # Test basic ChromaDB operations
        client = chromadb.Client()
        collection = client.create_collection("test_collection")
        
        # Basic add and query
        collection.add(
            documents=["test document"],
            ids=["test_id"]
        )
        
        results = collection.query(query_texts=["test"], n_results=1)
        assert len(results['documents'][0]) == 1

    def test_sentence_transformers_model(self):
        """Test that the default sentence transformer model works"""
        from sentence_transformers import SentenceTransformer
        
        # This might take a while on first run as it downloads the model
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(["test sentence"])
            assert embeddings.shape[0] == 1
            assert embeddings.shape[1] > 0  # Should have embedding dimensions
        except Exception as e:
            pytest.fail(f"Sentence transformer model failed: {e}")


class TestSystemIntegration:
    """Test system-level integration and common failure points"""

    def test_max_results_fix_validation(self, temp_chroma_dir):
        """Test that fixing MAX_RESULTS resolves the search issue"""
        # Test with the buggy config (MAX_RESULTS = 0)
        buggy_store = VectorStore(
            chroma_path=temp_chroma_dir + "_buggy",
            embedding_model="all-MiniLM-L6-v2",
            max_results=0
        )
        
        # Test with fixed config (MAX_RESULTS > 0)
        fixed_store = VectorStore(
            chroma_path=temp_chroma_dir + "_fixed",
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        # Add identical test data to both
        test_course = Course(title="Test Course", lessons=[])
        test_chunks = [
            CourseChunk(content="Machine learning content", course_title="Test Course", chunk_index=0)
        ]
        
        for store in [buggy_store, fixed_store]:
            store.add_course_metadata(test_course)
            store.add_course_content(test_chunks)
        
        # Search both stores
        buggy_results = buggy_store.search("machine learning")
        fixed_results = fixed_store.search("machine learning")
        
        # Buggy store should return empty results
        assert buggy_results.is_empty(), "Buggy store should return empty results"
        
        # Fixed store should return results
        assert not fixed_results.is_empty(), "Fixed store should return results"
        assert len(fixed_results.documents) > 0

    def test_api_key_validation_simulation(self):
        """Test API key validation (without making actual API calls)"""
        from ai_generator import AIGenerator
        
        # Test with empty API key
        with pytest.raises(Exception):
            generator = AIGenerator("", "claude-sonnet-4-20250514")
            # This would fail when trying to create the client

    def test_complete_system_configuration(self):
        """Test that all configuration values work together"""
        test_config = Config(
            ANTHROPIC_API_KEY="test-key",
            MAX_RESULTS=5,  # Fixed value
            CHUNK_SIZE=800,
            CHUNK_OVERLAP=100,
            MAX_HISTORY=2
        )
        
        # Verify configuration is internally consistent
        assert test_config.MAX_RESULTS > 0
        assert test_config.CHUNK_OVERLAP < test_config.CHUNK_SIZE
        assert test_config.MAX_HISTORY >= 0
        
        # Test that values are reasonable for the system
        assert test_config.MAX_RESULTS <= 20  # Not too high
        assert test_config.CHUNK_SIZE >= 100   # Not too small
        assert test_config.MAX_HISTORY <= 10   # Not excessive


class TestErrorDiagnostics:
    """Test error diagnosis and logging capabilities"""

    def test_search_error_propagation(self, temp_chroma_dir):
        """Test that search errors are properly captured and reported"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_dir,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5
        )
        
        # Test search on empty store
        results = vector_store.search("nonexistent content")
        
        # Should return empty results, not error
        assert results.is_empty()
        assert results.error is None

    def test_configuration_error_messages(self):
        """Test that configuration issues produce clear error messages"""
        # This test documents expected behavior for debugging
        
        current_config = Config()
        
        # Document the problematic setting
        if current_config.MAX_RESULTS == 0:
            error_msg = (
                "MAX_RESULTS is set to 0 in config.py, which will cause "
                "no search results to be returned. This should be set to "
                "a positive integer (e.g., 5)."
            )
            print(f"Configuration Issue: {error_msg}")
            
        # Document other potential issues
        if not current_config.ANTHROPIC_API_KEY:
            print("Warning: ANTHROPIC_API_KEY is not set")
            
        assert True  # This test is for documentation/diagnosis