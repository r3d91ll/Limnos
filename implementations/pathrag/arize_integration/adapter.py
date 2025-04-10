#!/usr/bin/env python3
"""
PathRAG Arize Phoenix Adapter using OpenTelemetry

This module provides an adapter for PathRAG to connect to Arize Phoenix
using OpenTelemetry for telemetry collection, which is more reliable than
the direct Phoenix SDK approach.
"""

import os
import time
import uuid
import json
import logging
import yaml
import requests
from typing import Dict, List, Any, Optional, Union, cast, TYPE_CHECKING
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pathrag.phoenix")

class PathRAGArizeAdapter:
    """
    Adapter for connecting PathRAG to Arize Phoenix using OpenTelemetry.
    This adapter provides methods for logging telemetry data about
    PathRAG's performance, including query/response pairs, path information,
    and evaluation metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the adapter with configuration.
        
        Args:
            config: Configuration dictionary or path to YAML configuration file
        """
        # Handle both dictionary and file path inputs
        if isinstance(config, dict):
            self.config = config
            self.config_file = None
        else:
            self.config_file = config
            self.config = self._load_config(config)
        
        # LLM provider settings
        self.llm_provider = self.config.get("llm_provider", "ollama")
        self.model_name = self.config.get("model_name", "unknown-model")
        
        # Ollama settings
        self.ollama_host = self.config.get("ollama_host", "localhost")
        self.ollama_port = self.config.get("ollama_port", 11434)
        self.ollama_model = self.config.get("ollama_model", "llama3")
        self.ollama_url = f"http://{self.ollama_host}:{self.ollama_port}"
        
        # Performance tracking
        self.track_performance = self.config.get("track_performance", True)
        
        # Phoenix configuration
        self.phoenix_host = self.config.get("phoenix_host", "localhost")
        self.phoenix_port = self.config.get("phoenix_port", 8084)
        self.project_name = self.config.get("project_name", "pathrag-inference")
        
        # Set up OpenTelemetry for Phoenix (if tracking is enabled)
        self.phoenix_available = False
        
    def initialize(self):
        """
        Initialize the adapter and set up OpenTelemetry for Phoenix.
        This method is called by the PathRAG runner after initialization.
        """
        if self.track_performance:
            try:
                self._setup_telemetry()
                logger.info(f"✅ Connected to Phoenix at http://{self.phoenix_host}:{self.phoenix_port}")
                self.phoenix_available = True
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to connect to Phoenix: {e}")
                logger.warning("Telemetry will not be recorded")
                self.phoenix_available = False
                return False
        return False
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.
        
        Args:
            config_file: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return cast(Dict[str, Any], config)
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")
            return cast(Dict[str, Any], {})
    
    def _generate_ollama_response(self, prompt: str) -> str:
        """
        Generate a response using Ollama API.
        
        Args:
            prompt: The prompt to send to the Ollama API
            
        Returns:
            The generated response text
        """
        try:
            # Log the request
            logger.info(f"🤖 Sending request to Ollama API using model: {self.ollama_model}")
            
            # Prepare the request payload
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False
            }
            
            # Send the request to Ollama API
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 120 seconds timeout
            )
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                logger.info(f"✅ Successfully generated response from Ollama")
                return cast(str, generated_text)
            else:
                logger.warning(f"⚠️ Ollama API returned status code: {response.status_code}")
                logger.warning(f"⚠️ Response: {response.text}")
                return cast(str, f"Error: Failed to generate response from Ollama. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error calling Ollama API: {e}")
            return cast(str, f"Error: {str(e)}")
    
    def _setup_telemetry(self):
        """
        Set up OpenTelemetry for Phoenix.
        """
        try:
            # Import required packages
            import requests
            
            # Import opentelemetry packages (they don't have type stubs)
            from opentelemetry import trace  # type: ignore
            from opentelemetry.sdk.resources import Resource  # type: ignore
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
            
            # First check if Phoenix is actually running
            try:
                response = requests.get(f"http://{self.phoenix_host}:{self.phoenix_port}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ Phoenix health check successful at http://{self.phoenix_host}:{self.phoenix_port}")
                else:
                    logger.warning(f"⚠️ Phoenix health check failed: {response.status_code}")
                    return
            except Exception as e:
                logger.warning(f"⚠️ Phoenix health check failed: {e}")
                return
            
            # Set up a resource (this identifies your service)
            resource = Resource.create({
                "service.name": "pathrag",
                "service.version": "1.0.0",
                "model.name": self.model_name,
                "phoenix.project": self.project_name
            })
            
            logger.info(f"🔍 Using Phoenix project: {self.project_name}")
            
            # Create a tracer provider with the resource
            tracer_provider = TracerProvider(resource=resource)
            
            # Set up the OTLP exporter to send traces to Phoenix
            otlp_endpoint = f"http://{self.phoenix_host}:{self.phoenix_port}/v1/traces"
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            
            # Add the exporter to the tracer provider
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Set the tracer provider as the global default
            trace.set_tracer_provider(tracer_provider)
            
            # Create a tracer for our service
            self.tracer = trace.get_tracer("pathrag")
            
            # Try to instrument LangChain if it's being used
            try:
                from openinference.instrumentation.langchain import LangChainInstrumentor  # type: ignore
                LangChainInstrumentor().instrument()
                logger.info("✅ Successfully instrumented LangChain for Phoenix telemetry")
            except ImportError:
                logger.debug("LangChain instrumentation not available (this is okay if not using LangChain)")
            except Exception as e:
                logger.debug(f"Failed to instrument LangChain: {e}")
            
            logger.info(f"✅ Phoenix telemetry set up successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ Failed to import OpenTelemetry: {e}")
            logger.warning("Run: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
            raise e
    
    def query(self, query: str, session_id: Optional[str] = None, user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process a query using the PathRAG approach, retrieving relevant documents
        and generating a response. Also logs telemetry data to Phoenix.
        
        Args:
            query: The user query to process
            session_id: Session identifier for tracking
            user_id: User identifier for tracking
            **kwargs: Additional arguments for the query process
            
        Returns:
            Dictionary containing the answer, paths, and other metadata
        """
        # In a full implementation, this would:
        # 1. Retrieve relevant documents based on the query
        # 2. Construct paths through the document graph
        # 3. Generate a response using an LLM
        # 4. Log telemetry data to Phoenix
        
        start_time = time.time()
        
        # TODO: Implement actual document retrieval
        # For now, we'll use placeholder data for paths and documents
        # In a real implementation, this would come from a vector database
        paths = [["Document 1", "Document 2"], ["Document 3"]]
        documents = ["This is document 1 about PathRAG.", "This is document 2 about retrieval augmented generation."]
        
        # Generate a response using the configured LLM
        prompt = f"""Based on the following documents, please answer the query: {query}

Documents:
{documents[0]}
{documents[1]}

Answer:"""
        
        # Generate response based on the LLM provider
        if self.llm_provider == "ollama":
            response = self._generate_ollama_response(prompt)
        elif self.llm_provider == "openai":
            # TODO: Implement OpenAI integration
            logger.warning("OpenAI integration not implemented yet, using Ollama as fallback")
            response = self._generate_ollama_response(prompt)
        else:
            logger.error(f"Unsupported LLM provider: {self.llm_provider}")
            response = f"Error: Unsupported LLM provider '{self.llm_provider}'"
        
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # For now, we'll use estimated token usage
        # In a real implementation, this would come from the LLM response
        token_usage = {
            "prompt_tokens": len(prompt) // 4,  # Rough estimate
            "completion_tokens": len(response) // 4,  # Rough estimate
            "total_tokens": (len(prompt) + len(response)) // 4  # Rough estimate
        }
        
        # Create path information for telemetry
        path_info = [
            {"text": documents[0], "score": 0.95, "metadata": {"source": "document1.pdf"}},
            {"text": documents[1], "score": 0.85, "metadata": {"source": "document2.pdf"}}
        ]
        
        # Log telemetry to Phoenix
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        }
        
        trace_id = self.log_telemetry(
            trace_id=str(uuid.uuid4()),
            query=query,
            response=response,
            path=path_info,
            latency_ms=latency_ms,
            token_usage=token_usage,
            metadata=metadata
        )
        
        # Return the result with expected structure
        result = {
            "answer": response,
            "paths": paths,
            "documents": documents,
            "metrics": {
                "latency_ms": latency_ms,
                "token_usage": token_usage,
                "trace_id": trace_id
            }
        }
        
        return result
        
    def log_telemetry(self, 
                trace_id: str,
                query: str, 
                response: str, 
                path: Optional[List[Dict[str, Any]]] = None,
                latency_ms: Optional[float] = None,
                token_usage: Optional[Dict[str, int]] = None,
                evaluation: Optional[Dict[str, Any]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log telemetry data to Phoenix.
        
        Args:
            trace_id: Unique identifier for this trace
            query: User query
            response: System response
            path: Path information (documents retrieved)
            latency_ms: Query processing latency in milliseconds
            token_usage: Token usage information (prompt, completion, total)
            evaluation: Evaluation metrics
            metadata: Additional metadata
            
        Returns:
            Trace ID if logged successfully, None otherwise
        """
        if not self.track_performance or not self.phoenix_available:
            return None
        
        try:
            # Generate a unique trace ID
            trace_id = str(uuid.uuid4())
            
            # Create and start the span
            with self.tracer.start_as_current_span(f"pathrag-query-{trace_id}") as span:
                # Set basic attributes
                span.set_attribute("app.trace_id", trace_id)
                span.set_attribute("app.query", query)
                span.set_attribute("app.response", response)
                span.set_attribute("app.model", self.model_name)
                
                # Set latency if provided
                if latency_ms is not None:
                    span.set_attribute("app.latency_ms", latency_ms)
                
                # Set token usage if provided
                if token_usage:
                    if "prompt_tokens" in token_usage:
                        span.set_attribute("app.token_usage.prompt", token_usage["prompt_tokens"])
                    if "completion_tokens" in token_usage:
                        span.set_attribute("app.token_usage.completion", token_usage["completion_tokens"])
                    if "total_tokens" in token_usage:
                        span.set_attribute("app.token_usage.total", token_usage["total_tokens"])
                
                # Set path information if provided
                if path:
                    span.set_attribute("app.path.node_count", len(path))
                    
                    # Add each path node as an attribute
                    for i, node in enumerate(path):
                        prefix = f"app.path.node.{i}"
                        
                        # Add node text
                        if "text" in node:
                            # Truncate very long texts to avoid span size limits
                            text = node["text"]
                            if len(text) > 1000:
                                text = text[:997] + "..."
                            span.set_attribute(f"{prefix}.text", text)
                        
                        # Add score
                        if "score" in node:
                            span.set_attribute(f"{prefix}.score", node["score"])
                        
                        # Add metadata
                        if "metadata" in node:
                            node_metadata = node["metadata"]
                            # Convert metadata to string to avoid complex types
                            span.set_attribute(f"{prefix}.metadata", str(node_metadata))
                
                # Set evaluation metrics if provided
                if evaluation:
                    for key, value in evaluation.items():
                        if isinstance(value, (int, float, str, bool)):
                            span.set_attribute(f"app.evaluation.{key}", value)
                        else:
                            # Convert complex types to string
                            span.set_attribute(f"app.evaluation.{key}", str(value))
                
                # Set additional metadata if provided
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str, bool)):
                            span.set_attribute(f"app.metadata.{key}", value)
                        else:
                            # Convert complex types to string
                            span.set_attribute(f"app.metadata.{key}", str(value))
                
                # Add an event to mark when the query was processed
                span.add_event(
                    name="query_processed",
                    attributes={"timestamp": datetime.now().isoformat()}
                )
            
            logger.info(f"✅ Logged trace {trace_id} to Phoenix project {self.project_name}")
            logger.info(f"✅ Phoenix endpoint: http://{self.phoenix_host}:{self.phoenix_port}/v1/traces")
            return trace_id
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to log trace to Phoenix: {e}")
            return None


# For testing
def test_adapter():
    """Test the adapter with sample data."""
    import tempfile
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        config = {
            "model_name": "test-model",
            "track_performance": True,
            "phoenix_host": "localhost",
            "phoenix_port": 8084
        }
        yaml.dump(config, f)
        config_file = f.name
    
    try:
        # Initialize the adapter
        adapter = PathRAGArizeAdapter(config_file)
        
        if not adapter.phoenix_available:
            logger.error("❌ Phoenix not available - test failed")
            return False
        
        # Log a test query
        trace_id = adapter.log_query(
            query="What is PathRAG?",
            response="PathRAG is a retrieval augmented generation system that uses a path-based approach to retrieval.",
            path=[
                {
                    "text": "PathRAG is an innovative retrieval augmented generation system.",
                    "score": 0.95,
                    "metadata": {"source": "document1.pdf", "page": 1}
                },
                {
                    "text": "PathRAG uses a path-based approach to improve retrieval quality.",
                    "score": 0.85,
                    "metadata": {"source": "document2.pdf", "page": 5}
                }
            ],
            latency_ms=234.56,
            token_usage={
                "prompt_tokens": 50,
                "completion_tokens": 30,
                "total_tokens": 80
            },
            evaluation={
                "relevance": 0.92,
                "accuracy": 0.87,
                "coverage": 0.95
            },
            metadata={
                "user_id": "test-user",
                "session_id": "test-session",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        if trace_id:
            logger.info(f"✅ Test succeeded - trace {trace_id} logged to Phoenix")
            logger.info(f"📊 View trace at http://localhost:8084")
            return True
        else:
            logger.error("❌ Failed to log trace - test failed")
            return False
    
    finally:
        # Clean up the temporary config file
        try:
            os.remove(config_file)
        except:
            pass

if __name__ == "__main__":
    logger.info("Testing PathRAGArizeAdapter...")
    test_adapter()
