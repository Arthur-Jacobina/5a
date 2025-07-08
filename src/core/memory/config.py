import os
from dotenv import load_dotenv

load_dotenv()

# default mem0ai config
default_mem0ai_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("OPENAI_SMALL_MODEL", "gpt-4o-mini"),
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
    },
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": os.getenv("PINECONE_COLLECTION", "testing"),
            "embedding_model_dims": os.getenv("PINECONE_MODEL_DIM", 1536),
            "serverless_config": {
                "cloud": os.getenv("PINECONE_CLOUD", "aws"),
                "region": os.getenv("PINECONE_REGION", "us-east-1")
            },
            "metric": os.getenv("PINECONE_METRIC", "cosine")
        }
    }
}

# graph mem0ai config
graph_mem0ai_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": os.getenv("OPENAI_SMALL_MODEL", "gpt-4o-mini"),
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        }
    },
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": os.getenv("PINECONE_COLLECTION", "testing"),
            "embedding_model_dims": os.getenv("PINECONE_MODEL_DIM", 1536),
            "serverless_config": {
                "cloud": os.getenv("PINECONE_CLOUD", "aws"),
                "region": os.getenv("PINECONE_REGION", "us-east-1")
            },
            "metric": os.getenv("PINECONE_METRIC", "cosine")
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            "username": os.getenv("NEO4J_USERNAME", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", "password"),
            "database": os.getenv("NEO4J_DATABASE", "neo4j")
        }
    }
}