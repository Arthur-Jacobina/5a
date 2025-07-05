# mem0ai config
mem0ai_config = {
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.1
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    "vector_store": {
        "provider": "pinecone",
        "config": {
            "collection_name": "testing",
            "embedding_model_dims": 1536,
            "serverless_config": {
                "cloud": "aws",
                "region": "us-east-1"
            },
            "metric": "cosine"
        }
    }
}