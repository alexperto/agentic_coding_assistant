import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Azure OpenAI API settings
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")  # Fallback for backward compatibility
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

    # OAuth Token settings (for dynamic API key)
    VERSA_CLIENT_ID: str = os.getenv("VERSA_CLIENT_ID", "")
    VERSA_CLIENT_SECRET: str = os.getenv("VERSA_CLIENT_SECRET", "")
    OKTA_TOKEN_URL: str = os.getenv(
        "OKTA_TOKEN_URL",
        "https://uc-sf.okta.com/oauth2/ausnwf6tyaq6v47QF5d7/v1/token"
    )

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Security settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")  # "production" or "development"

config = Config()


