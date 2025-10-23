"""Authentication system for the RAG application"""
from typing import Optional, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta
import secrets
import bcrypt


class User(BaseModel):
    """User model"""
    username: str
    password_hash: str


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class AuthSession(BaseModel):
    """Authentication session model"""
    session_token: str
    username: str
    created_at: datetime
    expires_at: datetime


class AuthManager:
    """Manages user authentication and sessions"""

    def __init__(self, session_duration_hours: int = 24):
        self.session_duration_hours = session_duration_hours
        # In-memory storage (in production, use a database)
        self.users: Dict[str, User] = {}
        self.auth_sessions: Dict[str, AuthSession] = {}

        # Create a default demo user for testing
        self._create_demo_users()

    def _create_demo_users(self):
        """Create demo users for testing"""
        # Demo user: username=demo, password=demo
        self.create_user("demo", "demo")
        # Admin user: username=admin, password=admin
        self.create_user("admin", "admin")

    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user"""
        if username in self.users:
            return False

        password_hash = self._hash_password(password)
        self.users[username] = User(username=username, password_hash=password_hash)
        return True

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Authenticate a user and create a session
        Returns session token if successful, None otherwise
        """
        if username not in self.users:
            return None

        user = self.users[username]

        # Use bcrypt's constant-time comparison to prevent timing attacks
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            return None

        # Create session token
        session_token = secrets.token_urlsafe(32)
        created_at = datetime.now()
        expires_at = created_at + timedelta(hours=self.session_duration_hours)

        auth_session = AuthSession(
            session_token=session_token,
            username=username,
            created_at=created_at,
            expires_at=expires_at
        )

        self.auth_sessions[session_token] = auth_session
        return session_token

    def validate_session(self, session_token: str) -> Optional[str]:
        """
        Validate a session token
        Returns username if valid, None otherwise
        """
        if session_token not in self.auth_sessions:
            return None

        auth_session = self.auth_sessions[session_token]

        # Check if session has expired
        if datetime.now() > auth_session.expires_at:
            del self.auth_sessions[session_token]
            return None

        return auth_session.username

    def logout(self, session_token: str) -> bool:
        """
        Logout a user by invalidating their session
        Returns True if successful, False otherwise
        """
        if session_token in self.auth_sessions:
            del self.auth_sessions[session_token]
            return True
        return False

    def cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        now = datetime.now()
        expired_tokens = [
            token for token, session in self.auth_sessions.items()
            if now > session.expires_at
        ]
        for token in expired_tokens:
            del self.auth_sessions[token]
