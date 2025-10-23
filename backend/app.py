import warnings
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

from fastapi import FastAPI, HTTPException, Cookie, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import os

from config import config
from rag_system import RAGSystem
from auth import AuthManager, LoginRequest

# Initialize FastAPI app
app = FastAPI(title="Course Materials RAG System", root_path="")

# Add trusted host middleware for proxy
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)

# Initialize Auth system
auth_manager = AuthManager()

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[Union[str, Dict[str, Any]]]  # Support both strings and dicts with text/url
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]

class LoginResponse(BaseModel):
    """Response model for login"""
    success: bool
    username: Optional[str] = None
    message: Optional[str] = None

class AuthStatusResponse(BaseModel):
    """Response model for auth status check"""
    authenticated: bool
    username: Optional[str] = None

# API Endpoints

# Authentication endpoints
@app.post("/api/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    """Login endpoint"""
    session_token = auth_manager.authenticate(request.username, request.password)

    if session_token:
        # Set session token as HTTP-only cookie with security flags
        is_production = config.ENVIRONMENT == "production"
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=is_production,  # Require HTTPS in production
            max_age=86400,  # 24 hours
            samesite="lax"
        )
        return LoginResponse(
            success=True,
            username=request.username,
            message="Login successful"
        )
    else:
        return LoginResponse(
            success=False,
            message="Invalid username or password"
        )

@app.post("/api/logout")
async def logout(response: Response, session_token: Optional[str] = Cookie(None)):
    """Logout endpoint"""
    if session_token:
        auth_manager.logout(session_token)

    # Clear the session cookie with matching parameters
    response.delete_cookie(
        key="session_token",
        path="/",
        samesite="lax"
    )
    return {"success": True, "message": "Logged out successfully"}

@app.get("/api/auth/status", response_model=AuthStatusResponse)
async def auth_status(session_token: Optional[str] = Cookie(None)):
    """Check authentication status"""
    if not session_token:
        return AuthStatusResponse(authenticated=False)

    username = auth_manager.validate_session(session_token)

    if username:
        return AuthStatusResponse(authenticated=True, username=username)
    else:
        return AuthStatusResponse(authenticated=False)

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    #try:
    # Create session if not provided
    session_id = request.session_id
    if not session_id:
        session_id = rag_system.session_manager.create_session()
    
    # Process query using RAG system
    answer, sources = rag_system.query(request.query, session_id)
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        session_id=session_id
    )
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(docs_path, clear_existing=False)
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from pathlib import Path


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response
    
    
# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")