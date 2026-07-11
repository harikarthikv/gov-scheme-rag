# Government Scheme Finder — FastAPI entry point
# Starts the server, registers routers, enables CORS for React dev server
# Run: uvicorn main:app --reload --port 8000

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.chat import router as chat_router
from routes.auth import router as auth_router
from routes.sessions import router as sessions_router
from db import init_db
from dotenv import load_dotenv
import os

from contextlib import asynccontextmanager

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Database on startup
    init_db()
    yield
    # Any cleanup on shutdown can go here

app = FastAPI(title="Government Scheme Finder", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "ok"}
