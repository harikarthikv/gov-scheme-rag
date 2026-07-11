from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import uuid
from typing import List
from db import get_db

router = APIRouter()

class SessionCreate(BaseModel):
    user_id: int
    title: str = "New Conversation"

@router.post("/sessions")
def create_session(req: SessionCreate):
    with get_db() as conn:
        cursor = conn.cursor()
        session_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO sessions (id, user_id, title) VALUES (?, ?, ?)", 
                       (session_id, req.user_id, req.title))
        conn.commit()
        return {"session_id": session_id, "title": req.title}

@router.get("/sessions")
def list_sessions(user_id: int = Query(...)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, created_at FROM sessions WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        sessions = [dict(row) for row in cursor.fetchall()]
        return {"sessions": sessions}

@router.get("/sessions/{session_id}/messages")
def get_session_messages(session_id: str, user_id: int = Query(...)):
    with get_db() as conn:
        cursor = conn.cursor()
        # Verify ownership
        cursor.execute("SELECT id FROM sessions WHERE id = ? AND user_id = ?", (session_id, user_id))
        if not cursor.fetchone():
            raise HTTPException(status_code=403, detail="Not authorized or session does not exist")
            
        cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        messages = [{"role": row["role"], "content": row["content"]} for row in cursor.fetchall()]
        return {"messages": messages}

@router.put("/sessions/{session_id}")
def update_session_title(session_id: str, title: str, user_id: int = Query(...)):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET title = ? WHERE id = ? AND user_id = ?", (title, session_id, user_id))
        conn.commit()
        return {"status": "ok"}
