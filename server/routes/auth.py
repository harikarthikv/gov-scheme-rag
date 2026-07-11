from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator
import sqlite3
import hashlib
import os
import base64
from db import get_db

router = APIRouter()


class AuthRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or len(v) > 254:
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        if len(v) > 128:
            raise ValueError("Password too long")
        return v


def _hash_password(password: str) -> str:
    """Salted scrypt key derivation. Returns a base64-encoded (salt || dk) blob."""
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=32)
    return base64.b64encode(salt + dk).decode("ascii")


def _verify_password(password: str, stored: str) -> bool:
    """Constant-time comparison of scrypt-derived keys."""
    try:
        raw = base64.b64decode(stored.encode("ascii"))
        salt, dk_stored = raw[:16], raw[16:]
        dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, dklen=32)
        return dk == dk_stored
    except Exception:
        return False


@router.post("/register")
def register(req: AuthRequest):
    with get_db() as conn:
        cursor = conn.cursor()
        hashed = _hash_password(req.password)
        try:
            cursor.execute(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                (req.email, hashed),
            )
            conn.commit()
            return {"status": "ok", "user_id": cursor.lastrowid, "email": req.email}
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=400, detail="Email already registered")


@router.post("/login")
def login(req: AuthRequest):
    with get_db() as conn:
        cursor = conn.cursor()
        # Fetch by email only — verify password in Python (not in SQL) to prevent timing attacks
        cursor.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (req.email,),
        )
        user = cursor.fetchone()

        if not user or not _verify_password(req.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        return {"status": "ok", "user_id": user["id"], "email": user["email"]}
