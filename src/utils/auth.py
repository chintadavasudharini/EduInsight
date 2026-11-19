"""
Authentication and User Management Module.
Implements local SQLite-backed user credentials with PBKDF2 salt-hashed passwords.
Provides registration, authentication, and session helpers.
"""

import sqlite3
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from src.utils.config import ARTIFACTS_DIR

DB_PATH = ARTIFACTS_DIR / "data" / "users.db"


def get_db_connection() -> sqlite3.Connection:
    """Initializes and returns SQLite database connection for users."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """Hashes password with cryptographic PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(16).hex()
    pwd_bytes = password.encode("utf-8")
    salt_bytes = salt.encode("utf-8")
    hash_bytes = hashlib.pbkdf2_hmac("sha256", pwd_bytes, salt_bytes, iterations=100000)
    return hash_bytes.hex(), salt


def is_valid_email(email: str) -> bool:
    """Validates basic email format."""
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return bool(re.match(pattern, email.strip()))


def register_user(name: str, email: str, password: str) -> Tuple[bool, str]:
    """Registers a new user into SQLite database."""
    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Name cannot be empty."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            return False, "An account with this email already exists."

        pwd_hash, salt = hash_password(password)
        cursor.execute(
            "INSERT INTO users (name, email, password_hash, salt) VALUES (?, ?, ?, ?)",
            (name, email, pwd_hash, salt),
        )
        conn.commit()
        return True, "Account registered successfully! Please log in."
    except Exception as e:
        return False, f"Registration error: {str(e)}"
    finally:
        conn.close()


def authenticate_user(email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Authenticates email and password against stored hash and salt."""
    email = email.strip().lower()
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, password_hash, salt FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        if not row:
            return False, None, "Invalid email or password."

        user_id, name, user_email, stored_hash, salt = row
        computed_hash, _ = hash_password(password, salt)

        if computed_hash == stored_hash:
            user_data = {
                "id": user_id,
                "name": name,
                "email": user_email,
            }
            return True, user_data, "Login successful!"
        else:
            return False, None, "Invalid email or password."
    except Exception as e:
        return False, None, f"Authentication error: {str(e)}"
    finally:
        conn.close()
