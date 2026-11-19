"""Unit tests for user authentication and credentials management."""

import pytest
import sqlite3
from src.utils.auth import (
    register_user,
    authenticate_user,
    hash_password,
    is_valid_email,
)


def test_hash_password():
    pwd = "secretpassword123"
    h1, salt1 = hash_password(pwd)
    assert len(h1) == 64
    assert len(salt1) == 32
    # Same password and salt should yield identical hash
    h2, _ = hash_password(pwd, salt1)
    assert h1 == h2


def test_is_valid_email():
    assert is_valid_email("student@university.edu") is True
    assert is_valid_email("invalid-email-format") is False
    assert is_valid_email("user@domain") is False


def test_user_registration_and_authentication():
    test_email = "academic_tester@aiml.edu"
    test_name = "Academic Tester"
    test_pwd = "SecurePassword@2026"

    # Register
    success, msg = register_user(test_name, test_email, test_pwd)
    assert success is True or "already exists" in msg

    # Duplicate registration should be rejected
    dup_success, dup_msg = register_user(test_name, test_email, test_pwd)
    assert dup_success is False
    assert "already exists" in dup_msg

    # Authenticate with correct password
    auth_ok, user_data, auth_msg = authenticate_user(test_email, test_pwd)
    assert auth_ok is True
    assert user_data is not None
    assert user_data["name"] == test_name
    assert user_data["email"] == test_email

    # Authenticate with incorrect password
    bad_ok, bad_user, bad_msg = authenticate_user(test_email, "WrongPassword999")
    assert bad_ok is False
    assert bad_user is None
    assert "Invalid email or password" in bad_msg
