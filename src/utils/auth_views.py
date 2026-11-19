"""
Authentication Views: Login and Register.
Implements automatic redirection flow:
- After registration -> redirects to Login
- After login -> redirects to Dashboard
Clean interface without syllabus/academic jargon.
"""

import streamlit as st
from src.utils.auth import register_user, authenticate_user


def render_register_page():
    st.markdown('<div class="main-header">Create an Account</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Join EduInsight to understand learning patterns and predict outcomes.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                <h3 style="margin-top:0; color:#1E3A8A; text-align:center;">Register</h3>
            """,
            unsafe_allow_html=True,
        )

        with st.form("register_form", clear_on_submit=False):
            name = st.text_input(
                "Full Name",
                placeholder="Enter your name",
            )
            email = st.text_input(
                "Email Address",
                placeholder="Enter your email address",
            )
            create_password = st.text_input(
                "Create Password",
                type="password",
                placeholder="Create a password (min 6 characters)",
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password",
                placeholder="Confirm your password",
            )

            submit_reg = st.form_submit_button("Register", use_container_width=True)

        if submit_reg:
            if not name.strip():
                st.error("Name field is required.")
            elif not email.strip():
                st.error("Email field is required.")
            elif not create_password:
                st.error("Password field is required.")
            elif create_password != confirm_password:
                st.error("Passwords do not match. Please verify your confirm password.")
            else:
                success, msg = register_user(name, email, create_password)
                if success:
                    # Automatic redirection to Login page
                    st.session_state["nav_selection"] = "Login"
                    st.session_state["reg_msg"] = "Registration successful! Please sign in with your credentials."
                    st.rerun()
                else:
                    st.error(msg)

        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("Already have an account? Sign In", use_container_width=True):
                st.session_state["nav_selection"] = "Login"
                st.rerun()
        with col_right:
            if st.button("Back to Home", use_container_width=True):
                st.session_state["nav_selection"] = "Home"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_login_page():
    st.markdown('<div class="main-header">Account Login</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sign in to EduInsight to access your analytics dashboard.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("reg_msg"):
        st.success(st.session_state.pop("reg_msg"))

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                <h3 style="margin-top:0; color:#1E3A8A; text-align:center;">Sign In</h3>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input(
                "Email Address",
                placeholder="Enter your registered email",
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
            )
            submit_login = st.form_submit_button("Sign In", use_container_width=True)

        if submit_login:
            if not email.strip():
                st.error("Please enter your email.")
            elif not password:
                st.error("Please enter your password.")
            else:
                success, user_data, msg = authenticate_user(email, password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    # Automatic redirection to Dashboard
                    st.session_state["nav_selection"] = "Dashboard"
                    st.rerun()
                else:
                    st.error(msg)

        col_left, col_right = st.columns(2)
        with col_left:
            if st.button("New user? Register here", use_container_width=True):
                st.session_state["nav_selection"] = "Register"
                st.rerun()
        with col_right:
            if st.button("Back to Home", use_container_width=True, key="login_back_home"):
                st.session_state["nav_selection"] = "Home"
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
