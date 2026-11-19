"""
Streamlit Authentication Views: Login and Register.
Implements:
- Registration with Name, Email, Create Password, Confirm Password
- Login with Email, Password
- Hide/Unhide toggle for password fields
- Input validation and feedback
"""

import streamlit as st
from src.utils.auth import register_user, authenticate_user


def render_register_page():
    st.markdown('<div class="main-header">📝 Create an Account</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Register to access the Student Performance Analytics and Laboratory platform.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                <h3 style="margin-top:0; color:#1E3A8A; text-align:center;">User Registration</h3>
            """,
            unsafe_allow_html=True,
        )

        with st.form("register_form", clear_on_submit=False):
            name = st.text_input(
                "Full Name",
                placeholder="e.g. John Doe",
                help="Enter your official name.",
            )
            email = st.text_input(
                "Email Address",
                placeholder="e.g. student@academic.edu",
                help="Enter a valid institutional or personal email address.",
            )

            # Hide / Unhide Toggle
            show_pwd = st.checkbox("👁️ Show Password", value=False, key="reg_show_pwd")
            pwd_type = "default" if show_pwd else "password"

            create_password = st.text_input(
                "Create Password",
                type=pwd_type,
                placeholder="Enter at least 6 characters",
                help="Create a secure password.",
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type=pwd_type,
                placeholder="Re-enter password",
                help="Must match Create Password exactly.",
            )

            submit_reg = st.form_submit_button("🚀 Register Account", use_container_width=True)

        if submit_reg:
            if not name.strip():
                st.error("❌ Name field is required.")
            elif not email.strip():
                st.error("❌ Email field is required.")
            elif not create_password:
                st.error("❌ Create Password field is required.")
            elif create_password != confirm_password:
                st.error("❌ Passwords do not match. Please verify Confirm Password.")
            else:
                success, msg = register_user(name, email, create_password)
                if success:
                    st.success(f"✅ {msg}")
                    st.info("💡 You can now proceed to the **Login Page** to sign in.")
                else:
                    st.error(f"❌ {msg}")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 15px; font-size: 0.9rem; color: #64748B;">
                Already have an account? Select <b>Login Page</b> from the sidebar.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_login_page():
    st.markdown('<div class="main-header">🔑 Member Login</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sign in with your email and password to access the academic suite.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                <h3 style="margin-top:0; color:#1E3A8A; text-align:center;">Account Login</h3>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            email = st.text_input(
                "Email Address",
                placeholder="e.g. student@academic.edu",
            )

            show_pwd = st.checkbox("👁️ Show Password", value=False, key="login_show_pwd")
            pwd_type = "default" if show_pwd else "password"

            password = st.text_input(
                "Password",
                type=pwd_type,
                placeholder="Enter your password",
            )

            submit_login = st.form_submit_button("🔓 Sign In", use_container_width=True)

        if submit_login:
            if not email.strip():
                st.error("❌ Please enter your email.")
            elif not password:
                st.error("❌ Please enter your password.")
            else:
                success, user_data, msg = authenticate_user(email, password)
                if success:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user_data
                    st.success(f"✅ Welcome back, **{user_data['name']}**!")
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")

        st.markdown(
            """
            <div style="text-align: center; margin-top: 15px; font-size: 0.9rem; color: #64748B;">
                Don't have an account? Select <b>Register Page</b> from the sidebar.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
