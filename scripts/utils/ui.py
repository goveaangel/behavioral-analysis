import streamlit as st

def set_background():
    bg_url = "https://www.globant.com/themes/custom/globant_bootstrap/logo.svg"  
    # O cualquier imagen que quieras usar

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Opcional: darle un fondo semitransparente al contenido */
        .block-container {{
            background: rgba(0, 0, 0, 0.55);
            padding: 2rem;
            border-radius: 1rem;
            backdrop-filter: blur(4px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )