# app.py
from ui.gradio_app import run_gradio
from app_core.llm_call import Chat

run_gradio(Chat)