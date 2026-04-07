"""
main.py — 
Application entry point. Starts the FastAPI server via uvicorn.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,       # HuggingFace Spaces default port
        reload=False,    # No reload in production
        log_level="info",
    )