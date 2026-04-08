"""
Main entrypoint for FakeNews Detection OpenEnv.
Starts the FastAPI server.
"""
import os
import sys
import uvicorn

# Make sure all modules are importable from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting FakeNews Detection OpenEnv server on port {port}...", flush=True)
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )