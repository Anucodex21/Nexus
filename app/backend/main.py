import uvicorn
from app.backend.api import app

def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app.backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
