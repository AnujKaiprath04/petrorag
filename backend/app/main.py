"""
PetroRAG Backend FastAPI Application Entrypoint
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.core.logging import logger

app = FastAPI(
    title=settings.APP_NAME,
    description="PetroRAG Intelligent Retrieval-Augmented Decision Support System for Oil & Gas Fields",
    version="1.0.0",
    debug=settings.DEBUG
)

# CORS Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.app.api.v1.endpoints.rag import router as rag_router
from backend.app.api.v1.endpoints.operational import router as operational_router
app.include_router(rag_router, prefix="/api/v1")
app.include_router(operational_router, prefix="/api/v1")


@app.get("/health", tags=["System"])
async def health_check():
    """System health and operational readiness endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.APP_NAME} on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
