import os
import httpx

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")

async def check_backend_health() -> dict:

    url = f"{BACKEND_BASE_URL}/health"
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
        