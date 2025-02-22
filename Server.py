import logging
import sys
from typing import Generator
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from Crag import crag  

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('../server.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this based on your needs (e.g., specific origins)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home/test route
@app.get("/")
async def home_or_test_route():
    logger.info("Server is alive")
    return {"message": "Hello World"}

# Query route with streaming response
@app.get("/query")
async def query_index(text: str = Query(None, description="The query text to process")):
    if not text or text.strip() == "":
        logger.warning("No text provided in query")
        raise HTTPException(
            status_code=400,
            detail="No text found, please include a 'text' query parameter in the URL"
        )

    def generate() -> Generator[str, None, None]:
        try:
            ans = crag.run(query_str=text)
            for token in ans.response_gen:
                yield f"{token}"
            yield "[END]\n\n"
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            yield f"Error: {str(e)}\n\n"

    try:
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"}
        )
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"There is some server error: {e}"
        )

 