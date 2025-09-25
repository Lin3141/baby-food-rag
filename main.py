from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.data_loader import DataLoader
from app.routers import ask

# Try to import the advanced retriever, fall back to simple one
try:
    from app.retriever import HybridRetriever
    print("Using advanced retriever with sentence transformers")
except ImportError as e:
    print(f"Sentence transformers not available ({e}), using simple retriever")
    from app.simple_retriever import SimpleHybridRetriever as HybridRetriever

# Initialize FastAPI app
app = FastAPI(
    title="Baby Food RAG API",
    description="A retrieval-augmented generation API for baby food recommendations",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize data and retriever
def initialize_retriever():
    try:
        # Load data
        data_loader = DataLoader("data/foods.csv")
        foods = data_loader.load_data()
        descriptions = data_loader.get_food_descriptions()
        
        # Initialize retriever
        retriever = HybridRetriever(foods, descriptions)
        
        # Set global retriever in ask router
        ask.retriever = retriever
        
        print(f"Successfully loaded {len(foods)} baby foods")
        return retriever
        
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        raise

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_retriever()

# Include routers
app.include_router(ask.router, prefix="/api", tags=["ask"])

# Serve the web interface
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Baby Food RAG API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
