from fastapi import APIRouter, HTTPException, Depends
from ..models import AskRequest, AskResponse
from ..retriever import HybridRetriever

router = APIRouter()

# Global retriever instance (will be set in main.py)
retriever: HybridRetriever = None

def get_retriever():
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever not initialized")
    return retriever

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, ret: HybridRetriever = Depends(get_retriever)):
    """
    Ask a question about baby foods and get recommendations with citations
    """
    try:
        # Retrieve relevant foods
        retrieved_foods, scores = ret.retrieve(request.question, request.top_k)
        
        if not retrieved_foods:
            raise HTTPException(status_code=404, detail="No relevant foods found")
        
        # Generate answer and confidence
        answer, confidence = ret.generate_answer(request.question, retrieved_foods, scores)
        
        # Get citations
        citations = ret.get_citations(retrieved_foods, scores)
        
        return AskResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieved_foods=retrieved_foods
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
