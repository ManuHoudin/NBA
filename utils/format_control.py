from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000
    )

class RAGResponse(BaseModel):
    answer: str = Field(
        min_length=1,
        max_length=5000
    )

class SearchResult(BaseModel):
    score: float
    raw_score: float
    text: str
    metadata: dict
