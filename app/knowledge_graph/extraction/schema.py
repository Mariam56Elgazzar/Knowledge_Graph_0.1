from pydantic import BaseModel, Field
from typing import List, Optional

class Entity(BaseModel):
    id: str = Field(..., description="Canonical entity name")
    type: str = Field(..., description="Entity type label")

class Relation(BaseModel):
    source: str
    target: str
    type: str
    evidence: Optional[str] = None  

class EntityResult(BaseModel):
    nodes: List[Entity]

class RelationResult(BaseModel):
    relationships: List[Relation]
