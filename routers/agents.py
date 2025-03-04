# routers/agents.py
from fastapi import APIRouter, HTTPException, Request, status
from models import Agent, AgentCreate, AgentUpdate, PyObjectId
from typing import List

router = APIRouter()

@router.post("/", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(request: Request, agent: AgentCreate):
    db = request.app.state.db
    result = await db.agents.insert_one(agent.dict())
    new_agent = await db.agents.find_one({"_id": result.inserted_id})
    return new_agent

@router.get("/", response_model=List[Agent])
async def get_agents(request: Request):
    db = request.app.state.db
    agents = await db.agents.find().to_list(100)
    return agents

@router.get("/{agent_id}", response_model=Agent)
async def get_agent(request: Request, agent_id: str):
    db = request.app.state.db
    agent = await db.agents.find_one({"_id": PyObjectId(agent_id)})
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@router.put("/{agent_id}", response_model=Agent)
async def update_agent(request: Request, agent_id: str, agent_update: AgentUpdate):
    db = request.app.state.db
    update_data = {k: v for k, v in agent_update.dict().items() if v is not None}
    result = await db.agents.update_one({"_id": PyObjectId(agent_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Agent not found or no changes made")
    agent = await db.agents.find_one({"_id": PyObjectId(agent_id)})
    return agent

@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(request: Request, agent_id: str):
    db = request.app.state.db
    result = await db.agents.delete_one({"_id": PyObjectId(agent_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Agent not found")
    return
