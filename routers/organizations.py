# routers/organizations.py
from fastapi import APIRouter, HTTPException, Request, status
from models import Organization, OrganizationCreate, OrganizationUpdate, PyObjectId
from typing import List

router = APIRouter()

@router.post("/", response_model=Organization, status_code=status.HTTP_201_CREATED)
async def create_organization(request: Request, org: OrganizationCreate):
    db = request.app.state.db
    existing = await db.organizations.find_one({"name": org.name})
    if existing:
        raise HTTPException(status_code=400, detail="Organization already exists")
    result = await db.organizations.insert_one(org.dict())
    new_org = await db.organizations.find_one({"_id": result.inserted_id})
    return new_org

@router.get("/", response_model=List[Organization])
async def get_organizations(request: Request):
    db = request.app.state.db
    organizations = await db.organizations.find().to_list(100)
    return organizations

@router.get("/{org_id}", response_model=Organization)
async def get_organization(request: Request, org_id: str):
    db = request.app.state.db
    org = await db.organizations.find_one({"_id": PyObjectId(org_id)})
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    return org

@router.put("/{org_id}", response_model=Organization)
async def update_organization(request: Request, org_id: str, org_update: OrganizationUpdate):
    db = request.app.state.db
    update_data = {k: v for k, v in org_update.dict().items() if v is not None}
    result = await db.organizations.update_one({"_id": PyObjectId(org_id)}, {"$set": update_data})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Organization not found or no changes made")
    org = await db.organizations.find_one({"_id": PyObjectId(org_id)})
    return org

@router.delete("/{org_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_organization(request: Request, org_id: str):
    db = request.app.state.db
    result = await db.organizations.delete_one({"_id": PyObjectId(org_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Organization not found")
    return
