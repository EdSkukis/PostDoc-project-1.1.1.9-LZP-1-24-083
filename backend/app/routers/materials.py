from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/materials", tags=["materials"])

@router.post("/", response_model=schemas.MaterialOut)
def create_material(
    payload: Optional[schemas.MaterialIn] = None,
    material_code: Optional[str] = Query(None),
    type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    if payload:
        mc, tp = payload.material_code, payload.type
    else:
        if not type:  # material_code может быть None
            raise HTTPException(400, "Provide JSON body {type[,material_code]} or query params ?type=&material_code=")
        mc, tp = material_code, type

    obj = models.Material(material_code=mc, type=tp)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.MaterialOut])
def list_materials(
    type: Optional[str] = Query(None),
    code: Optional[str] = Query(None, alias="material_code"),
    db: Session = Depends(get_db)
):
    q = db.query(models.Material)
    if type: q = q.filter(models.Material.type == type)
    if code: q = q.filter(models.Material.material_code == code)
    return q.all()
