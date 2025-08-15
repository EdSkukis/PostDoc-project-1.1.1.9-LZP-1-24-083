from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/materials", tags=["materials"])

@router.post("/", response_model=schemas.MaterialOut)
def create_material(item: schemas.MaterialIn, db: Session = Depends(get_db)):
    obj = models.Material(material_code=item.material_code, type=item.type)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.MaterialOut])
def list_materials(db: Session = Depends(get_db)):
    return db.query(models.Material).all()
