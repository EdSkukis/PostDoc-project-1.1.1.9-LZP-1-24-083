from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/", response_model=schemas.ProjectOut)
def create_project(item: schemas.ProjectIn, db: Session = Depends(get_db)):
    if db.query(models.Project).filter_by(code=item.code).first():
        raise HTTPException(400, "Project code already exists")
    obj = models.Project(code=item.code, name=item.name)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return db.query(models.Project).all()
