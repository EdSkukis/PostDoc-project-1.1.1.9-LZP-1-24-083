from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/projects", tags=["projects"])

@router.post("/", response_model=schemas.ProjectOut)
def create_project(
    payload: Optional[schemas.ProjectIn] = None,
    code: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    # Приоритет: JSON body > query params
    if payload:
        code_in, name_in = payload.code, payload.name
    else:
        if not code or not name:
            raise HTTPException(400, "Provide JSON body {code,name} or query params ?code=&name=")
        code_in, name_in = code, name

    if db.query(models.Project).filter_by(code=code_in).first():
        raise HTTPException(400, "Project code already exists")

    obj = models.Project(code=code_in, name=name_in)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.ProjectOut])
def list_projects(
    code: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(models.Project)
    if code: q = q.filter(models.Project.code == code)
    if name: q = q.filter(models.Project.name.ilike(f"%{name}%"))
    return q.all()
