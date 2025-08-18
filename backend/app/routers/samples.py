from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/samples", tags=["samples"])

@router.post("/", response_model=schemas.SampleOut)
def create_sample(
    payload: Optional[schemas.SampleIn] = None,
    project_id: Optional[int] = Query(None),
    material_id: Optional[int] = Query(None),
    sample_code: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    if payload:
        prj_id, mat_id, sc = payload.project_id, payload.material_id, payload.sample_code
    else:
        if not (project_id and material_id):
            raise HTTPException(400, "Provide JSON body {project_id,material_id[,sample_code]} or query params")
        prj_id, mat_id, sc = project_id, material_id, sample_code

    if not db.query(models.Project).get(prj_id):
        raise HTTPException(400, "project_id not found")
    if not db.query(models.Material).get(mat_id):
        raise HTTPException(400, "material_id not found")

    obj = models.Sample(project_id=prj_id, material_id=mat_id, sample_code=sc)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.SampleOut])
def list_samples(
    project_id: Optional[int] = Query(None),
    material_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    q = db.query(models.Sample)
    if project_id: q = q.filter(models.Sample.project_id == project_id)
    if material_id: q = q.filter(models.Sample.material_id == material_id)
    return q.all()
