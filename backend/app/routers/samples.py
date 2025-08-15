from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/samples", tags=["samples"])

@router.post("/", response_model=schemas.SampleOut)
def create_sample(item: schemas.SampleIn, db: Session = Depends(get_db)):
    # простая валидация ссылок
    if not db.query(models.Project).get(item.project_id):
        raise HTTPException(400, "project_id not found")
    if item.material_ref_type == "material":
        if not db.query(models.Material).get(item.material_ref_id):
            raise HTTPException(400, "material_ref_id not found in materials")
    elif item.material_ref_type == "material_component":
        if not db.query(models.MaterialComponent).get(item.material_ref_id):
            raise HTTPException(400, "material_ref_id not found in material_components")

    obj = models.Sample(
        project_id=item.project_id,
        material_ref_id=item.material_ref_id,
        material_ref_type=item.material_ref_type,
        sample_code=item.sample_code,
    )
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.SampleOut])
def list_samples(db: Session = Depends(get_db)):
    return db.query(models.Sample).all()
