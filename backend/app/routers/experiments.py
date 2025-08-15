from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("/", response_model=schemas.ExperimentOut)
def create_experiment(item: schemas.ExperimentIn, db: Session = Depends(get_db)):
    if not db.query(models.Sample).get(item.sample_id):
        raise HTTPException(400, "sample_id not found")
    if not db.query(models.TestType).get(item.test_type_id):
        raise HTTPException(400, "test_type_id not found")

    obj = models.Experiment(sample_id=item.sample_id, test_type_id=item.test_type_id)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj

@router.get("/", response_model=list[schemas.ExperimentOut])
def list_experiments(db: Session = Depends(get_db)):
    return db.query(models.Experiment).all()
