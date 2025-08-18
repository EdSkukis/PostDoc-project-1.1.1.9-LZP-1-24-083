from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from ..db import get_db
from .. import models, schemas

router = APIRouter(prefix="/experiments", tags=["experiments"])

@router.post("/", response_model=schemas.ExperimentOut)
def create_experiment(
    payload: Optional[schemas.ExperimentIn] = None,
    sample_id: Optional[int] = Query(None),
    test_type_id: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    if payload:
        sid, ttid = payload.sample_id, payload.test_type_id
    else:
        if not (sample_id and test_type_id):
            raise HTTPException(400, "Provide JSON body {sample_id,test_type_id} or query params")
        sid, ttid = sample_id, test_type_id

    if not db.query(models.Sample).get(sid):
        raise HTTPException(400, "sample_id not found")
    if not db.query(models.TestType).get(ttid):
        raise HTTPException(400, "test_type_id not found")

    obj = models.Experiment(sample_id=sid, test_type_id=ttid)
    db.add(obj); db.commit(); db.refresh(obj)
    return obj
