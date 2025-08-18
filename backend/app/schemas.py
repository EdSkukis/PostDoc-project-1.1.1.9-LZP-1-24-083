from pydantic import BaseModel
from typing import Optional

class ProjectIn(BaseModel):
    code: str
    name: str
class ProjectOut(ProjectIn):
    project_id: int

class MaterialIn(BaseModel):
    material_code: Optional[str] = None
    type: str
class MaterialOut(MaterialIn):
    material_id: int

class SampleIn(BaseModel):
    project_id: int
    material_id: int
    sample_code: Optional[str] = None
class SampleOut(SampleIn):
    sample_id: int

class ExperimentIn(BaseModel):
    sample_id: int
    test_type_id: int
class ExperimentOut(ExperimentIn):
    experiment_id: int
