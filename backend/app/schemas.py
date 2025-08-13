from pydantic import BaseModel
from typing import Optional

class ProjectIn(BaseModel):
    code: str
    name: str

class ProjectOut(ProjectIn):
    project_id: int

class SampleIn(BaseModel):
    project_id: int
    material_ref_id: int
    material_ref_type: str  # 'material' | 'material_component'
    sample_code: Optional[str] = None

class SampleOut(SampleIn):
    sample_id: int
