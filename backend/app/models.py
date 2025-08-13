from sqlalchemy import Column, BigInteger, Integer, Text, Date, TIMESTAMP, ForeignKey, JSON, CheckConstraint
from .db import Base

class Project(Base):
    __tablename__ = "projects"
    __table_args__ = {"schema": "lab"}
    project_id = Column(BigInteger, primary_key=True)
    code = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)

class TestType(Base):
    __tablename__ = "test_types"
    __table_args__ = {"schema": "lab"}
    test_type_id = Column(Integer, primary_key=True)
    code = Column(Text, unique=True, nullable=False)
    name = Column(Text, nullable=False)

class Sample(Base):
    __tablename__ = "samples"
    __table_args__ = (
        CheckConstraint("material_ref_type IN ('material','material_component')"),
        {"schema": "lab"}
    )
    sample_id = Column(BigInteger, primary_key=True)
    project_id = Column(BigInteger, ForeignKey("lab.projects.project_id", ondelete="RESTRICT"))
    material_ref_id = Column(BigInteger, nullable=False)
    material_ref_type = Column(Text, nullable=False)
    sample_code = Column(Text, unique=True)
    geometry = Column(JSON)
    manufacture_date = Column(Date)
    notes = Column(Text)

class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = {"schema": "lab"}
    experiment_id = Column(BigInteger, primary_key=True)
    sample_id = Column(BigInteger, ForeignKey("lab.samples.sample_id", ondelete="CASCADE"))
    test_type_id = Column(Integer, ForeignKey("lab.test_types.test_type_id", ondelete="RESTRICT"))
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    status = Column(Text, default="draft")
