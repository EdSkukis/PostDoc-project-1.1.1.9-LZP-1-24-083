import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import projects, materials, samples, experiments

app = FastAPI(title="Exp Lab Platform API")

# origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(materials.router)
app.include_router(samples.router)
app.include_router(experiments.router)
