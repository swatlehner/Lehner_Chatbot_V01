# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:09:59 2026

@author: Wetzel
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from document_qa import ask_gemini_stream


import os

ENV = os.getenv("ENV", "dev")


app = FastAPI()

# allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "https://swatlehner.github.io",
        "https://www.lehner.eu"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Query(BaseModel):
    question: str
    session_id: str
    manual: str
    
@app.post("/ask")
def ask(q: Query):

    def generator():
        for token in ask_gemini_stream(q.question, q.session_id, q.manual):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")

@app.get("/")
def root():
    return {"status": "API running"}