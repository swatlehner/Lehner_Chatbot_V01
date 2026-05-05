# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:09:59 2026

@author: Wetzel
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from document_qa import retrieve_chunks, ask_gemini,ask_gemini_stream


import os

ENV = os.getenv("ENV", "dev")

if ENV == "dev":
    origins = ["*"]
    allow_credentials = False
else:
    origins = [
        "https://yourdomain.com",
        "https://www.yourdomain.com"
    ]
    allow_credentials = False

app = FastAPI()

# allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Query(BaseModel):
    question: str
    session_id: str
    
@app.post("/ask")
def ask(q: Query):

    def generator():
        for token in ask_gemini_stream(q.question, q.session_id):
            yield token

    return StreamingResponse(generator(), media_type="text/plain")

@app.get("/")
def root():
    return {"status": "API running"}