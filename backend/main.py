"""NoveltyGate Lab API.  Run:  uvicorn main:app --reload   then open http://127.0.0.1:8000"""
import json
from pathlib import Path
from typing import Literal
import numpy as np, pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from simulator import run_all

HERE = Path(__file__).parent
META = json.loads((HERE / "data/model_meta.json").read_text())


def load():
    ev = pd.read_csv(HERE / "data/replay_events.csv")
    cams = []
    for cam, g in ev[ev.split == "test"].groupby("camera_id"):
        g = g.sort_values("order_in_camera")
        cams.append({"id": cam, "n": (g.is_novel.astype(int) + 2 * g.is_empty.astype(int)).tolist(),
                     "s": {d: g[f"score_{d}"].tolist() for d in (8, 16, 32, 64)}})
    v = ev[ev.split == "val"]
    vq = {d: np.quantile(v[f"score_{d}"], np.linspace(0, 1, 101)).tolist() for d in (8, 16, 32, 64)}
    return {"cams": cams, "vq": vq}


DATA = load()
app = FastAPI(title="NoveltyGate Lab")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Params(BaseModel):
    dim: Literal[8, 16, 32, 64] = 64
    budget: float = Field(200, gt=0, le=100000)
    rate: float = Field(0.12, gt=0, le=1)
    epd: int = Field(24, ge=1, le=500)
    cap: float = Field(1500, gt=0)
    solar: float = Field(130, ge=0)
    seed: int = 1
    hdr: int = 4
    start: float = 0.6
    eEnc: float = 1.0
    eTx0: float = 5.0
    eTxB: float = 0.35
    eIdle: float = 2.0


@app.get("/api/meta")
def meta():
    return {**META, "n_cameras": len(DATA["cams"])}


@app.post("/api/simulate")
def simulate(P: Params):
    results, traces = run_all(DATA, P)
    return {"results": results, "traces": traces, "cameras": [c["id"] for c in DATA["cams"]],
            "auroc": META["test_auroc"].get(f"{P.dim}d")}


app.mount("/", StaticFiles(directory=HERE.parent / "frontend", html=True), name="frontend")
