# NoveltyGate Lab: full-stack demo
Budget-aware novelty gating for camera-trap nodes. FastAPI backend runs the node/controller simulator on real
Snapshot Serengeti scores (DINOv2-small, PCA, 64 prototypes); the frontend is a dashboard that calls the API.
Scores/labels are real; battery, solar and radio costs are illustrative parameters (see backend/simulator.py).

## Run locally
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload
    # open http://127.0.0.1:8000   (API docs: /docs)

## API
- GET  /api/meta      model info + test AUROC per embedding size
- POST /api/simulate  body: {dim, budget, rate, epd, cap, solar, seed} -> table + battery traces per camera

## Deploy
- Easiest: deploy the whole folder as one service on Render (Web Service, root `backend`, build `pip install -r requirements.txt`,
  start `uvicorn main:app --host 0.0.0.0 --port $PORT`). It serves the frontend too. Free tier sleeps when idle (~30 s wake-up).
- Split hosting: put `frontend/` on Vercel and open it with `?api=https://YOUR-BACKEND-URL` (CORS is enabled).

## Data
backend/data/replay_events.csv and model_meta.json come from export_for_app.py. Re-run it and replace them to update results.
