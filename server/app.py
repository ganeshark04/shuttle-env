import uvicorn
import os
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI()
env = ShuttleEnv(task="easy")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step():
    # Your original logic
    action = Action(assign={"S1": ["A", "B", "C"]})
    obs, reward, done, _ = env.step(action)

    # --- THE FIX: Change 6.00 to 0.60 ---
    # We divide by 10 to get a score between 0 and 1
    score = float(reward) / 10.0
    
    # Strictly between 0 and 1 (0.0 and 1.0 are NOT allowed)
    if score <= 0.0: score = 0.01
    if score >= 1.0: score = 0.99

    return {
        "observation": obs.dict(),
        "reward": score, 
        "done": done,
        "error": None
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
