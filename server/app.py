import uvicorn
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
    action = Action(assign={"S1": ["A", "B", "C"]})
    obs, reward, done, _ = env.step(action)

    # --- FIX FOR PHASE 2: NORMALIZE REWARD ---
    # Ensures reward is strictly between 0 and 1
    score = float(reward) / 10.0
    if score <= 0.0: score = 0.01
    if score >= 1.0: score = 0.99

    return {
        "observation": obs.dict(),
        "reward": score, # Send the normalized score
        "done": done,
        "error": None
    }

@app.get("/state")
def state():
    return env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
