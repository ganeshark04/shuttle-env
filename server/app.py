from fastapi import FastAPI
from env import ShuttleEnv, Action
import uvicorn

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

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "error": None
    }

@app.get("/state")
def state():
    return env.state()

# ✅ REQUIRED main function
def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)
    if __name__ == "__main__":
    main()
