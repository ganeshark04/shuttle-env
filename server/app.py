import uvicorn
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI()
env = ShuttleEnv(task="easy")

def clip(score):
    return round(max(0.001, min(0.999, float(score))), 4)

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
    safe_score = clip(env.grade())

    return {
        "observation": obs.dict(),
        "reward": safe_score,
        "done": done,
        "error": None
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
