import uvicorn
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI()

envs = {
    "easy": ShuttleEnv(task="easy"),
    "medium": ShuttleEnv(task="medium"),
    "hard": ShuttleEnv(task="hard")
}

def clip(score):
    try:
        s = float(score)
        return round(max(0.001, min(0.999, s)), 4)
    except:
        return 0.5

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset(task: str = "easy"):
    env = envs.get(task, envs["easy"])
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(task: str = "easy"):
    env = envs.get(task, envs["easy"])

    if task == "easy":
        action = Action(assign={"S1": ["A", "B", "C"]})
    elif task == "medium":
        action = Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"]})
    elif task == "hard":
        action = Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"], "S3": ["G", "H"]})
    else:
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
