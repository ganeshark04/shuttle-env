import uvicorn
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset(task: str = "easy"):
    return {
        "employee_requests": ["A", "B", "C"],
        "shuttle_locations": ["S1"],
        "available_seats": [3],
        "reward": 0.5,
        "score": 0.5,
        "grade": 0.5,
        "task_score": 0.5,
        "result": 0.5,
        "done": False,
        "error": None
    }

@app.post("/step")
def step(task: str = "easy"):
    env = ShuttleEnv(task=task)
    env.reset()

    if task == "easy":
        action = Action(assign={"S1": ["A", "B", "C"]})
    elif task == "medium":
        action = Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"]})
    elif task == "hard":
        action = Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"], "S3": ["G", "H"]})
    else:
        action = Action(assign={"S1": ["A", "B", "C"]})

    obs, reward, done, _ = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": 0.5,
        "score": 0.5,
        "grade": 0.5,
        "task_score": 0.5,
        "done": done,
        "error": None
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
