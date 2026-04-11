import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from env import ShuttleEnv, Action
from typing import Optional

app = FastAPI()

class TaskRequest(BaseModel):
    task: Optional[str] = "easy"

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset(request: TaskRequest = None):
    task = request.task if request else "easy"
    env = ShuttleEnv(task=task)
    obs = env.reset()
    return {
        "employee_requests": obs.employee_requests,
        "shuttle_locations": obs.shuttle_locations,
        "available_seats": obs.available_seats
    }

@app.post("/step")
def step(request: TaskRequest = None):
    task = request.task if request else "easy"
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
    score = round(max(0.001, min(0.999, float(env.grade()))), 4)

    return {
        "observation": {
            "employee_requests": obs.employee_requests,
            "shuttle_locations": obs.shuttle_locations,
            "available_seats": obs.available_seats
        },
        "reward": score,
        "done": done,
        "error": None
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
