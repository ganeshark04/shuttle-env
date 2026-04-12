import uvicorn
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/reset")
def reset(task: str = "easy"):
    env = ShuttleEnv(task=task)
    obs = env.reset()
    return {
        "employee_requests": obs.employee_requests,
        "shuttle_locations": obs.shuttle_locations,
        "available_seats": obs.available_seats
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
        "observation": {
            "employee_requests": obs.employee_requests,
            "shuttle_locations": obs.shuttle_locations,
            "available_seats": obs.available_seats
        },
        "reward": reward,
        "done": done,
        "error": None
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
