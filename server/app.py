import uvicorn
from fastapi import FastAPI
from env import ShuttleEnv, Action

app = FastAPI(version="1.0.0")

@app.get("/")
def home():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    return {
        "name": "shuttle-routing-env",
        "description": "Dynamic routing of office shuttles based on real-time employee demand"
    }

@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "assign": {"type": "object"}
            }
        },
        "observation": {
            "type": "object",
            "properties": {
                "employee_requests": {"type": "array"},
                "shuttle_locations": {"type": "array"},
                "available_seats": {"type": "array"}
            }
        },
        "state": {
            "type": "object",
            "properties": {
                "remaining": {"type": "array"},
                "picked": {"type": "array"},
                "steps": {"type": "integer"}
            }
        }
    }

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

@app.get("/state")
def state(task: str = "easy"):
    env = ShuttleEnv(task=task)
    env.reset()
    return env.state()

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
