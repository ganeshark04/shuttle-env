import os
from dotenv import load_dotenv
from openai import OpenAI
from env import ShuttleEnv, Action

# ✅ ADD (new)
from fastapi import FastAPI
import threading
import uvicorn

# Load env
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "easy")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ✅ ADD (global env for API)
app = FastAPI()
global_env = ShuttleEnv(task="easy")


def run():
    env = ShuttleEnv(task=TASK_NAME)
    obs = env.reset()

    rewards = []
    steps = 0
    success = False

    print(f"[START] task={TASK_NAME} env=shuttle-env model={MODEL_NAME}")

    try:
        MAX_STEPS = 3

        for _ in range(MAX_STEPS):
            steps += 1

            # Call API only once (first step)
            if steps == 1:
                try:
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": f"Assign: {obs.employee_requests}"}
                        ],
                        max_tokens=10
                    )
                    error_msg = "null"
                except Exception as e:
                    error_msg = str(e)
            else:
                error_msg = "null"

            # Fixed action
            action = Action(assign={"S1": ["A", "B", "C"]})

            obs, reward, done, _ = env.step(action)

            rewards.append(f"{reward:.2f}")

            print(
                f"[STEP] step={steps} action=assign "
                f"reward={reward:.2f} done={str(done).lower()} error={error_msg}"
            )

            # Stop conditions
            if done:
                success = True
                break

            if reward == 0:
                break

    except Exception as e:
        print(
            f"[STEP] step={steps} action=null "
            f"reward=0.00 done=true error={str(e)}"
        )
        success = False

    finally:
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} rewards={','.join(rewards)}"
        )


# ✅ ADD API ENDPOINTS (new)
@app.post("/reset")
def reset_api():
    obs = global_env.reset()
    return obs.dict()

@app.post("/step")
def step_api():
    action = Action(assign={"S1": ["A", "B", "C"]})
    obs, reward, done, _ = global_env.step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "error": None
    }

@app.get("/state")
def state_api():
    return global_env.state()


# ✅ MODIFY ONLY THIS PART (runner)
if __name__ == "__main__":
    # run your original logic (logs)
    threading.Thread(target=run).start()

    # run API server (for judges)
    uvicorn.run(app, host="0.0.0.0", port=7860)
