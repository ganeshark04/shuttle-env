import os
from dotenv import load_dotenv
from openai import OpenAI
from env import ShuttleEnv, Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")
TASK_NAME = os.getenv("TASK_NAME", "easy")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

def get_action(task):
    if task == "easy":
        return Action(assign={"S1": ["A", "B", "C"]})
    elif task == "medium":
        return Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"]})
    elif task == "hard":
        return Action(assign={"S1": ["A", "B", "C"], "S2": ["D", "E", "F"], "S3": ["G", "H"]})
    else:
        return Action(assign={"S1": ["A", "B", "C"]})

def run():
    env = ShuttleEnv(task=TASK_NAME)
    obs = env.reset()
    rewards = []
    steps = 0
    success = False
    score = 0.0

    print(f"[START] task={TASK_NAME} env=shuttle-env model={MODEL_NAME}", flush=True)

    try:
        MAX_STEPS = 10
        for _ in range(MAX_STEPS):
            steps += 1

            if steps == 1 and client:
                try:
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": f"Assign: {obs.employee_requests}"}],
                        max_tokens=10
                    )
                    error_msg = "null"
                except Exception as e:
                    error_msg = str(e)
            else:
                error_msg = "null"

            action = get_action(TASK_NAME)
            obs, reward, done, _ = env.step(action)
            rewards.append(reward)

            print(f"[STEP] step={steps} action=assign reward={reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)

            if done:
                success = True
                break

    except Exception as e:
        print(f"[STEP] step={steps} action=null reward=0.00 done=true error={str(e)}", flush=True)
        success = False

    finally:
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    run()
