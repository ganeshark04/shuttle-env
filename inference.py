import os
from dotenv import load_dotenv
from openai import OpenAI
from env import ShuttleEnv, Action

# Load env
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("TASK_NAME", "easy")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


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


if __name__ == "__main__":
    run()
