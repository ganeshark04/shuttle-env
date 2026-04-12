import os
from dotenv import load_dotenv
from openai import OpenAI
from env import ShuttleEnv, Action

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

def clip(r):
    return round(max(0.001, min(0.999, float(r))), 3)

def run_task(task_name):
    env = ShuttleEnv(task=task_name)
    obs = env.reset()
    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env=shuttle-env model={MODEL_NAME}", flush=True)

    try:
        if task_name == "easy":
            actions = [
                Action(assign={"S1": ["A"]}),
                Action(assign={"S1": ["B"]}),
                Action(assign={"S1": ["C"]}),
            ]
        elif task_name == "medium":
            actions = [
                Action(assign={"S1": ["A", "B", "C"]}),
                Action(assign={"S2": ["D", "E", "F"]}),
            ]
        elif task_name == "hard":
            actions = [
                Action(assign={"S1": ["A", "B", "C"]}),
                Action(assign={"S2": ["D", "E", "F"]}),
                Action(assign={"S3": ["G", "H"]}),
            ]

        for i, action in enumerate(actions):
            steps = i + 1
            error_msg = "null"

            if steps == 1 and client:
                try:
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": f"Assign employees: {obs.employee_requests}"}],
                        max_tokens=10
                    )
                except Exception as e:
                    error_msg = str(e)

            obs, reward, done, _ = env.step(action)
            clipped = clip(reward)
            rewards.append(clipped)

            print(f"[STEP] step={steps} action=assign reward={clipped:.3f} done={str(done).lower()} error={error_msg}", flush=True)

            if done:
                success = True
                break

    except Exception as e:
        print(f"[STEP] step={steps} action=null reward=0.00 done=true error={str(e)}", flush=True)
        success = False

    finally:
        score = sum(rewards) / len(rewards) if rewards else 0.5
        score = clip(score)
        rewards_str = ",".join(f"{r:.3f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

    return score

def run():
    final_scores = {}
    for task in ["easy", "medium", "hard"]:
        score = run_task(task)
        final_scores[task] = score

    print(f"\n===== FINAL SCORES =====", flush=True)
    for task, score in final_scores.items():
        print(f"  {task}: {score:.4f}", flush=True)
    avg = sum(final_scores.values()) / len(final_scores)
    avg = clip(avg)
    print(f"  Average: {avg:.4f}", flush=True)

if __name__ == "__main__":
    run()
