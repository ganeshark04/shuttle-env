from env import ShuttleEnv, Action

env = ShuttleEnv(task="easy")
obs = env.reset()

action = Action(assign={"S1": ["A", "B", "C"]})

obs, reward, done, _ = env.step(action)

print(obs)
print(reward, done)