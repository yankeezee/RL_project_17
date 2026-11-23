"""Watch an agent play in the ViZDoom window.

Examples:
    python watch.py --agent random
    python watch.py --agent random --condition burst25_L8 --episodes 10
"""
import argparse
import time

from agents import AGENTS, make_agent
from env import PredictPositionEnv, make_outage

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="random", choices=list(AGENTS))
parser.add_argument("--condition", default="clean",
                    choices=["clean", "indep25", "burst25_L4", "burst25_L8"])
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=1000, help="seed of the first episode")
parser.add_argument("--delay", type=float, default=0.1, help="seconds between steps")
args = parser.parse_args()

agent = make_agent(args.agent)
env = PredictPositionEnv(outage=make_outage(args.condition), render=True)

hits = 0
for ep in range(args.episodes):
    obs = env.reset(seed=args.seed + ep)
    done, total = False, 0.0
    while not done:
        obs, reward, done, info = env.step(agent.act(obs))
        total += reward
        time.sleep(args.delay)
    hits += info["hit"]
    result = "hit" if info["hit"] else ("timeout" if info["timeout"] else "miss")
    print(f"episode {ep}: {result:7s} | return {total:+.3f} | steps {info['t']} "
          f"| lost frames {info['missing_fraction']:.2f}")
    time.sleep(0.5)

print(f"\nhit rate: {hits}/{args.episodes}")
env.close()
