
from envs import *
from minigrid.wrappers import ImgObsWrapper

def test(agent, work, env):
    done = False
    truncated = False
    all_reward = []
    ob, _ = env.reset()
    while len(all_reward) < 50:
        ob = work.pre_proc(ob)
        action, _, _, _ = agent.get_action(ob)
        action = int(action)
        next_ob, rew, done, truncated, _ = env.step(action)

        if done or truncated:
            all_reward.append(rew)
            ob, _ = env.reset()
        else:
            ob = next_ob
    
    print("Length all reward", len(all_reward))
    return np.mean(all_reward)

