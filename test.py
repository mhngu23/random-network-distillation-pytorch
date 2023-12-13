
from envs import *
from minigrid.wrappers import ImgObsWrapper

history = np.zeros([4, 84, 84])

def test_minigrid(agent, work, env):
    done = False
    truncated = False
    all_reward = []
    ob, _ = env.reset()
    while len(all_reward) < 50:

        history[:3, :, :] = history[1:, :, :]
        history[3, :, :] = work.pre_proc(ob)

        # ob = work.pre_proc(ob)
        history = np.expand_dims(history, axis=0)
        history = np.float32(history) / 255.
        print(history)
        exit()
        action, _, _, _ = agent.get_action(history)
        action = int(action)
        next_ob, rew, done, truncated, _ = env.step(action)

        if done or truncated:
            all_reward.append(rew)
            ob, _ = env.reset()
        else:
            ob = next_ob
    
    # print("Length all reward", len(all_reward))
    return np.mean(all_reward)



