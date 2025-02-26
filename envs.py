from __future__ import annotations

import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

from minigrid.wrappers import ImgObsWrapper, PositionBonus, ActionBonus

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball

from torch.multiprocessing import Pipe, Process

from model import *
from config import *
from PIL import Image

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class BlockedUnlockPickUpEnv_v0(RoomGrid):
    def __init__(self, max_steps: int | None = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 4
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box", color="blue")
        
        # Make sure the two rooms are directly connected by a locked door
        door_1, pos_1 = self.add_door(0, 0, 0, locked=True, color="red")

        # Block the door with a ball
        color = "blue"
        self.grid.set(pos_1[0] - 1, pos_1[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door_1.color)

        # self.add_distractors(0, 0, 1)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = 1
                terminated = True

        return obs, reward, terminated, truncated, info
    
    
class BlockedUnlockPickUpEnv_v1(RoomGrid):

    def __init__(self, max_steps: int | None = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 4
        if max_steps is None:
            max_steps = 16 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=3,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(2, 0, kind="box", color="blue")
        
        # Make sure the two rooms are directly connected by a locked door
        door_1, pos_1 = self.add_door(0, 0, 0, locked=True, color="red")

        # Make sure the two rooms are directly connected by a locked door
        door_2, pos_2 = self.add_door(1, 0, 0, locked=True, color="green")


        # Block the door with a ball
        color = "yellow"
        self.grid.set(pos_1[0] - 1, pos_1[1], Ball(color))
        color = "purple"
        self.grid.set(pos_2[0] - 1, pos_2[1], Ball(color))

        # Add a key to unlock the door
        self.add_object(0, 0, "key", door_1.color)

        # Add a key to unlock the door
        self.add_object(1, 0, "key", door_2.color)

        # self.add_distractors(0, 0, 1)

        # self.add_distractors(1, 0, 1)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = 1
                terminated = True

        return obs, reward, terminated, truncated, info
    
class MinigridEnvironment(Environment):
    def __init__(self,
            env_id,
            is_render,
            env_idx,
            child_conn,
            history_size=4,
            h=7,
            w=7,
            life_done=True,
            sticky_action=True,
            p=0.25):
        super(MinigridEnvironment, self).__init__()
        self.daemon = True
        if env_id == "BlockedUnlockPickUpEnv_v0":
            self.env = ImgObsWrapper(BlockedUnlockPickUpEnv_v0())
        elif env_id == "BlockedUnlockPickUpEnv_v1":
            self.env = ImgObsWrapper(BlockedUnlockPickUpEnv_v1(max_steps=2000))
        self.env_id = env_id
        self.is_render = is_render
        self.env_idx = env_idx
        self.steps = 0
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)
        self.child_conn = child_conn

        self.sticky_action = sticky_action
        self.last_action = 0
        self.p = p

        self.history_size = history_size
        self.history = np.zeros([history_size, h, w])
        self.h = h
        self.w = w

        self.reset()

    def run(self):
        super(MinigridEnvironment, self).run()
        while True:
            action = self.child_conn.recv()

            if 'Breakout' in self.env_id:
                action += 1

            # sticky action
            if self.sticky_action:
                if np.random.rand() <= self.p:
                    action = self.last_action
                self.last_action = action

            s, reward, terminated, truncated, info = self.env.step(action)
            done = terminated

            # if max_step_per_episode < self.steps:
                # done = True
            if terminated or truncated:
                done = True

            log_reward = reward
            force_done = done

            self.history[:3, :, :] = self.history[1:, :, :]
            self.history[3, :, :] = self.pre_proc(s)

            self.rall += reward
            self.steps += 1

            if done:
                self.recent_rlist.append(self.rall)
                print(f"[Episode {self.episode}({self.env_idx})] Step: {self.steps}  Reward: {self.rall}  Recent Reward: {np.mean(self.recent_rlist)}]")
                self.history = self.reset()

            self.child_conn.send(
                [self.history[:, :, :], reward, force_done, done, log_reward])

    def reset(self):
        self.last_action = 0
        self.steps = 0
        self.episode += 1
        self.rall = 0
        s = self.env.reset()
        self.get_init_state(
            self.pre_proc(s))
        return self.history[:, :, :]

    def pre_proc(self, X):
        if isinstance(X, tuple):
        # X = np.array(Image.fromarray(X[0]).convert('L')).astype('float32')
            X = np.array(Image.fromarray(X[0]).convert('L')).astype('float32')
        else:
            X = np.array(Image.fromarray(X).convert('L')).astype('float32')

        # X = np.array((X[0])).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        # x = cv2.resize(X, (7, 7))
        return x

    def get_init_state(self, s):
        for i in range(self.history_size):
            self.history[i, :, :] = self.pre_proc(s)    

# class MaxAndSkipEnv(gym.Wrapper):
#     def __init__(self, env, is_render, skip=4):
#         """Return only every `skip`-th frame"""
#         gym.Wrapper.__init__(self, env)
#         # most recent raw observations (for max pooling across time steps)
#         self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
#         self._skip = skip
#         self.is_render = is_render

#     def step(self, action):
#         """Repeat action, sum reward, and max over last observations."""
#         total_reward = 0.0
#         done = None
#         for i in range(self._skip):
#             obs, reward, done, info = self.env.step(action)
#             if self.is_render:
#                 self.env.render()
#             if i == self._skip - 2:
#                 self._obs_buffer[0] = obs
#             if i == self._skip - 1:
#                 self._obs_buffer[1] = obs
#             total_reward += reward
#             if done:
#                 break
#         # Note that the observation on the done=True frame
#         # doesn't matter
#         max_frame = self._obs_buffer.max(axis=0)

#         return max_frame, total_reward, done, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)


# class MontezumaInfoWrapper(gym.Wrapper):
#     def __init__(self, env, room_address):
#         super(MontezumaInfoWrapper, self).__init__(env)
#         self.room_address = room_address
#         self.visited_rooms = set()

#     def get_current_room(self):
#         ram = unwrap(self.env).ale.getRAM()
#         assert len(ram) == 128
#         return int(ram[self.room_address])

#     def step(self, action):
#         obs, rew, terminated, truncated, info = self.env.step(action)
#         self.visited_rooms.add(self.get_current_room())

#         if 'episode' not in info:
#             info['episode'] = {}
#         info['episode'].update(visited_rooms=copy(self.visited_rooms))

#         if terminated or truncated:
#             self.visited_rooms.clear()
#         return obs, rew, terminated, truncated, info

#     def reset(self):
#         return self.env.reset()


# class AtariEnvironment(Environment):
#     def __init__(
#             self,
#             env_id,
#             is_render,
#             env_idx,
#             child_conn,
#             history_size=4,
#             h=84,
#             w=84,
#             life_done=True,
#             sticky_action=True,
#             p=0.25):
#         super(AtariEnvironment, self).__init__()
#         self.daemon = True
#         self.env = MaxAndSkipEnv(gym.make(env_id), is_render)
#         if 'Montezuma' in env_id:
#             self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
#         self.env_id = env_id
#         self.is_render = is_render
#         self.env_idx = env_idx
#         self.steps = 0
#         self.episode = 0
#         self.rall = 0
#         self.recent_rlist = deque(maxlen=100)
#         self.child_conn = child_conn

#         self.sticky_action = sticky_action
#         self.last_action = 0
#         self.p = p

#         self.history_size = history_size
#         self.history = np.zeros([history_size, h, w])
#         self.h = h
#         self.w = w

#         self.reset()

#     def run(self):
#         super(AtariEnvironment, self).run()
#         while True:
#             action = self.child_conn.recv()

#             if 'Breakout' in self.env_id:
#                 action += 1

#             # sticky action
#             if self.sticky_action:
#                 if np.random.rand() <= self.p:
#                     action = self.last_action
#                 self.last_action = action

#             s, reward, terminated, truncated, info = self.env.step(action)

#             if max_step_per_episode < self.steps:
#                 done = True

#             log_reward = reward
#             force_done = done

#             self.history[:3, :, :] = self.history[1:, :, :]
#             self.history[3, :, :] = self.pre_proc(s)

#             self.rall += reward
#             self.steps += 1

#             if done:
#                 self.recent_rlist.append(self.rall)
#                 # print("[Episode {}({})] Step: {}  Reward: {}  Recent Reward: {}  Visited Room: [{}]".format(
#                 #     self.episode, self.env_idx, self.steps, self.rall, np.mean(self.recent_rlist),
#                 #     info.get('episode', {}).get('visited_rooms', {})))
                
#                 print(f"[Episode {self.episode}({self.env_idx})] Step: {self.steps}  Reward: {self.rall}  Recent Reward: {np.mean(self.recent_rlist)}]")

#                 self.history = self.reset()

#             self.child_conn.send(
#                 [self.history[:, :, :], reward, force_done, done, log_reward])

#     def reset(self):
#         self.last_action = 0
#         self.steps = 0
#         self.episode += 1
#         self.rall = 0
#         s = self.env.reset()
#         self.get_init_state(
#             self.pre_proc(s))
#         return self.history[:, :, :]

#     def pre_proc(self, X):
#         X = np.array(Image.fromarray(X[0]).convert('L')).astype('float32')
#         x = cv2.resize(X, (self.h, self.w))
#         return x

#     def get_init_state(self, s):
#         for i in range(self.history_size):
#             self.history[i, :, :] = self.pre_proc(s)


