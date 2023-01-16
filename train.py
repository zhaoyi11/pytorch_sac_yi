#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils
from env import make_env
from agent.sac import SACAgent

import hydra
import wandb

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()

        print(f'workspace: {self.work_dir}')

        if cfg.seed == -1:
            import random
            cfg.seed = random.randint(0, 1000)

        cfg.num_train_steps = int(cfg.num_train_steps / cfg.action_repeat) 
        cfg.replay_buffer_capacity = int(cfg.replay_buffer_capacity / cfg.action_repeat) 
        cfg.num_seed_steps = int(cfg.num_seed_steps / cfg.action_repeat)
        cfg.eval_frequency = int(cfg.eval_frequency / cfg.action_repeat)
        cfg.max_episode_length = int(cfg.max_episode_length / cfg.action_repeat)
        cfg.log_frequency = int(cfg.log_frequency / cfg.action_repeat)
        self.cfg = cfg

        self.logger = Logger(self.work_dir + f'/logs/{cfg.env}/{cfg.seed}',
                             save_tb=cfg.log_save_tb,
                             log_frequency=int(cfg.log_frequency),
                             agent=cfg.agent.name)
        if self.cfg.use_wandb:
            # wandb.init()
            pass # TODO:

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_env(cfg.env, cfg.seed, cfg.action_repeat)

        cfg.agent.params.obs_dim = int(self.env.observation_space.shape[0])
        cfg.agent.params.action_dim = int(self.env.action_space.shape[0])
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        # self.agent = hydra.utils.instantiate(cfg.agent)
        self.agent = SACAgent(**cfg.agent.params)

        self.replay_buffer = ReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.device)

        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step, self.env_step = 0, 0

    def evaluate(self):
        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(f'{self.env_step}.mp4')
        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.env_step)
        self.logger.dump(self.env_step)

    def run(self):
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration',
                                    time.time() - start_time, self.env_step)
                    start_time = time.time()
                    self.logger.dump(
                        self.env_step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.env_step)
                    self.evaluate()

                self.logger.log('train/episode_reward', episode_reward,
                                self.env_step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.env_step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.cfg.max_episode_length else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done,
                                   done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            self.env_step += self.cfg.action_repeat


@hydra.main(config_path='config', config_name='train.yaml')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
