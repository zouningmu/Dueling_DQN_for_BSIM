from gymnasium.envs.registration import register
from gym_bsim.envs.bsim import BSIMEnv

register(
    id="gym_bsim/bsim-v0",
    entry_point="gym_bsim.envs.bsim:BSIMEnv",
    max_episode_steps=300,
)