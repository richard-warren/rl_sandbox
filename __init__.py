from gym.envs.registration import register

register(
    id='RickGrid-v0',
    entry_point='rl_bootcamp.envs.RickGrid:RickGrid'
)