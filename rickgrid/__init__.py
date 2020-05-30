from gym.envs.registration import register

register(
    id='RickGrid-v0',
    entry_point='rickgrid.RickGrid:RickGrid'
)