from gym.envs.registration import register
from gym_edgesimulator.simulator.simulator import Simulator


register(
    id='EdgeSim-v0',
    entry_point='gym_edgesimulator.envs:EdgeSimV0',
)

register(
    id='EdgeSim-v1',
    entry_point='gym_edgesimulator.envs:EdgeSimV1',
)

register(
    id='EdgeSim-v2',
    entry_point='gym_edgesimulator.envs:EdgeSimV2',
)

register(
    id='EdgeSim-v3',
    entry_point='gym_edgesimulator.envs:EdgeSimV3',
)