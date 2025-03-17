from gymnasium.envs.registration import register

register(
    id='SimpleGrid-proj',
    entry_point='custom_envs.environment:SimpleGridEnv',
    max_episode_steps=200
)


register(
    id='MovingObstaclesGrid-v0',
    entry_point='custom_envs.environment:DynamicGridEnv',
    max_episode_steps=200
)

