from gym import envs

env_specs = envs.registry.values()
envs_ids = [env_spec.id for env_spec in env_specs]
print(envs_ids)
