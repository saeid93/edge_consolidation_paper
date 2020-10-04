from experiments_scripts.check_scripts import CheckScripts


# ins = CheckScripts(dataset=2, type_env=1)
ins = CheckScripts.with_model(dataset=2, type_env=1, model=3)

# ins.is_on_GPU()
# ins.check_env()
# ins.learn(total_timesteps=15)
ins.check_learned()
