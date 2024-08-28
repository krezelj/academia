from academia.curriculum import load_task_config

stats = []

for run_no in range(10):
    agent = ...  # initialise some agent here
    task = load_task_config('./doorkey.task.yml')
    task.run(agent)
    stats.append(task.stats)
