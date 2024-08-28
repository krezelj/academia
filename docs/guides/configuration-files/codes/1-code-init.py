from academia.curriculum import LearningTask, Curriculum
from academia.environments import LavaCrossing

# define tasks
task1 = LearningTask(
    env_type=LavaCrossing,
    env_args={'difficulty': 0, 'render_mode': 'human', 'append_step_count': True},
    stop_conditions={'max_episodes': 500},
)
task2 = LearningTask(
    env_type=LavaCrossing,
    env_args={'difficulty': 1, 'render_mode': 'human', 'append_step_count': True},
    stop_conditions={'max_episodes': 1000},
)

# define a curriculum
curriculum = Curriculum(
    tasks=[task1, task2],
    output_dir='./my_curriculum/',
)
