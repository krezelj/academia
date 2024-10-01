from academia.agents.base import Agent
from academia.curriculum import LearningStats, load_curriculum_config


def my_task_callback(agent: Agent, stats: LearningStats, task_id: str) -> None:
    agent.reset_exploration(0.8)


task = load_curriculum_config('my_curriculum.yml', variables={
    'task_callback': my_task_callback,
})
