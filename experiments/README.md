## Experiments

| Who | Environment | Experiment Name | Agent | Runs | Detailed Description |
|-|-|-|-|-|-|
Szymon | LavaCrossing | Epsilon Reset Impact | DQN | 5x10 | 1.0(0.999), 0.6(0.9994), 0.3(0.9995), 0.1(0.9996), 0.03(0.9997) on 0,1,2 @ 0.8
Maciek | DoorKey | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,2 vs 2
Janek | LunarLander | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,2,3 vs 3
|| LavaCrossing | Time/Evaluation Impact | PPO | 2x5x5 | 500/1000/1500/2000/2500 vs (0.8/1000) on 1,2
|| DoorKey | Time/Evaluation Impact | DQN | 2x5x5 | 500/1000/1500/2000/2500 vs (0.8/1000) on 1,2
|| LunarLander | Task Skip Curricula | DQN | 4x10 | 0,1,2,3 vs 0,1,3 vs 0,3 vs 0, 2,3
|| MsPacman | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,3 vs 3
|| BridgeBuilding | Sparse vs Dense Reward | PPO | 2x10 | 0,1,2
|| DoorKey | Task Skip Curricula | PPO | 2x10 | 0,1,2 vs 0,2

- experiments
    - experiment_1
        - outputs
            - curriculum_1
                - *.json
                - *.zip
            - curriculum_2
            - nocurriculum_1
                - *.json
                - *.zip
            - nocurriculum_1
        - configs
            - *.yml
        - README.md
        - run.py
        - results.ipynb
        - meta.txt