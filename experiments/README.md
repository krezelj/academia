## Experiments

ID | Who | Environment | Experiment Name | Agent | Runs | Detailed Description |
|-|-|-|-|-|-|-|
1 | Szymon | LavaCrossing | Epsilon Reset Impact | DQN | 5x10 | 1.0(0.999), 0.6(0.9994), 0.3(0.9995), 0.1(0.9996), 0.03(0.9997) on 0,1,2 @ 0.8
2 | Maciek | DoorKey | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,2 vs 2
3 | Janek | LunarLander | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,2,3 vs 3
4 | Szymon & Maciek | LavaCrossing | Time/Evaluation Impact | PPO | 2x5x10 | 1500/2500/3500/4500/5500 vs (0.8/2000) on 0,1
5 | Szymon | DoorKey | Time/Evaluation Impact | PPO | 2x5x10 | 750/1000/1250/1500/1750 vs (0.8/1500) on 1,2
6 | Janek | LunarLander | Task Skip Curricula | DQN | 2x10 | 0,4  vs 0,3,4
7 | Maciej & Janek | MsPacman | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,3 vs 3
8 | Janek | BridgeBuilding | Sparse vs Dense Reward | PPO | 4x10 | 0,1,2
9 | Maciek | DoorKey | Task Skip Curricula | PPO | 2x10 | 0,1,2 vs 0,2
10 | Szymon | LunarLander | Curriculum vs No Curriculum | DQN/PPO | 2x2x10 | 0,1,2,3,4 vs 4
11 | Janek | LunarLander | Epsilon 0.6 with Task Skip | DQN | 1x10 | 0,4

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
