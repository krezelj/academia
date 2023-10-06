import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class DQN:
    def __init__(self, state_size: int, 
                 num_of_actions: int, gamma: float =1., 
                 epsilon: float =1., epsilon_decay: float =0.99,
                 epsilon_min: float =0.01, learning_rate: float =0.01
                 ):
        self.state_size = state_size
        self.num_of_actions = num_of_actions
        self.memory = np.array([])
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.network = self._build_network_()
        self.target_network = self._build_network_()

    def _build_network_(self):
        network = Sequential()
        network.add(Dense(24, input_dim=self.state_size, activation='relu'))
        network.add(Dense(24, activation='relu'))
        network.add(Dense(self.num_of_actions, activation='linear'))
        network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return network
    
    def remember(self, state, action, reward, next_state, done):
        np.append(self.memory, (state, action, reward, next_state, done))

    def get_action(self, state):
        if np.random.uniform() <= self.epsilon:
            return np.random.randint(0, self.num_of_actions)
        else:
            q_val_act = self.network.predict(state)
            return np.argmax(q_val_act)
        
    def update_target(self):
        self.target_network.set_weights(self.network)
    
    def replay(self, batch_size):
        batch_indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[i] for i in batch_indices]
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_action = np.argmax(self.network.predict(next_state))
                target = reward + self.gamma * self.target_network.predict(next_state)[next_action]
            target 