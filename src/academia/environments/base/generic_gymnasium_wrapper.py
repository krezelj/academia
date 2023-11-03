from typing import Any
from collections import deque
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import gymnasium

from . import ScalableEnvironment


class GenericGymnasiumWrapper(ScalableEnvironment):
    """
    A wrapper for Gymnasium environments. The purpose of it is to contain common Gymnasium syntax so that
    it does not have to be copied and pasted in every wrapper. At the same time, it aims to deliver
    flexibility that is required to handle generalised nature of Gymnasium API such as varying state
    representations.

    Args:
        difficulty: The difficulty level of the environment.
        environment_id: Gymnasium environment ID.
        n_frames_stacked: How many most recent states should be stacked together to form a final state
            representation.
        kwargs: Arguments passed down to ``gymnasium.make``

    Attributes:
        step_count (int): Current step count since the last reset.
        difficulty (int): Difficulty level. Higher values indicate more difficult environments.
        n_frames_stacked (int): How many most recent states should be stacked together to form a final state
            representation. Defaults to 1.
    """

    def __init__(self, difficulty: int, environment_id: str, n_frames_stacked: int = 1, **kwargs):
        super().__init__(
            difficulty=difficulty,
            n_frames_stacked=n_frames_stacked,
        )
        self._base_env = gymnasium.make(environment_id, **kwargs)
        self.step_count = 0
        self._state = None  # properly set in self.reset()
        """note: self._state IS NOT STACKED. To obtain a stacked state use self.observe()"""

        self._past_n_states = deque()  # properly set in self.reset()
        self.reset()
        self.STATE_SIZE = len(self.observe())

    def step(self, action: int) -> tuple[Any, float, bool]:
        """
        Advances the environment by one step given the specified action.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the new state, reward, and a flag indicating episode termination.
        """
        action_transformed = self._transform_action(action)
        new_state, reward, terminated, truncated, _ = self._base_env.step(action_transformed)
        self.step_count += 1
        self._state = self._transform_state(new_state)

        # frame stacking
        self._past_n_states.append(self._state)
        self._past_n_states.popleft()

        is_episode_end = terminated or truncated
        return self.observe(), float(reward), is_episode_end

    def observe(self) -> Any:
        """
        Returns the current state of the environment. Performs state stacking if :attr:`n_frames_stacked` is
        greater than 1.

        Returns:
            The current state of the environment.
        """
        stacked_state = np.concatenate(list(self._past_n_states))
        return stacked_state

    def get_legal_mask(self) -> npt.NDArray[int]:
        """
        Notes:
            For all gymnasium-based environments in this package (and probably most in general) it is hard
            to cheaply obtain a legal mask, so this default implementation always returns an array of ones

        Returns:
            A binary mask with 0s in place for illegal actions (actions that
            have no effect) and 1s for legal actions.
        """
        return np.array([1 for _ in range(self.N_ACTIONS)])

    def reset(self) -> Any:
        """
        Resets the environment to its initial state.

        Returns:
            The new state after resetting the environment.
        """
        self._state = self._transform_state(self._base_env.reset()[0])
        self._past_n_states = deque([self._state for _ in range(self.n_frames_stacked)])
        self.step_count = 0
        # after resetting there's only one state so it doesn't make any difference
        # whether self.observe() or self._state is returned.
        return self._state

    def render(self) -> None:
        """
        Renders the environment in the current render mode.
        """
        self._base_env.render()

    def _transform_action(self, action: int) -> int:
        """
        Transforms an action ID in case mappings used by agents are different from action mappings in
        the underlying environment. The default implementation assumes that both mappings are identical -
        otherwise this method has to be overriden.

        Args:
            action: Action ID according to the package mapping

        Returns:
            Action ID according to the underlying environment mapping
        """
        return action

    @abstractmethod
    def _transform_state(self, raw_state: Any) -> npt.NDArray[np.float32]:
        """
        Transforms a state returned by the underlying environment to a numpy array, which is a format
        commonly used throughout this package.
        """
        pass
