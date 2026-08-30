"""Language wrapper for handling text observations in vectorized environments."""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


class LanguageWrapper(gym.Wrapper):
    """Wrapper that handles language/text observations properly in vectorized environments.

    This wrapper ensures that text-based observations are preserved when environments
    are vectorized using SyncVectorEnv or AsyncVectorEnv.
    """

    LANGUAGE_KEY = "annotation.human.action.task_description"

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._language_instruction = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment and preserve language instruction."""
        obs, info = self.env.reset(seed=seed, options=options)

        # Store the language instruction if present
        if self.LANGUAGE_KEY in obs:
            self._language_instruction = obs[self.LANGUAGE_KEY]
            # Ensure it's preserved in the observation
            obs[self.LANGUAGE_KEY] = self._language_instruction

        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Step the environment and preserve language instruction."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Ensure language instruction is preserved across steps
        if self._language_instruction is not None:
            obs[self.LANGUAGE_KEY] = self._language_instruction
        elif self.LANGUAGE_KEY in obs:
            # Update stored instruction if it changes
            self._language_instruction = obs[self.LANGUAGE_KEY]

        return obs, reward, terminated, truncated, info


class LanguageVectorWrapper(gym.Wrapper):
    """Special wrapper for vectorized environments to handle language observations.

    This wrapper is applied after vectorization to handle the special case of
    text observations that don't stack like numpy arrays.
    """

    LANGUAGE_KEY = "annotation.human.action.task_description"

    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        self.is_vector_env = True
        self._language_instructions = None

    def reset(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None) -> Tuple[Any, List[dict]]:
        """Reset vectorized environment and handle language instructions."""
        obs, infos = self.env.reset(seed=seed, options=options)

        # Handle language instructions for vectorized observations
        obs = self._process_vectorized_obs(obs, infos)

        return obs, infos

    def step(self, actions: Any) -> Tuple[Any, Any, Any, Any, List[dict]]:
        """Step vectorized environment and handle language instructions."""
        obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # Handle language instructions for vectorized observations
        obs = self._process_vectorized_obs(obs, infos)

        return obs, rewards, terminateds, truncateds, infos

    def _process_vectorized_obs(self, obs: Any, infos: List[dict]) -> Any:
        """Process vectorized observations to handle language instructions.

        Args:
            obs: Vectorized observations (could be dict or array)
            infos: List of info dicts from each environment

        Returns:
            Processed observations with language instructions preserved
        """
        if isinstance(obs, dict):
            # Check if we have stored language instructions
            if self._language_instructions is None:
                # Try to extract from first environment's observation
                # This is a workaround for SyncVectorEnv dropping non-array fields
                self._language_instructions = []
                for i, info in enumerate(infos):
                    # Try to get from info first
                    if self.LANGUAGE_KEY in info:
                        self._language_instructions.append(info[self.LANGUAGE_KEY])
                    else:
                        # Use a default instruction
                        self._language_instructions.append("Complete the task")

            # Add language instructions to observations
            if self._language_instructions and self.LANGUAGE_KEY not in obs:
                # For vectorized envs, we need to handle this specially
                # Just use the first instruction for all envs (they should be the same for the same task)
                obs[self.LANGUAGE_KEY] = self._language_instructions[0]

        return obs