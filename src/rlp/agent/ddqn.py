from __future__ import annotations

import torch

from rlp.agent.dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """
    Double DQN Agent.
    
    Uses the online network to select actions and the target network to evaluate them,
    reducing overestimation bias.
    """
    def _compute_td_target(self,
                           rewards: torch.Tensor,
                           next_obs: torch.Tensor,
                           dones: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_q_values = self.network(next_obs)
            next_actions = torch.argmax(next_q_values, dim=1)

            target_next_q_values = self.target_network(next_obs)
            target_max = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()

            return rewards + self.cfg.gamma * target_max * (1 - dones)