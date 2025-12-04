from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from rlp.agent.dqn import DQNAgent


class DDQNAgent(DQNAgent):
    """
    Double DQN Agent.
    
    Uses the online network to select actions and the target network to evaluate them,
    reducing overestimation bias.
    """

    def update(self, batch: Any) -> Dict[str, float]:
        """
        Update agent parameters based on a batch of data using Double Q-Learning.
        """
        self.steps += 1
        
        obs = batch.observations.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.flatten().to(self.device)
        next_obs = batch.next_observations.to(self.device)
        dones = batch.dones.flatten().to(self.device)

        with torch.no_grad():
            # Double DQN: selection with online, evaluation with target
            next_q_values = self.network(next_obs)
            next_actions = torch.argmax(next_q_values, dim=1)
            
            target_next_q_values = self.target_network(next_obs)
            target_max = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
            
            td_target = rewards + self.gamma * target_max * (1 - dones)

        old_val = self.network(obs).gather(1, actions.unsqueeze(1)).squeeze()
        loss = F.mse_loss(td_target, old_val)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        if self.steps % self.target_network_frequency == 0:
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

        return {
            "loss": loss.item(),
            "q_values": old_val.mean().item(),
        }
