import pytest
from unittest.mock import MagicMock
import torch.nn as nn
from rlp.agent.dqn import DQNAgent, DQNConfig
from rlp.components.network import QNetwork
from rlp.pruning.base import PrunerProtocol

class MockPruner(PrunerProtocol):
    def __init__(self):
        self.pruned_modules = []
        
    def prune(self, model: nn.Module, step: int) -> float | None:
        self.pruned_modules.append(model)
        return 0.5

@pytest.fixture
def agent_setup():
    encoder = nn.Linear(10, 10)
    head = nn.Linear(10, 2)
    network = QNetwork(encoder, head)
    optimizer = MagicMock()
    config = DQNConfig(
        num_actions=2,
        gamma=0.99,
        target_network_frequency=100
    )
    return network, optimizer, config

def test_init_none_pruner(agent_setup):
    network, optimizer, config = agent_setup
    agent = DQNAgent(network, optimizer, None, config, "cpu")
    assert agent.pruner is None
    assert agent.prune(1) is None

def test_init_single_pruner(agent_setup):
    network, optimizer, config = agent_setup
    pruner = MockPruner()
    agent = DQNAgent(network, optimizer, pruner, config, "cpu")
    
    assert agent.pruner == pruner
    sparsity = agent.prune(1)
    
    # Verify prune called on full network
    assert len(pruner.pruned_modules) == 1
    assert pruner.pruned_modules[0] == network
    assert sparsity == 0.5

def test_init_dict_pruner(agent_setup):
    network, optimizer, config = agent_setup
    encoder_pruner = MockPruner()
    head_pruner = MockPruner()
    pruners = {"encoder": encoder_pruner, "head": head_pruner}
    
    agent = DQNAgent(network, optimizer, pruners, config, "cpu")
    
    assert agent.pruner == pruners
    
    agent.prune(1)
    
    # Check encoder pruner called on encoder
    assert len(encoder_pruner.pruned_modules) == 1
    assert encoder_pruner.pruned_modules[0] == network.encoder
    
    # Check head pruner called on head
    assert len(head_pruner.pruned_modules) == 1
    assert head_pruner.pruned_modules[0] == network.head

def test_init_dict_pruner_invalid_key(agent_setup):
    network, optimizer, config = agent_setup
    pruners = {"invalid": MockPruner()}
    
    with pytest.raises(ValueError, match="Invalid pruner key"):
        DQNAgent(network, optimizer, pruners, config, "cpu")

def test_prune_partial_dict(agent_setup):
    network, optimizer, config = agent_setup
    encoder_pruner = MockPruner()
    pruners = {"encoder": encoder_pruner}
    
    agent = DQNAgent(network, optimizer, pruners, config, "cpu")
    agent.prune(1)
    
    assert len(encoder_pruner.pruned_modules) == 1
    assert encoder_pruner.pruned_modules[0] == network.encoder
