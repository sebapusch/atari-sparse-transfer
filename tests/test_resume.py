
import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf, DictConfig
from rlp.core.builder import Builder
from rlp.core.logger import WandbLogger
from rlp.core.trainer import Trainer, TrainingConfig, TrainingContext
from rlp.core.checkpointer import Checkpointer

class TestResumeLogic:
    
    @patch("rlp.core.builder.wandb")
    def test_fetch_remote_config(self, mock_wandb):
        # Setup mock
        mock_api = MagicMock()
        mock_run = MagicMock()
        mock_run.config = {'foo': 'bar', 'train': {'resume': False}} # Remote config usually has resume=False initially
        mock_api.run.return_value = mock_run
        mock_wandb.Api.return_value = mock_api
        
        # Test
        config = Builder.fetch_remote_config("test_run_id")
        
        # Verify
        assert config.foo == 'bar'
        mock_api.run.assert_called_with("test_run_id")

    def test_wandb_logger_buffering(self):
        mock_run = MagicMock()
        logger = WandbLogger(run=mock_run)
        
        # Log metrics - should be buffered
        logger.log_metrics({'loss': 0.1}, step=1)
        logger.log_metrics({'accuracy': 0.9}, step=2)
        
        mock_run.log.assert_not_called()
        assert len(logger._metric_buffer) == 2
        
        # Commit - should flush
        logger.commit()
        
        assert mock_run.log.call_count == 2
        mock_run.log.assert_any_call({'loss': 0.1}, step=1)
        mock_run.log.assert_any_call({'accuracy': 0.9}, step=2)
        assert len(logger._metric_buffer) == 0

    @patch("rlp.core.trainer.Checkpointer")
    def test_trainer_init_start_step(self, MockCheckpointer):
        mock_ctx = MagicMock()
        mock_cfg = MagicMock()
        checkpointer = MockCheckpointer()
        
        trainer = Trainer(mock_ctx, mock_cfg, checkpointer, start_step=100)
        assert trainer.start_step == 100
        
    def test_trainer_save_commits_logs(self):
        mock_ctx = MagicMock()
        mock_logger = MagicMock()
        mock_ctx.logger = mock_logger
        
        mock_cfg = MagicMock()
        mock_checkpointer = MagicMock()
        
        trainer = Trainer(mock_ctx, mock_cfg, mock_checkpointer, start_step=0)
        
        trainer._save(step=10, epsilon=0.1)
        
        mock_logger.commit.assert_called_once()
