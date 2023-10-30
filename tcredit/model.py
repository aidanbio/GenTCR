import argparse
import unittest
from collections import OrderedDict
from functools import partial
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import copy
import torchmetrics.functional as FM
from adamp import AdamP, SGDP
# from pytorch_lightning import Trainer
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from pytorch_lightning.utilities.types import STEP_OUTPUT
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
from torch import nn as nn
from torch.functional import F
import logging.config
from torch.optim import Adam, SGD, AdamW
import pytorch_lightning as pl
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup, AutoModelForMaskedLM
)
from tcredit.common import TypeUtils, TorchUtils, CollectionUtils
from tcredit.data import BaseDatasetTest, EpitopeTargetDSDataLoaderTest, DatasetTestFixture
from tcredit.modeling_utils import PredictionHeadTransform, SimpleMLP

# Logger
logger = logging.getLogger('tcredit')

# CUDA
use_cuda = torch.cuda.is_available()
# torch.autograd.set_detect_anomaly(True)
# torch.set_float32_matmul_precision('medium')  # bfloat16
torch.set_autocast_enabled(True)

class BaseLitModel(pl.LightningModule):
    class PredictionHead(nn.Module):
        def __init__(self):
            super().__init__()

        @property
        def metrics(self):
            raise NotImplementedError()

        def init_scorer_map(self, metrics=['accuracy']):
            raise NotImplementedError()

        def loss(self, output, target):
            raise NotImplementedError()

        def score_map(self, output, target):
            raise NotImplementedError()

        def output_labels(self, output):
            raise NotImplementedError()

    class BasePredictionHead(PredictionHead):
        def __init__(self, metrics=['accuracy']):
            super().__init__()
            self.init_scorer_map(metrics)

        @property
        def metrics(self):
            return self.scorer_map.keys()

        def init_scorer_map(self, metrics=['accuracy']):
            self.scorer_map = OrderedDict({
                metric: self.available_metric_scorer_map[metric] for metric in metrics
            })

        @property
        def available_metric_scorer_map(self):
            return OrderedDict({
                'accuracy': self._accuracy,
                'precision': self._precision,
                'recall': self._recall,
                'f1': self._f1,
                'auroc': self._auroc
            })

        @classmethod
        def _accuracy(cls, y_pred, y_true, y_prob=None, n_classes=None):
            return FM.accuracy(y_pred, y_true)

        @classmethod
        def _precision(cls, y_pred, y_true, y_prob=None, n_classes=None):
            return FM.precision(y_pred, y_true,
                                task=('binary' if n_classes == 2 else 'multiclass'),
                                num_classes=n_classes)

        @classmethod
        def _recall(cls, y_pred, y_true, y_prob=None, n_classes=None):
            return FM.recall(y_pred, y_true,
                             task=('binary' if n_classes == 2 else 'multiclass'),
                             num_classes=n_classes)  # , average='macro')

        @classmethod
        def _f1(cls, y_pred, y_true, y_prob=None, n_classes=None):
            return FM.f1_score(y_pred, y_true,
                               task=('binary' if n_classes == 2 else 'multiclass'),
                               num_classes=n_classes)  # , average='macro')

        @classmethod
        def _auroc(cls, y_pred, y_true, y_prob=None, n_classes=None):
            return FM.auroc(y_prob[:, 1] if n_classes == 2 else y_prob, y_true,
                            task=('binary' if n_classes == 2 else 'multiclass'),
                            num_classes=n_classes)

    def __init__(self, train_config=None):
        super().__init__()

        self.train_config = train_config
        # Create model architecture: backbone is for extracting features and abstracted representations, head is for
        # predicting the final outputs and evaluating the predictions
        self.backbone = self._create_backbone()
        self.head = self._create_head()

        if self.train_config.get('init_weights', False):
            self.init_weights()

    def init_weights(self):
        self.apply(TorchUtils.init_module_weights)

    def clone(self):
        raise NotImplementedError()

    def configure_optimizers(self):
        config = OrderedDict()
        config['optimizer'] = self._create_optimizer()
        if 'lr_scheduler' in self.train_config:
            lr_scheduler = self._create_lr_scheduler(config['optimizer'], self.train_config['lr_scheduler'])
            config['lr_scheduler'] = {
                'scheduler': lr_scheduler
            }
        logger.info(f'Configured optimizer and lr_scheduler: {config}')
        return config

    def training_step(self, batch, batch_idx):
        return self._loss_metric_scores(batch, log_prefix='train')

    # def training_step_end(self, step_output) -> STEP_OUTPUT:
    #     if self.trainer.strategy and self.trainer.strategy.strategy_name == 'dp':
    #         return self._reduce_step_output(step_output)
    #     else:
    #         return super().training_step_end(step_output)

    def validation_step(self, batch, batch_idx):
        return self._loss_metric_scores(batch, log_prefix='val')


    # def validation_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
    #     if self.trainer.strategy and self.trainer.strategy.strategy_name == 'dp':
    #         return self._reduce_step_output(args[0])
    #     else:
    #         return super().validation_step_end(*args, **kwargs)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._loss_metric_scores(batch, log_prefix='test')

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     input, target = batch
    #     output = self(input)
    #     return output

    def freeze_backbone(self):
        self._freeze_backbone(True)

    def melt_backbone(self):
        self._freeze_backbone(False)

    def _create_backbone(self):
        raise NotImplementedError()

    def _create_head(self):
        params = copy.deepcopy(self.train_config['head'])
        return eval(params.pop('type'))(**params)

    def _freeze_backbone(self, on=True):
        for param in self.backbone.parameters():
            param.requires_grad = (not on)

    def _create_optimizer(self):
        params = copy.deepcopy(self.train_config['optimizer'])
        return eval(params.pop('type'))(self.parameters(), **params)

    def _create_lr_scheduler(self, optimizer, params):
        params = copy.deepcopy(params)
        tname = params.pop('type')
        if 'warmup' in tname:
            n_train_steps = self.trainer.estimated_stepping_batches
            steps_per_epoch = n_train_steps // self.trainer.max_epochs
            warmup_epochs = float(params.pop('warmup_epochs'))
            warmup_steps = int(warmup_epochs * steps_per_epoch)
            if tname == 'warmup_constant':
                return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
            elif tname == 'warmup_linear':
                return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                       num_training_steps=n_train_steps)
            elif tname == 'warmup_cosine':
                return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                       num_training_steps=n_train_steps)
            elif tname == 'warmup_poly':
                return get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                 num_training_steps=n_train_steps)
            elif tname == 'warmup_cosine_hard_restarts':
                return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                                          num_training_steps=n_train_steps)
        else:
            raise ValueError(f'Unknown lr_scheduler: {tname}')

    def _loss_metric_scores(self, batch, log_prefix=None):
        input, target = batch
        output = self(input)
        loss = self.head.loss(output, target)
        score_map = self.head.score_map(output, target)
        score_map['loss'] = loss
        self.log_dict({f'{log_prefix}.{k}': v for k, v in score_map.items()},
                      prog_bar=True, rank_zero_only=True, sync_dist=True)
        return {'loss': loss, 'score_map': score_map, 'output': output}

    # def forward(self, input):
    #     output = self.backbone(**input)
    #     return self.head(output)


class BERTLitModel(BaseLitModel):
    class LabelPredictionHead(BaseLitModel.BasePredictionHead):
        def __init__(self,
                     metrics=['accuracy'],
                     predictor=None,
                     use_first_token=True,
                     input_len=None,
                     hidden_size=768,
                     dropout=0.1,
                     n_labels=2,
                     bert_config=None):
            super().__init__(metrics=metrics)
            logger.debug(f'[LabelPredictionHead]: use_first_token: {use_first_token}, input_len: {input_len}')
            in_dim = bert_config.hidden_size if use_first_token else (input_len - 2) * bert_config.hidden_size
            self.use_first_token = use_first_token
            if predictor == 'msp':
                self.predictor = SimpleMLP(in_dim=in_dim,
                                           hid_dim=hidden_size,
                                           out_dim=n_labels,
                                           dropout=dropout)
            else:
                self.predictor = nn.Linear(in_dim, n_labels)

            self.loss_fn = nn.NLLLoss()

        def forward(self, sequence_output, pooled_output):
            # sequence_output.shape, pooled_output.shape: (batch_size, input_len, hidden_size), (batch_size, hidden_size)
            batch_size = sequence_output.shape[0]
            input = pooled_output if self.use_first_token else sequence_output[:, 1:-1].view(batch_size, -1)
            output = F.log_softmax(self.predictor(input), dim=-1)  # (batch_size, num_labels)
            return output

        def loss(self, output, target):
            # output.shape: (batch_size, num_labels)
            # target.shape: target_labels(batch_size), (target_tokens: batch_size x input_len)
            return self.loss_fn(output, target[0])

        def score_map(self, output, target):
            target_labels = target[0]
            output_labels, output_probs = self.output_labels(output)

            sm = OrderedDict()
            for metric, scorer in self.scorer_map.items():
                sm[metric] = scorer(target_labels, output_labels, output_probs, n_classes=output_probs.shape[1])
            return sm

        def output_labels(self, output):
            # output.shape: (batch_size, num_labels)
            labels = torch.argmax(output, dim=1)  # (batch_size,)
            probs = torch.exp(output)  # probs.shape: (batch_size, num_labels)
            return labels, probs

    class TokenPredictionHead(BaseLitModel.BasePredictionHead):
        def __init__(self, metrics=['accuracy'], bert_config=None):
            super().__init__(metrics=metrics)
            self.transform = PredictionHeadTransform(bert_config.hidden_size,
                                                     bert_config.hidden_act,
                                                     bert_config.layer_norm_eps)
            # The output weights are the same as the input embeddings, but there is
            # an output-only bias for each token.
            self.pad_token_id = bert_config.pad_token_id
            self.decoder = nn.Linear(bert_config.hidden_size, bert_config.vocab_size, bias=False)
            self.bias = nn.Parameter(data=torch.zeros(bert_config.vocab_size))
            self.loss_fn = nn.NLLLoss(ignore_index=self.pad_token_id)

        def forward(self, sequence_output, pooled_output):
            # sequence_output.shape, pooled_output.shape: (batch_size, input_len, hidden_size), (batch_size, hidden_size)
            output = self.transform(sequence_output)
            output = self.decoder(output) + self.bias
            output = F.log_softmax(output, dim=-1)  # (batch_size, input_len, vocab_size)
            return output

        def loss(self, output, target):
            # output.shape: batch_size x input_len x vocab_size
            # target.shape: target_labels(batch_size), (target_tokens: batch_size x input_len)
            target_labels, target_tokens = target
            loss = self.loss_fn(output.transpose(1, 2), target_tokens['input_ids'])
            return loss

        def score_map(self, output, target):
            # output.shape: (batch_size, input_len, vocab_size)
            # output_tokens.shape: batch_size x input_len, output_probs.shape: batch_size x input_len x vocab_size
            output_tokens, output_probs = self.output_labels(output)
            # target.shape: target_labels(batch_size), (target_tokens: batch_size x input_len)
            target_labels, target_tokens = target

            # Get token prediction N(=batch_size) scores for each token ignoring pads
            score_map = OrderedDict()
            for metric, scorer in self.scorer_map.items():
                scores = []
                for i in range(target_tokens.shape[1]):  # for each token
                    cur_target_tokens = target_tokens[:, i]
                    select = torch.where(cur_target_tokens != self.pad_token_id)[
                        0]  # select not PAD tokens from batch items
                    if (select is not None) and (select.shape[0] > 0):
                        cur_target_tokens = cur_target_tokens[select]
                        cur_output_tokens = output_tokens[select, i]
                        cur_output_probs = output_probs[select, i]
                        score = scorer(cur_target_tokens, cur_output_tokens, cur_output_probs,
                                       n_classes=cur_output_probs.shape[1])
                        scores.append(score)

                score_map[metric] = torch.mean(torch.stack(scores))

            return score_map

        def output_labels(self, output):
            # output.shape: (batch_size x, input_len x vocab_size)
            output_tokens = torch.argmax(output, dim=2)  # batch_size x input_len
            output_probs = torch.exp(output)  # batch_size x input_len x vocab_size
            return output_tokens, output_probs

    class BalancedPredictionHead(BaseLitModel.PredictionHead):
        def __init__(self,
                     metrics=['accuracy'],
                     predictor=None,
                     use_first_token=True,
                     input_len=None,
                     hidden_size=768,
                     dropout=0.1,
                     n_labels=2,
                     bert_config=None,
                     lamda=None):
            super().__init__()
            self.token_head = BERTLitModel.TokenPredictionHead(metrics=metrics, bert_config=bert_config)
            self.label_head = BERTLitModel.LabelPredictionHead(metrics=metrics,
                                                               predictor=predictor,
                                                               use_first_token=use_first_token,
                                                               input_len=input_len,
                                                               hidden_size=hidden_size,
                                                               dropout=dropout,
                                                               n_labels=n_labels,
                                                               bert_config=bert_config)
            self.lamda = lamda

        @property
        def metrics(self):
            return self.token_head.metrics

        def init_scorer_map(self, metrics=['accuracy']):
            self.token_head.init_scorer_map(metrics)
            self.label_head.init_scorer_map(metrics)

        def forward(self, sequence_output, pooled_output):
            # sequence_output.shape, pooled_output.shape: (batch_size, input_len, hidden_size), (batch_size, hidden_size)
            return (self.token_head(sequence_output, pooled_output),
                    self.label_head(sequence_output, pooled_output))

        def loss(self, output, target):
            if self.lamda == 1:
                return self.token_head.loss(output[0], target)
            elif self.lamda == 0:
                return self.label_head.loss(output[1], target)
            else:
                token_loss = self.token_head.loss(output[0], target)
                label_loss = self.label_head.loss(output[1], target)
                return (self.lamda * token_loss) + ((1 - self.lamda) * label_loss)

        def score_map(self, output, target):
            if self.lamda == 1:
                return self.token_head.score_map(output[0], target)
            elif self.lamda == 0:
                return self.label_head.score_map(output[1], target)
            else:
                token_sm = self.token_head.score_map(output[0], target)
                label_sm = self.label_head.score_map(output[1], target)
                sm = OrderedDict()
                for metric in self.metrics:
                    token_score = token_sm[metric]
                    label_score = label_sm[metric]
                    sm[metric] = (self.lamda * token_score) + ((1 - self.lamda) * label_score)
                return sm

        def output_labels(self, output):
            # if self.lamda == 1:
            #     self.token_head.output_labels(output[0])
            # elif self.lamda == 0:
            #     return self.label_head.output_labels(output[1])
            # else:
            #     return (self.token_head.output_labels(output[0]),
            #             self.label_head.output_labels(output[1]))
            return (self.token_head.output_labels(output[0]),
                    self.label_head.output_labels(output[1]))

    def __init__(self, train_config=None):
        super().__init__(train_config=train_config)
        # if self.bert_config.joint_io_weights and self.token_head is not None:
        #     self.bert._tie_or_clone_weights(self.token_head.decoder, self.bert.embeddings.word_embeddings)

    @property
    def bert(self):
        return self.backbone

    @property
    def bert_config(self):
        return self.bert.config

    @property
    def token_head(self):
        if isinstance(self.head, BERTLitModel.TokenPredictionHead):
            return self.head
        elif isinstance(self.head, BERTLitModel.BalancedPredictionHead):
            return self.head.token_head
        else:
            return None

    @property
    def label_head(self):
        if isinstance(self.head, BERTLitModel.LabelPredictionHead):
            return self.head
        elif isinstance(self.head, BERTLitModel.BalancedPredictionHead):
            return self.head.label_head
        else:
            return None

    # Checkpoint hooks
    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    #     CollectionUtils.update_dict_key_prefix(sd, source_prefix='bert.', target_prefix='backbone.')
    #
    # def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    #     CollectionUtils.update_dict_key_prefix(sd, source_prefix='backbone.', target_prefix='bert.', )

    def clone(self):
        the = BERTLitModel(train_config=copy.deepcopy(self.train_config))
        the.load_state_dict(self.state_dict())
        the.to(self.device)
        return the

    # For freeze and melt bert encoders
    def freeze_bert(self):
        self.freeze_backbone()

    def melt_bert(self):
        self.melt_backbone()

    # def train_bert_encoders(self, layer_range=(-2, None)):
    #     self.freeze_bert()
    #
    #     # Melt bert embeddings
    #     for param in self.bert.embeddings.parameters():
    #         param.requires_grad = True
    #
    #     # Melt target encoder layers and pooler
    #     for layer in self.bert.encoder.layer[layer_range[0]:layer_range[1]]:
    #         for param in layer.parameters():
    #             param.requires_grad = True
    #
    #     for param in self.bert.pooler.parameters():
    #         param.requires_grad = True

    def forward(self, input):
        # bert_out: # sequence_output, pooled_output, (hidden_states), (attentions)
        bert_out = self.bert(**input)

        # sequence_out.shape: (batch_size, input_len, hidden_size), pooled_out.shape: (batch_size, hidden_size)
        sequence_out, pooled_out = bert_out[:2]
        output = self.head(sequence_out, pooled_out)
        return output + bert_out[2:] if isinstance(output, tuple) else (output,) + bert_out[2:]

    # def load_bert(self, ckpt_path, bert_prefix='bert.'):
    #     sd = TorchUtils.load_state_dict(ckpt_path, prefix=bert_prefix)
    #     assert sd.keys() == self.bert.state_dict().keys(), 'There are mismatched bert key(s)'
    #     self.bert.load_state_dict(sd)

    def _create_backbone(self):
        bert_name_or_path = self.train_config.get('bert_name_or_path', 'bert-base-uncased')
        bert = AutoModelForMaskedLM.from_pretrained(bert_name_or_path)
        return bert

    def _create_head(self):
        params = copy.deepcopy(self.train_config['head'])
        params['bert_config'] = self.bert_config
        return eval(params.pop('type'))(**params)

    # @classmethod
    # def from_pretrained_bert(cls, train_config, bert_path, ckpt_path=None, bert_prefix='bert.'):
    #     bert_config = ProteinConfig.from_pretrained(bert_path)
    #     model = BERTLitModel(train_config=train_config, bert_config=bert_config)
    #     model.load_bert(ckpt_path=ckpt_path if ckpt_path else f'{bert_path}/pytorch_model.bin', bert_prefix=bert_prefix)
    #     return model


class BaseLitModelTest(BaseDatasetTest):
    class TestModel(BaseLitModel):
        class PredictionHead(BaseLitModel.BasePredictionHead):
            def __init__(self, metrics=['accuracy']):
                super().__init__(metrics=metrics)
                self.predictor = nn.Softmax(dim=1)
                self.loss_fn = nn.CrossEntropyLoss()

            def forward(self, input):
                return self.predictor(input)

            def loss(self, output, target):
                return self.loss_fn(output, target)

            def score_map(self, output, target):
                y_true = target
                y_pred, y_prob = self.output_labels(output)
                # logger.info(f'PredictionEvaluator.score_map: y_true: {y_true}, y_pred: {y_pred}, y_prob: {y_prob}')
                score_map = OrderedDict()
                for metric, scorer in self.scorer_map.items():
                    score_map[metric] = scorer(y_true, y_pred, y_prob, n_classes=y_prob.shape[1])

                logger.debug('score_map: %s' % score_map)
                return score_map

            def output_labels(self, output):
                labels = torch.argmax(output, dim=1)
                return labels, output

        def __init__(self, train_config):
            super().__init__(train_config=train_config)

        def forward(self, x):
            out = x
            for layer in self.backbone:
                out = layer(out)
            return self.head(out)

        def clone(self):
            c = BaseLitModelTest.TestModel(train_config=copy.deepcopy(self.train_config))
            sd = self.state_dict()
            c.load_state_dict(sd)
            c.to(self.device)
            return c

        def _create_backbone(self):
            return nn.ModuleList([
                nn.Linear(4, 100),
                nn.Linear(100, 100),
                nn.Linear(100, 3)
            ])

        # def _create_head(self, kwargs):
        #     return super()._create_head()

    def setUp(self):
        super().setUp()

        logger.setLevel(logging.INFO)

        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        self.device_count = torch.cuda.device_count() if use_cuda else 0
        self.train_config = self.create_train_config()
        self.model = self.create_model()
        self.batch_size = 32
        self.val_size = 0.1
        self.train_data_loader, self.val_data_loader = self.create_train_val_data_loader()
        self.test_data_loader = [self.val_data_loader, self.val_data_loader]

    def create_train_val_data_loader(self):
        return DatasetTestFixture.create_data_loader('iris',
                                                     batch_size=self.batch_size,
                                                     n_workers=1,
                                                     val_size=self.val_size)
    @property
    def train_ds(self):
        return self.train_data_loader.dataset

    @property
    def val_ds(self):
        return self.val_data_loader.dataset

    @property
    def first_batch(self):
        batch = next(iter(self.train_data_loader))
        return TorchUtils.collection_to(batch, self.device)

    def create_train_config(self):
        return {
            "head": {
                "type": "BaseLitModelTest.TestModel.PredictionHead",
                "metrics": ["f1", "auroc"]
            },
            "optimizer": {
                "type": "torch.optim.AdamW",
                "lr": 0.001
            },
            "lr_scheduler": {
                "type": "warmup_poly",
                "warmup_epochs": 0.0125,
                "n_train_steps": 10
            },
        }

    def create_model(self):
        model = self.TestModel(train_config=self.train_config)
        model.to(self.device)
        return model

    def assert_model_output(self, output, batch_size):
        expected = (batch_size, 3)
        self.assertEqual(expected, output.shape)

    def assert_head_output_labels(self, output_labels=None, batch_size=None):
        labels, probs = output_labels
        self.assertEqual((batch_size,), labels.shape)
        self.assertEqual((batch_size, 3), probs.shape)
        self.assertAlmostEqual(batch_size, torch.sum(probs).item(), delta=3)

    def state_dict_equal(self, st1, st2):
        return TorchUtils.equal_state_dict(st1, st2)

    def module_weights_equal(self, m1, m2):
        return self.state_dict_equal(m1.state_dict(), m2.state_dict())

    def get_trainer_args(self):
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        args = argparse.Namespace()
        args.accelerator = ('gpu' if n_gpus > 0 else 'cpu')
        args.devices = (n_gpus if n_gpus > 0 else None)
        args.strategy = ('ddp_spawn' if n_gpus > 0 else None)
        args.num_nodes = None
        args.precision = 16
        print(f'>>>trainer_args:{args}')
        return args

    # Tests
    def test_clone(self):
        c = self.model.clone()
        c.to(self.device)
        self.assertIsNotNone(c)
        self.assertNotEqual(id(c), id(self.model))
        self.assertTrue(self.module_weights_equal(c, self.model))

    def test_forward(self):
        input, target = self.first_batch
        output = self.model(input)
        self.assert_model_output(output, batch_size=self.batch_size)

    def test_prediction_head(self):
        input, target = self.first_batch
        head = self.model.head
        output = self.model(input)
        self.assert_model_output(output, batch_size=self.batch_size)
        self.assert_head_output_labels(head.output_labels(output), batch_size=self.batch_size)

        loss = head.loss(output, target)
        self.assertIsNotNone(loss)
        self.assertTrue(TypeUtils.is_numeric_value(loss.item()))

        sm = head.score_map(output, target)
        self.assertSetEqual(set(head.metrics), set(sm.keys()))
        self.assertTrue(all(map(lambda x: x >= 0 and x <= 1, sm.values())))

    def test_change_metrics_of_head(self):
        input, target = self.first_batch
        head = self.model.head
        output = self.model(input)

        self.assert_model_output(output, batch_size=self.batch_size)
        self.assert_head_output_labels(head.output_labels(output), batch_size=self.batch_size)

        sm = head.score_map(output, target)
        self.assertSetEqual(set(head.metrics), set(sm.keys()))
        self.assertTrue(all(map(lambda x: x >= 0 and x <= 1, sm.values())))

        old_metrics = head.metrics
        metrics = ['f1', 'recall', 'precision']
        head.init_scorer_map(metrics=metrics)

        sm = head.score_map(output, target)
        self.assertSetEqual(set(metrics), set(head.metrics))
        self.assertSetEqual(set(head.metrics), set(sm.keys()))
        self.assertTrue(all(map(lambda x: x >= 0 and x <= 1, sm.values())))

    def test_fit(self):
        logger.info('>>>test_fit')
        old_model = self.model.clone()
        old_model.to(self.device)
        self.assertTrue(self.state_dict_equal(old_model.state_dict(), self.model.state_dict()))

        args = self.get_trainer_args()
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             strategy=args.strategy,
                             num_nodes=args.num_nodes,
                             precision=args.precision,
                             max_epochs=10,
                             enable_checkpointing=False)

        trainer.fit(self.model, train_dataloaders=self.train_data_loader, val_dataloaders=self.val_data_loader)

        # old_model.to(torch.device('cpu'))
        self.assertFalse(self.state_dict_equal(old_model.state_dict(), self.model.state_dict()))

    def test_test(self):
        head = self.model.head
        args = self.get_trainer_args()
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             strategy=args.strategy,
                             num_nodes=args.num_nodes,
                             precision=args.precision,
                             max_epochs=1,
                             enable_checkpointing=False)
        trainer.fit(self.model, train_dataloaders=self.train_data_loader, val_dataloaders=self.val_data_loader)
        outputs = trainer.test(self.model, dataloaders=self.test_data_loader)
        result_map = outputs[0]
        self.assertEqual(len(head.metrics) + 1, len(result_map))
        for metric in head.metrics:
            self.assertEqual(len(list(filter(lambda key: metric in key, result_map.keys()))), 1)

    def test_test_changed_metrics_of_head(self):
        head = self.model.head
        args = self.get_trainer_args()
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             strategy=args.strategy,
                             num_nodes=args.num_nodes,
                             precision=args.precision,
                             max_epochs=1,
                             enable_checkpointing=False)
        trainer.fit(self.model, train_dataloaders=self.train_data_loader, val_dataloaders=self.val_data_loader)

        old_metrics = head.metrics
        metrics = ['f1', 'recall', 'precision']
        head.init_scorer_map(metrics)
        self.assertNotEqual(old_metrics, head.metrics)

        outputs = trainer.test(self.model, dataloaders=self.test_data_loader)
        for output_map in outputs:
            print(output_map)
            self.assertEqual(len(head.metrics) + 1, len(output_map))
            for metric in head.metrics:
                self.assertEqual(len(list(filter(lambda key: metric in key, output_map.keys()))), 1)

    # def test_predict(self):
    #     args = self.get_trainer_args()
    #     trainer = pl.Trainer(accelerator=args.accelerator,
    #                          devices=args.devices,
    #                          strategy=args.strategy,
    #                          num_nodes=args.num_nodes,
    #                          precision=args.precision,
    #                          max_epochs=1,
    #                          enable_checkpointing=False)
    #     trainer.fit(self.model, train_dataloaders=self.train_data_loader, val_dataloaders=self.val_data_loader)
    #
    #     data_loader = self.train_data_loader
    #     outputs = trainer.predict(self.model, dataloaders=data_loader)
    #
    #     self.assertEqual(len(data_loader), len(outputs))
    #     head = self.model.head
    #     for i, (input, target) in enumerate(data_loader):
    #         if isinstance(input, list) or isinstance(input, tuple):
    #             batch_size = input[0].shape[0]
    #         else:
    #             batch_size = input.shape[0]
    #         self.assert_model_output(outputs[i], batch_size=batch_size)
    #         self.assert_head_output_labels(head.output_labels(outputs[i]), batch_size=batch_size)


class BERTLitModelTest(BaseLitModelTest):
    bert_path = '../output/peft_esm2_t33_650M_UR50D'

    def setUp(self):
        super().setUp()

    def create_train_val_data_loader(self):
        return DatasetTestFixture.create_data_loader('epitope_target',
                                                     batch_size=self.batch_size,
                                                     n_workers=1,
                                                     val_size=self.val_size)

    @property
    def collator(self):
        return self.train_data_loader.collate_fn

    @property
    def tokenizer(self):
        return self.train_data_loader.collate_fn.tokenizer

    def create_model(self):
        model = BERTLitModel(train_config=self.train_config)
        model.to(self.device)
        return model

    def create_train_config(self):
        return {
            "bert_name_or_path": self.bert_path,
            "head": {
                "type": "BERTLitModel.BalancedPredictionHead",
                "metrics": ["f1", "auroc"],
                "use_first_token": False,
                "input_len": 48,
                "n_labels": 2,
                "lamda": 0.5
            },
            "optimizer": {
                "type": "torch.optim.AdamW",
                "lr": 0.001,
            },
            "lr_scheduler": {
                "type": "warmup_poly",
                "warmup_epochs": 0.0125
            },
        }

    def assert_model_output(self, output, batch_size):
        token_out, label_out, hidden_states, attentions = output
        input_len = self.train_ds.max_len
        vocab_size = self.train_ds.vocab_size
        self.assertEqual((batch_size, input_len, vocab_size), token_out.shape)
        self.assertEqual((batch_size, 2), label_out.shape)

        n_encoders = self.model.bert_config.num_hidden_layers
        self.assertEqual(n_encoders, len(hidden_states) - 1)  # First hstate is for embedding
        self.assertEqual(n_encoders, len(attentions))

        hidden_size = self.model.bert_config.hidden_size
        n_heads = self.model.bert_config.num_attention_heads
        expected_hstate_shape = (batch_size, input_len, hidden_size)
        expected_attn_shape = (batch_size, n_heads, input_len, input_len)

        self.assertTrue(all([expected_hstate_shape == hstate.shape for hstate in hidden_states[1:]]))
        self.assertTrue(all([expected_attn_shape == attn.shape for attn in attentions]))

    def assert_head_output_labels(self, output_labels=None, batch_size=None):
        token_out, label_out = output_labels
        self._assert_token_head_output_labels(token_out, batch_size=batch_size)
        self._assert_label_head_output_labels(label_out, batch_size=batch_size)

    def _assert_token_head_output_labels(self, output_labels, batch_size):
        labels, probs = output_labels
        input_len = self.train_ds.max_len
        vocab_size = self.train_ds.vocab_size
        self.assertEqual((batch_size, input_len), labels.shape)
        self.assertEqual((batch_size, input_len, vocab_size), probs.shape)
        for i in range(input_len):
            self.assertAlmostEqual(batch_size, torch.sum(probs[:, i]).item(), delta=3)

    def _assert_label_head_output_labels(self, output_labels, batch_size):
        labels, probs = output_labels
        self.assertEqual((batch_size,), labels.shape)
        self.assertEqual((batch_size, 2), probs.shape)
        self.assertAlmostEqual(batch_size, torch.sum(probs).item(), delta=3)

    @property
    def first_batch(self):
        batch = next(iter(self.train_data_loader))
        input, target = batch
        input = input.to(self.device)
        target = (target[0].to(self.device), target[1].to(self.device))
        return input, target

    def test_prediction_head(self):
        config = self.train_config['head']
        self.assertTrue(isinstance(self.model.head, BERTLitModel.BalancedPredictionHead))
        self.assertTrue(isinstance(self.model.head.token_head, BERTLitModel.TokenPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head, BERTLitModel.LabelPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head.predictor, nn.Linear))
        expected_weight_shape = (config['n_labels'], (config['input_len'] - 2) * self.model.bert_config.hidden_size)
        self.assertEqual(expected_weight_shape, self.model.head.label_head.predictor.weight.shape)
        super().test_prediction_head()

        # Only TokenPredictionHead
        self.train_config['head'] = {
            "type": "BERTLitModel.BalancedPredictionHead",
            "metrics": ["f1", "auroc"],
            "use_first_token": False,
            "input_len": 48,
            "lamda": 1
        }
        self.model = self.create_model()
        self.assertTrue(isinstance(self.model.head, BERTLitModel.BalancedPredictionHead))
        self.assertTrue(isinstance(self.model.head.token_head, BERTLitModel.TokenPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head, BERTLitModel.LabelPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head.predictor, nn.Linear))
        self.assertEqual((2, 46 * self.model.bert_config.hidden_size),
                         self.model.head.label_head.predictor.weight.shape)
        super().test_prediction_head()

        # Only LabelPredictionHead
        self.train_config['head'] = {
            "type": "BERTLitModel.BalancedPredictionHead",
            "metrics": ["f1", "auroc"],
            "predictor": "msp",
            "use_first_token": False,
            "input_len": 48,
            "dropout": 0.2,
            "lamda": 0
        }
        self.model = self.create_model()
        self.assertTrue(isinstance(self.model.head, BERTLitModel.BalancedPredictionHead))
        self.assertTrue(isinstance(self.model.head.token_head, BERTLitModel.TokenPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head, BERTLitModel.LabelPredictionHead))
        self.assertTrue(isinstance(self.model.head.label_head.predictor, SimpleMLP))
        super().test_prediction_head()

    def test_fit(self):
        print(self.model)
        super().test_fit()

        self.train_config['head'] = {
            "type": "BERTLitModel.BalancedPredictionHead",
            "metrics": ["f1", "auroc"],
            "use_first_token": False,
            "input_len": 48,
            "lamda": 0.3
        }
        self.model = self.create_model()
        print(self.model)
        super().test_fit()

    def _test_load_bert(self, ckpt_path):
        old_model = self.model.clone()
        old_model.to(self.device)
        self.assertEqual(old_model.bert.state_dict().keys(), self.model.bert.state_dict().keys())
        self.assertTrue(self.module_weights_equal(self.model.bert, old_model.bert))
        self.model.load_bert(ckpt_path=ckpt_path)
        self.assertFalse(self.module_weights_equal(self.model.bert, old_model.bert))
        other = self.create_model()
        other.load_bert(ckpt_path=ckpt_path)
        self.assertEqual(other.bert.state_dict().keys(), self.model.bert.state_dict().keys())
        self.assertTrue(self.module_weights_equal(self.model.bert, other.bert))

    def test_load_bert(self):
        self._test_load_bert(ckpt_path=f'{self.bert_path}/pytorch_model.bin')
        self._test_load_bert(ckpt_path='../output/exp1/V3/pretrain.2.3/pretrain.2.3.best_model.ckpt')

    def test_load_from_ckpt_no_strict(self):
        ckpt_path = '../output/exp1/V3/pretrain.3.3/pretrain.3.3.best_model.ckpt'
        expected_sd = torch.load(ckpt_path) if use_cuda else torch.load(ckpt_path, map_location=torch.device('cpu'))
        if 'state_dict' in expected_sd:
            expected_sd = expected_sd['state_dict']
        bert_keys = list(filter(lambda k: k.startswith('bert.'), expected_sd.keys()))
        expected_bert_sd = OrderedDict({k: expected_sd[k] for k in bert_keys})

        model = BERTLitModel.load_from_checkpoint(ckpt_path,
                                                  bert_config=self.model.bert_config,
                                                  train_config=self.train_config,
                                                  strict=False)
        model.to(self.device)
        sd = model.state_dict()
        CollectionUtils.update_dict_key_prefix(sd, 'backbone.', 'bert.')
        self.assertTrue(self.state_dict_equal(expected_bert_sd, OrderedDict({k: sd[k] for k in bert_keys})))

    def test_from_pretrained_bert(self):
        model = ProteinBertModel.from_pretrained(self.bert_path)
        other = BERTLitModel.from_pretrained_bert(self.train_config, self.bert_path)
        self.assertEqual(model.state_dict().keys(), other.bert.state_dict().keys())
        self.assertTrue(self.module_weights_equal(model, other.bert))

        ckpt_path = '../output/exp1/V3/pretrain.2.3/pretrain.2.3.best_model.ckpt'
        model = BERTLitModel.load_from_checkpoint(ckpt_path,
                                                  bert_config=self.model.bert_config,
                                                  train_config=self.train_config,
                                                  strict=False)
        other = BERTLitModel.from_pretrained_bert(self.train_config, ckpt_path=ckpt_path, bert_path=self.bert_path)
        self.assertEqual(model.bert.state_dict().keys(), other.bert.state_dict().keys())
        self.assertTrue(self.module_weights_equal(model.bert, other.bert))

    def test_train_bert_encoders(self):
        layer_range = [-4, None]

        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

        self.model.train_bert_encoders(layer_range=layer_range)

        for param in self.model.bert.embeddings.parameters():
            self.assertTrue(param.requires_grad)

        for layer in self.model.bert.encoder.layer[0:-4]:
            for param in layer.parameters():
                self.assertFalse(param.requires_grad)

        for layer in self.model.bert.encoder.layer[-4:None]:
            for param in layer.parameters():
                self.assertTrue(param.requires_grad)

        for param in self.model.bert.pooler.parameters():
            self.assertTrue(param.requires_grad)

        for param in self.model.head.parameters():
            self.assertTrue(param.requires_grad)

        old_model = self.model.clone()

        args = self.get_trainer_args()
        trainer = Trainer(accelerator=args.accelerator,
                          devices=args.devices,
                          strategy=args.strategy,
                          num_nodes=args.num_nodes,
                          amp_backend=args.amp_backend,
                          precision=args.precision,
                          max_epochs=1,
                          enable_checkpointing=False)

        trainer.fit(self.model, train_dataloaders=self.create_train_data_loaer(),
                    val_dataloaders=self.create_val_data_loaer())

        self.assertFalse(self.module_weights_equal(old_model.bert.embeddings, self.model.bert.embeddings))
        for old_layer, layer in zip(old_model.bert.encoder.layer[0:-4], self.model.bert.encoder.layer[0:-4]):
            self.assertTrue(self.module_weights_equal(old_layer, layer))

        for old_layer, layer in zip(old_model.bert.encoder.layer[-4:None], self.model.bert.encoder.layer[-4:None]):
            self.assertFalse(self.module_weights_equal(old_layer, layer))

        self.assertFalse(self.module_weights_equal(old_model.bert.pooler, self.model.bert.pooler))
        self.assertFalse(self.module_weights_equal(old_model.head, self.model.head))

        self.model.melt_bert()

        for param in self.model.bert.parameters():
            self.assertTrue(param.requires_grad)

        trainer = Trainer(accelerator=args.accelerator,
                          devices=args.devices,
                          strategy=args.strategy,
                          num_nodes=args.num_nodes,
                          amp_backend=args.amp_backend,
                          precision=args.precision,
                          max_epochs=1,
                          enable_checkpointing=False)
        trainer.fit(self.model, train_dataloaders=self.create_train_data_loaer(),
                    val_dataloaders=self.create_val_data_loaer())

        self.assertFalse(self.module_weights_equal(old_model.bert.embeddings, self.model.bert.embeddings))
        for old_layer, layer in zip(old_model.bert.encoder.layer, self.model.bert.encoder.layer):
            self.assertFalse(self.module_weights_equal(old_layer, layer))

        self.assertFalse(self.module_weights_equal(old_model.bert.pooler, self.model.bert.pooler))
        self.assertFalse(self.module_weights_equal(old_model.head, self.model.head))

    def test_freeze_melt_bert(self):
        # TODO: If self.model.bert_config['joint_io_weights'] is True, the parameter of self.model.bert.word_embeddings
        #  are tied with the parameter of self.model.token_head, there assert the 'joint_io_weights' is False
        self.assertFalse(self.model.bert_config.joint_io_weights)

        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

        old_model = self.model.clone()
        self.model.freeze_bert()
        for param in self.model.bert.parameters():
            self.assertFalse(param.requires_grad)
        for param in self.model.head.parameters():
            self.assertTrue(param.requires_grad)
        args = self.get_trainer_args()
        trainer = Trainer(accelerator=args.accelerator,
                          devices=args.devices,
                          strategy=args.strategy,
                          num_nodes=args.num_nodes,
                          amp_backend=args.amp_backend,
                          precision=args.precision,
                          max_epochs=1,
                          enable_checkpointing=False)
        trainer.fit(self.model, train_dataloaders=self.create_train_data_loaer(),
                    val_dataloaders=self.create_val_data_loaer())
        self.assertTrue(self.module_weights_equal(old_model.bert, self.model.bert))
        self.assertFalse(self.module_weights_equal(old_model.head, self.model.head))

        old_model = self.model.clone()
        self.model.melt_bert()
        args = self.get_trainer_args()
        trainer = Trainer(accelerator=args.accelerator,
                          devices=args.devices,
                          strategy=args.strategy,
                          num_nodes=args.num_nodes,
                          amp_backend=args.amp_backend,
                          precision=args.precision,
                          max_epochs=1,
                          enable_checkpointing=False)
        trainer.fit(self.model, train_dataloaders=self.create_train_data_loaer(),
                    val_dataloaders=self.create_val_data_loaer())
        self.assertFalse(self.module_weights_equal(old_model.bert, self.model.bert))
        self.assertFalse(self.module_weights_equal(old_model.head, self.model.head))


if __name__ == '__main__':
    unittest.main()
