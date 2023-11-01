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


class BertMLMLitModel(pl.LightningModule):
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
        super().__init__()
        self.mlm = AutoModelForMaskedLM.from_pretrained(train_config['mlm_name_or_path'])
        self.train_config = train_config
        self.loss_fn = nn.NLLLoss(ignore_index=self.mlm_config.pad_token_id)

    @property
    def mlm_config(self):
        return self.mlm.config

    @property
    def mlm_head(self):
        return self.mlm.lm_head

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

    def clone(self):
        the = BertMLMLitModel(train_config=copy.deepcopy(self.train_config))
        the.load_state_dict(self.state_dict())
        the.to(self.device)
        return the

    def training_step(self, batch, batch_idx):
        return self._loss_log(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self._loss_metric_scores(batch, log_prefix='val')

    def forward(self, input):
        # mlm_out: MaskedLMOutput, (loss), logits, (hidden_states), (attentions)
        # logits.shape: (batch_size, input_len, vocab_size)
        mlm_out = self.mlm(**input)
        output = (F.log_softmax(mlm_out.logits, dim=-1),)
        if mlm_out.hidden_states is not None:
            output += (mlm_out.hidden_states,)
        if mlm_out.attentions is not None:
            output += (mlm_out.attentions,)
        return output

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

    def _loss_log(self, batch, log_prefix=None):
        input, target = batch
        output = self(input)
        loss = self.loss_fn(output[0].transpose(1, 2), target[1]['input_ids'])
        self.log(f'{log_prefix}.loss', loss, prog_bar=True, rank_zero_only=True, sync_dist=True)
        return {'loss': loss}


class BertMLMLitModelTest(BaseDatasetTest):
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

    @property
    def train_ds(self):
        return self.train_data_loader.dataset

    @property
    def val_ds(self):
        return self.val_data_loader.dataset

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
        model = BertMLMLitModel(train_config=self.train_config)
        model.to(self.device)
        return model

    def create_train_config(self):
        return {
            "mlm_name_or_path": "../output/peft_esm2_t33_650M_UR50D",
            "optimizer": {
                "type": "torch.optim.AdamW",
                "lr": 0.001,
            },
            "lr_scheduler": {
                "type": "warmup_poly",
                "warmup_epochs": 0.0125
            },
        }

    @property
    def first_batch(self):
        batch = next(iter(self.train_data_loader))
        input, target = batch
        input = input.to(self.device)
        target = (target[0].to(self.device), target[1].to(self.device))
        return input, target

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
        logits = output[0]
        expected = (self.batch_size, self.collator.max_len, self.tokenizer.vocab_size)
        self.assertEqual(expected, logits.shape)

    def test_fit(self):
        old_model = self.model.clone()
        old_model.to(self.device)
        self.assertTrue(self.state_dict_equal(old_model.state_dict(), self.model.state_dict()))

        args = self.get_trainer_args()
        trainer = pl.Trainer(accelerator=args.accelerator,
                             devices=args.devices,
                             strategy=args.strategy,
                             num_nodes=args.num_nodes,
                             precision=args.precision,
                             max_epochs=3,
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
                                                  bert_config=self.model.mlm_config,
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
                                                  bert_config=self.model.mlm_config,
                                                  train_config=self.train_config,
                                                  strict=False)
        other = BERTLitModel.from_pretrained_bert(self.train_config, ckpt_path=ckpt_path, bert_path=self.bert_path)
        self.assertEqual(model.bert.state_dict().keys(), other.bert.state_dict().keys())
        self.assertTrue(self.module_weights_equal(model.bert, other.bert))


if __name__ == '__main__':
    unittest.main()
