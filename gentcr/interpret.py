import copy

import captum.insights.attr_vis.server
import torch.cuda
import torch.nn.functional as F
import unittest
from captum.attr import configure_interpretable_embedding_layer, LayerConductance, LayerIntegratedGradients
from peft import AutoPeftModel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np

from gentcr.bioseq import FixedPosAASeqMutator
from gentcr.data import EpitopeTargetDataset, CN, EpitopeTargetMaskedLMCollator
from gentcr.common import TorchUtils


class EsmMLMAttrInterpreter:
    def __init__(self, model=None):
        self.model = model
        self.device = model.device

    def clone_model(self):
        clone = copy.deepcopy(self.model)
        clone.esm.embeddings.token_dropout = False  # ESM has a bug when inputs_embeds is used
        clone = clone.to(self.device).eval()
        clone.zero_grad()
        return clone

    def layer_integrated_gradients_attrs(self, inputs, bl_input_ids=None):
        def attr_forward(input_ids, model, attention_mask=None, target_mask=None):
            # inputs.shape = (batch_size, max_len, hidden_size)
            # shape of input_ids, attention_mask, target_mask = (batch_size, max_len)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            # output.logits.shape = (batch_size, max_len, vocab_size)
            logits = output.logits[target_mask] # logits.view(-1, vocab_size)
            logits = F.softmax(logits, dim=-1)
            logits = logits.max(dim=-1).values
            return logits

        model = self.clone_model()
        inputs = inputs.to(self.device)
        bl_input_ids = bl_input_ids.to(self.device)
        targets = inputs['labels']
        target_mask = (targets != -100)
        targets = targets[target_mask]
        lig = LayerIntegratedGradients(attr_forward, model.esm.embeddings)
        attrs = lig.attribute(inputs=inputs['input_ids'],
                              baselines=bl_input_ids,
                              internal_batch_size=8,
                              additional_forward_args=(model, inputs['attention_mask'], target_mask),
                              # target=targets.view(-1),
                              return_convergence_delta=False)
        return attrs


    def layer_conductance_attrs(self, inputs, bl_input_ids=None):
        def attr_forward(inputs, model, input_ids=None, attention_mask=None, target_mask=None):
            # inputs.shape = (batch_size, max_len, hidden_size)
            # shape of input_ids, attention_mask, target_mask = (batch_size, max_len)
            output = model(inputs_embeds=inputs, attention_mask=attention_mask)
            # output.logits.shape = (batch_size, max_len, vocab_size)
            logits = output.logits[target_mask] # logits.view(-1, vocab_size)
            logits = F.softmax(logits, dim=-1)
            logits = logits.max(dim=-1).values
            return logits

        model = self.clone_model()
        inputs = inputs.to(self.device)
        bl_input_ids = bl_input_ids.to(self.device)

        input_ids = inputs["input_ids"]
        targets = inputs['labels']
        target_mask = (targets != -100)
        targets = targets[target_mask]
        interpretable_embedding = configure_interpretable_embedding_layer(model,
                                                                          'esm.embeddings.word_embeddings')
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        bl_input_embeddings = interpretable_embedding.indices_to_embeddings(bl_input_ids)

        layer_attrs = []
        layer_attns = []
        for i, layer in enumerate(model.esm.encoder.layer):
            lc = LayerConductance(attr_forward, layer)
            # attrs.shape = (batch_size, max_len, hidden_size), attns.shape = (batch_size, num_heads, max_len, max_len)
            attrs, attns = lc.attribute(inputs=input_embeddings,
                                        baselines=bl_input_embeddings,
                                        internal_batch_size=8,
                                        # n_steps=40,
                                        # target=targets.view(-1),
                                        additional_forward_args=(model,
                                                                 inputs['input_ids'],
                                                                 inputs['attention_mask'],
                                                                 target_mask))
            layer_attrs.append(attrs)
            layer_attns.append(attns)
            print(f'Layer {i} done')

        layer_attrs = torch.stack(layer_attrs, dim=0)
        layer_attns = torch.stack(layer_attns, dim=0)
        return layer_attrs, layer_attns


class EsmMLMAttrInterpreterTest(unittest.TestCase):
    def setUp(self):
        self.mlm_name_or_path = '../output/exp3/mlm_finetune'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer = self.load_model_and_tokenizer(self.mlm_name_or_path)
        self.model = self.model.to(self.device)
        self.interpreter = EsmMLMAttrInterpreter(model=self.model)
        self.eval_ds = EpitopeTargetDataset.from_key('shomuradova_minervina_gfeller')
        epitope_seq_mutator = FixedPosAASeqMutator(mut_positions=[3], mut_probs=[1, 0])

        self.collator = EpitopeTargetMaskedLMCollator(tokenizer=self.tokenizer,
                                                      epitope_seq_mutator=epitope_seq_mutator,
                                                      target_seq_mutator=None,
                                                      max_epitope_len=self.eval_ds.max_epitope_len,
                                                      max_target_len=self.eval_ds.max_target_len)

    def first_batch(self, batch_size=1):
        return next(iter(DataLoader(self.eval_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collator)))

    def load_model_and_tokenizer(self, esm_name_or_path):
        model = AutoPeftModel.from_pretrained(esm_name_or_path, output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(esm_name_or_path)
        return model, tokenizer

    def get_baseline_input_ids(self, input_ids=None):
        bl_input_ids = torch.fill(input_ids.clone(), self.tokenizer.pad_token_id)
        bl_input_ids[:, 0] = self.tokenizer.cls_token_id
        bl_input_ids[:, -1] = self.tokenizer.eos_token_id
        return bl_input_ids

    def test_clone_model(self):
        clone = self.interpreter.clone_model()
        self.assertIsNot(clone, self.model)
        self.assertTrue(TorchUtils.equal_state_dict(clone.state_dict(), self.model.state_dict()))
        self.assertIsNot(clone.config, self.model.config)
        self.assertEqual(clone.config, self.model.config)

    def test_layer_conductance_attrs(self):
        self.collator.epitope_seq_mutator.mut_positions = [3]
        inputs = self.first_batch(batch_size=1)
        input_ids = inputs['input_ids']
        print(f'Masked input[0]: {self.tokenizer.decode(input_ids[0])}')
        bl_input_ids = self.get_baseline_input_ids(inputs['input_ids'])
        attrs, attns = self.interpreter.layer_conductance_attrs(inputs, bl_input_ids=bl_input_ids)
        # attrs.shape = (num_layers, batch_size, max_len, hidden_size)
        # attns.shape = (num_layers, batch_size, num_heads, max_len, max_len)
        attrs = TorchUtils.to_numpy(attrs)
        attns = TorchUtils.to_numpy(attns)
        print(attrs.shape)
        print(attns.shape)
        attrs = np.sum(attrs, axis=(0, 3))
        attrs = attrs / np.linalg.norm(attrs)
        # attrs = np.mean(attrs, axis=(0, 3))
        attns = np.sum(attns, axis=(0, 2, 3))
        attns = attns / np.linalg.norm(attns)
        # attns = np.mean(attns, axis=(0, 2, 3))

        seq = self.collator.format_seqs(self.eval_ds.df.epitope_seq[0],
                                        self.eval_ds.df.cdr3b_seq[0])
        tokens = [self.tokenizer.cls_token] + list(seq) + [self.tokenizer.eos_token]

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 12))
        plt.tight_layout()
        ax = sns.barplot(attrs[0][:len(tokens)], ax=axes[0])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens)
        # ax = sns.heatmap(attns[0], cmap='coolwarm', linewidth=0.2, ax=axes[1])
        # ax.set_yticklabels(tokens)
        ax = sns.barplot(attns[0][:len(tokens)], ax=axes[1])
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens)
        plt.show()

    def test_layer_integrated_grad_attrs(self):
        self.collator.epitope_seq_mutator.mut_positions = [3]
        inputs = self.first_batch(batch_size=1)
        input_ids = inputs['input_ids']
        print(f'Masked input[0]: {self.tokenizer.decode(input_ids[0])}')
        bl_input_ids = self.get_baseline_input_ids(inputs['input_ids'])
        attrs = self.interpreter.layer_integrated_gradients_attrs(inputs, bl_input_ids=bl_input_ids)
        # attrs.shape = (batch_size, max_len, hidden_size)
        attrs = TorchUtils.to_numpy(attrs)
        print(attrs.shape)
        attrs = np.sum(attrs, axis=-1)
        attrs = attrs / np.linalg.norm(attrs)
        # attrs = np.mean(attrs, axis=-1)
        seq = self.collator.format_seqs(self.eval_ds.df.epitope_seq[0],
                                        self.eval_ds.df.cdr3b_seq[0])
        tokens = [self.tokenizer.cls_token] + list(seq) + [self.tokenizer.eos_token]

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        plt.tight_layout()
        ax = sns.barplot(attrs[0][:len(tokens)], ax=axes)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens)
        plt.show()

if __name__ == '__main__':
    unittest.main()
