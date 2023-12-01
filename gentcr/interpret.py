import captum.insights.attr_vis.server
import torch.cuda
import unittest
from captum.attr import configure_interpretable_embedding_layer, LayerConductance
from peft import AutoPeftModel
from transformers import AutoTokenizer

from gentcr.data import EpitopeTargetDataset, CN


class EsmMLMAttrInterpreter:
    def __init__(self,
                 esm_name_or_path,
                 device=('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.model, self.tokenizer = self.load_model_and_tokenizer(esm_name_or_path, device)
        self.device = device
        self.interpretable_embedding = None

    def load_model_and_tokenizer(self, esm_name_or_path, device):
        model = AutoPeftModel.from_pretrained(esm_name_or_path, output_attentions=True)
        model = model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(esm_name_or_path)
        model.esm.embeddings.token_dropout = False  # ESM has a bug when inputs_embeds is used
        return model, tokenizer

    def interpret_attributions(self, input_seqs):
        def attribute_forward(inputs, input_ids=None, attention_mask=None):
            output = self.model(inputs_embeds=inputs, attention_mask=attention_mask)
            return output.logits.max(dim=-1).values

        def reduce_attrs(attrs):
            attrs = attrs.sum(dim=-1).squeeze(0)
            attrs = attrs / torch.norm(attrs)
            return attrs

        
        max_len = max([len(seq) for seq in input_seqs]) + 2 # +2 for [CLS] and [EOS]
        inputs = self.tokenizer(input_seqs,
                                padding="max_length",
                                truncation=False,
                                max_length=max_len,
                                return_overflowing_tokens=False,
                                return_tensors="pt").to(self.device)

        input_ids = inputs["input_ids"]
        ref_input_ids = self.get_ref_input_ids(input_ids)
        if self.interpretable_embedding is None:
            self.interpretable_embedding = configure_interpretable_embedding_layer(self.model,
                                                                                   'esm.embeddings.word_embeddings')
        input_embeddings = self.interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = self.interpretable_embedding.indices_to_embeddings(ref_input_ids)

        layer_attrs = []
        layer_attns = []
        for layer in self.model.esm.encoder.layer:
            lc = LayerConductance(attribute_forward, layer)
            attrs = lc.attribute(inputs=input_embeddings,
                                 baselines=ref_input_embeddings,
                                 additional_forward_args=(inputs['input_ids'],
                                                          inputs['attention_mask']))
            layer_attrs.append(reduce_attrs(attrs[0]))
            layer_attns.append(attrs[1])

        layer_attrs = torch.stack(layer_attrs)
        layer_attns = torch.stack(layer_attns)
        return layer_attrs, layer_attns

    def get_ref_input_ids(self, input_ids=None):
        ref_input_ids = torch.fill(input_ids.clone(), self.tokenizer.pad_token_id)
        ref_input_ids[:, 0] = self.tokenizer.cls_token_id
        ref_input_ids[:, -1] = self.tokenizer.eos_token_id
        return ref_input_ids


class BertMLMAttrInterpreterTest(unittest.TestCase):
    def setUp(self):
        self.mlm_name_or_path = '../output/exp3/mlm_finetune'
        self.interpreter = EsmMLMAttrInterpreter(self.mlm_name_or_path)
        self.eval_ds = EpitopeTargetDataset.from_key('shomuradova_minervina_gfeller')
        self.seq_format = '{epitope_seq}{target_seq}'
        self.input_seqs = [self.seq_format.format(epitope_seq=e_seq, target_seq=t_seq)
                           for e_seq, t_seq in zip(self.eval_ds.df[CN.epitope_seq].values,
                                                   self.eval_ds.df[CN.cdr3b_seq].values)]

    def test_interpret_attribution(self):
        attrs, attns = self.interpreter.interpret_attributions(self.input_seqs[:3])
        print(attrs.shape)
        print(attns.shape)

if __name__ == '__main__':
    unittest.main()
