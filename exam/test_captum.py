import unittest

import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig

from captum.attr import IntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# We need to split forward pass into two part:
# 1) embeddings computation
# 2) classification

def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(
        dtype=next(model_bert.parameters()).dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(
            dtype=next(model_bert.parameters()).dtype)  # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                  1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertModelWrapper(nn.Module):
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings):
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 1].unsqueeze(1)


class CaptumTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def test_bert_classification(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        model_wrapper = BertModelWrapper(model)
        ig = IntegratedGradients(model_wrapper)
        model_wrapper = model_wrapper.to(self.device).eval()
        model_wrapper.zero_grad()

        sentence = 'text to classify'
        label = 0
        input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)]).to(self.device)
        input_embedding = model_wrapper.model.bert.embeddings(input_ids)

        pred = model_wrapper(input_embedding).item()
        pred_ind = round(pred)

        # compute attributions and approximation delta using integrated gradients
        attrs, delta = ig.attribute(input_embedding, n_steps=500, return_convergence_delta=True)
        print('pred: ', pred_ind, '(', '%.2f' % pred, ')', ', delta: ', abs(delta))

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].numpy().tolist())
        self.show_attrs(attrs, tokens, pred, pred_ind, label, delta, vis_data_records_ig)

    def show_attrs(self, attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
        attributions = attributions.sum(dim=2).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.detach().numpy()

        visualization.visualize_text([visualization.VisualizationDataRecord(
            attributions,
            pred,
            pred_ind,
            label,
            "label",
            attributions.sum(),
            tokens[:len(attributions)],
            delta)])


if __name__ == '__main__':
    unittest.main()
