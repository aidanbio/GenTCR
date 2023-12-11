import unittest
import torch
import torch.nn.functional as F
from captum.attr import configure_interpretable_embedding_layer, LayerConductance
from transformers import BertForQuestionAnswering, AutoTokenizer, BertTokenizer, AutoModelForQuestionAnswering
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gentcr.common import TorchUtils

def show_token_by_token_scores(scores_mat, title='Untitled', x_label_name='Head', tokens=None):
    fig = plt.figure(figsize=(25, 30))
    fig.suptitle(title, fontsize=30)
    plt.tight_layout(pad=5)

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(4, 3, idx + 1)
        # append the attention weights
        im = ax.imshow(scores, cmap='coolwarm')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))

        ax.set_xticklabels(tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(tokens, fontdict=fontdict)
        ax.set_xlabel('{} {}'.format(x_label_name, idx + 1))
        fig.colorbar(im, fraction=0.046, pad=0.04)

    plt.show()

def show_token_by_head_scores(scores_mat, tokens=None):
    fig = plt.figure(figsize=(30, 50))

    for idx, scores in enumerate(scores_mat):
        scores_np = np.array(scores)
        ax = fig.add_subplot(6, 2, idx+1)
        # append the attention weights
        im = ax.matshow(scores_np, cmap='viridis')

        fontdict = {'fontsize': 20}

        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(scores)))

        ax.set_xticklabels(tokens, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(range(len(scores[0])), fontdict=fontdict)
        ax.set_xlabel('Layer {}'.format(idx+1))

        fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

class DrawAttnTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = 'quangb1910128/bert-finetuned-squad'
        self.model, self.tokenizer = self.load_model_and_tokenizer()

        self.ref_token_id = self.tokenizer.pad_token_id  # A token used for generating token reference
        self.sep_token_id = self.tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
        self.cls_token_id = self.tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence
        # self.interpretable_embedding = configure_interpretable_embedding_layer(self.model, 'bert.embeddings.word_embeddings')

    def load_model_and_tokenizer(self):
        model = BertForQuestionAnswering.from_pretrained(self.model_path, output_attentions=True)
        model = model.to(self.device).eval()
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        return model, tokenizer

    def get_input_ref_pair(self, question, answer):
        question_ids = self.tokenizer.encode(question, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)

        # construct input token ids
        input_ids = [self.cls_token_id] + question_ids + [self.sep_token_id] + answer_ids + [self.sep_token_id]

        # construct reference token ids
        ref_input_ids = [self.cls_token_id] + [self.ref_token_id] * len(question_ids) + [self.sep_token_id] + \
                        [self.ref_token_id] * len(answer_ids) + [self.sep_token_id]

        return (torch.tensor([input_ids], device=self.device),
                torch.tensor([ref_input_ids], device=self.device),
                len(question_ids))

    def get_input_ref_token_type_pair(self, input_ids, sep_ind=0):
        seq_len = input_ids.size(1)
        token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=self.device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device)  # * -1
        return token_type_ids, ref_token_type_ids

    def get_input_ref_pos_id_pair(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
        # we could potentially also use random permutation with `torch.randperm(seq_length, device=device)`
        ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=self.device)

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids, ref_position_ids

    def get_attention_mask(self, input_ids):
        return torch.ones_like(input_ids)

    def get_interpretable_word_embeddings(self, input_ids, ref_input_ids):
        interpretable_embedding = self.configure_interpretable_word_embedding_layer()
        input_embeddings = interpretable_embedding.indices_to_embeddings(input_ids)
        ref_input_embeddings = interpretable_embedding.indices_to_embeddings(ref_input_ids)
        return input_embeddings, ref_input_embeddings

    def configure_interpretable_word_embedding_layer(self):
        return configure_interpretable_embedding_layer(self.model, 'bert.embeddings.word_embeddings')

    def predict(self, question, answer):
        input_ids, ref_input_ids, sep_index = self.get_input_ref_pair(question, answer)
        token_type_ids, ref_token_type_ids = self.get_input_ref_token_type_pair(input_ids, sep_index)
        position_ids, ref_position_ids = self.get_input_ref_pos_id_pair(input_ids)
        attention_mask = self.get_attention_mask(input_ids)
        print(f'input_ids.shape: {input_ids.shape}, token_type_ids.shape: {token_type_ids.shape}, '
              f'position_ids.shape: {position_ids.shape}, attention_mask.shape: {attention_mask.shape}')

        output = self.model(input_ids,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            attention_mask=attention_mask)
        return input_ids, output

    def test_predict(self):
        question = 'What is important to us?'
        answer = 'It is important to us to include, empower and support humans of all kinds.'
        input_ids, output = self.predict(question=question, answer=answer)

        start_logits = output.start_logits
        end_logits = output.end_logits
        attentions = torch.stack(output.attentions)
        print(f'start_logits.shape: {start_logits.shape},'
              f'end_logits.shape: {end_logits.shape},'
              f'attentions.shape: {attentions.shape}')

        input_len = input_ids.size(1)
        expected_shape = (1, input_len)
        self.assertEqual(start_logits.shape, expected_shape)
        self.assertEqual(end_logits.shape, expected_shape)
        expected_shape = (self.model.config.num_hidden_layers, 1, self.model.config.num_attention_heads, input_len, input_len)
        self.assertEqual(expected_shape, attentions.shape)

    def test_show_token_by_token_norm_attns(self):
        question = 'What is important to us?'
        answer = 'It is important to us to include, empower and support humans of all kinds.'
        input_ids, output = self.predict(question, answer)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        attentions = torch.stack(output.attentions)
        layer = 11
        norm_attns = TorchUtils.to_numpy(torch.norm(attentions, dim=2).squeeze())
        show_token_by_token_scores(norm_attns, x_label_name='Layer', tokens=tokens)

    def test_show_token_by_token_attrs(self):
        question = 'What is important to us?'
        answer = 'It is important to us to include, empower and support humans of all kinds.'
        layer_attrs, layer_attn_mat, tokens = self.get_attr_scores(question, answer)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        plt.tight_layout(pad=6)

        xticklabels = tokens
        yticklabels = list(range(1, 13))
        ax = sns.heatmap(layer_attrs[0],
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                         linewidth=0.2,
                         cmap='coolwarm',
                         ax=axes[0])
        ax.set_title('Predicted Start Position')
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        ax = sns.heatmap(layer_attrs[1],
                         xticklabels=xticklabels,
                         yticklabels=yticklabels,
                         linewidth=0.2,
                         cmap='coolwarm',
                         ax=axes[1])
        ax.set_title('Predicted End Position')
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        plt.show()

    def test_show_token_by_token_attn_attrs(self):
        question = 'What is important to us?'
        answer = 'It is important to us to include, empower and support humans of all kinds.'
        layer_attrs, layer_attn_mat, tokens = self.get_attr_scores(question, answer)
        layer = 11
        show_token_by_token_scores(layer_attn_mat[0][layer].squeeze(), title='Attention of Start Positions by Heads', x_label_name='Head', tokens=tokens)
        show_token_by_token_scores(layer_attn_mat[1][layer].squeeze(), title='Attention of End Positions by Heads', x_label_name='Head', tokens=tokens)

        norm_attns = np.linalg.norm(layer_attn_mat[0], axis=2)
        show_token_by_token_scores(norm_attns.squeeze(), title='Attention of Start Positions by Layers', x_label_name='Layer', tokens=tokens)
        norm_attns = np.linalg.norm(layer_attn_mat[1], axis=2)
        show_token_by_token_scores(norm_attns.squeeze(), title='Attention of End Positions by Layers', x_label_name='Layer', tokens=tokens)


    def get_attr_scores(self, question, answer):
        def forward_func(inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0):
            pred = self.model(inputs_embeds=inputs,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              attention_mask=attention_mask)
            pred = pred[position]
            # pred = F.softmax(pred, dim=-1)
            return pred.max(1).values

        def summarize_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            return attributions

        input_ids, ref_input_ids, sep_index = self.get_input_ref_pair(question, answer)
        token_type_ids, ref_token_type_ids = self.get_input_ref_token_type_pair(input_ids, sep_index)
        position_ids, ref_position_ids = self.get_input_ref_pos_id_pair(input_ids)
        attention_mask = self.get_attention_mask(input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())


        input_embeddings, ref_input_embeddings = self.get_interpretable_word_embeddings(input_ids, ref_input_ids)
        layer_attrs_start = []
        layer_attrs_end = []
        layer_attn_mat_start = []
        layer_attn_mat_end = []
        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(forward_func, self.model.bert.encoder.layer[i])
            layer_attributions_start = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings,
                                                    additional_forward_args=(
                                                        token_type_ids, position_ids, attention_mask, 0))
            layer_attributions_end = lc.attribute(inputs=input_embeddings, baselines=ref_input_embeddings,
                                                  additional_forward_args=(
                                                      token_type_ids, position_ids, attention_mask, 1))

            layer_attrs_start.append(summarize_attributions(layer_attributions_start[0]))
            layer_attrs_end.append(summarize_attributions(layer_attributions_end[0]))

            layer_attn_mat_start.append(layer_attributions_start[1])
            layer_attn_mat_end.append(layer_attributions_end[1])
        # layer x seq_len
        layer_attrs_start = TorchUtils.to_numpy(torch.stack(layer_attrs_start))
        # layer x seq_len
        layer_attrs_end = TorchUtils.to_numpy(torch.stack(layer_attrs_end))
        # layer x batch x head x seq_len x seq_len
        layer_attn_mat_start = TorchUtils.to_numpy(torch.stack(layer_attn_mat_start))
        # layer x batch x head x seq_len x seq_len
        layer_attn_mat_end = TorchUtils.to_numpy(torch.stack(layer_attn_mat_end))
        return (layer_attrs_start, layer_attrs_end), (layer_attn_mat_start, layer_attn_mat_end), tokens


if __name__ == '__main__':
    unittest.main()
