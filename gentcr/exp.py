import copy
import os
import logging
# from transformers.utils import logging as hf_logging
import unittest
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, AutoModelForMaskedLM, AutoTokenizer, TrainingArguments, Trainer, \
    EarlyStoppingCallback
import bitsandbytes as bnb

from bioseq import UniformAASeqMutator, CalisImmunogenicAASeqMutator
from gentcr.common import SlurmUtils, StrUtils, FileUtils, CollectionUtils, BaseTest
from gentcr.data import EpitopeTargetDataset, EpitopeTargetMaskedLMCollator

# Logger
logger = logging.getLogger(__name__)


class Experiment(object):
    class Task(object):
        CALLBACK_CLS_MAP = {
            'EarlyStoppingCallback': EarlyStoppingCallback
        }

        def __init__(self, config=None):
            self.config = config

        def run(self, args):
            task_key = self.config['key']
            task_type = self.config['type']
            logger.info(f"Run task key: {task_key}, type: {task_type}")
            self._run(args)

        def _run(self, args):
            raise NotImplementedError()

    class MLMFinetuneTask(Task):
        def __init__(self, config=None):
            super().__init__(config=config)

        def _run(self, args=None):
            logger.info(f'Start to MLMFinetuneTask.run with args: {args}...')
            output_dir = self.config['output_dir']
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
                logger.info(f'{output_dir} was created')

            # Load the pretrained model and tokenizer
            model, tokenizer = self._load_peft_model()
            train_ds, val_ds, data_collator = self._load_train_val_datasets(tokenizer)
            train_args = TrainingArguments(**self.config['trainer']['args'])
            callbacks = self._create_callbacks()
            trainer = Trainer(model=model,
                              train_dataset=train_ds,
                              eval_dataset=val_ds,
                              data_collator=data_collator,
                              args=train_args,
                              callbacks=callbacks)

            logger.info(f'Start trainer.train with train_args: {train_args}')
            model.config.use_cache = False  # re-enable for inference to speed up predictions for similar inputs
            model.config.pretraining_tp = 1

            train_result = trainer.train()
            logger.info(f'Done to train. train_result: {train_result}')
            logger.info(f'Saving final model and tokenizer to {output_dir}')
            trainer.model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

        def _load_peft_model(self):
            def create_bnb_config(bits=4):
                if bits == 4:
                    return BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16)
                elif bits == 8:
                    return BitsAndBytesConfig(load_in_8bit=True)
                else:
                    return None

            def create_peft_config(modules):
                config = LoraConfig(
                    r=16,  # dimension of the updated matrices
                    lora_alpha=32,  # parameter for scaling
                    target_modules=modules,
                    lora_dropout=0.05,  # dropout probability for layers
                    bias="none",
                    # task_type=TaskType.TOKEN_CLS
                )
                return config

            def find_all_linear_names(model, bits=None):
                cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
                lora_module_names = set()
                for name, module in model.named_modules():
                    if isinstance(module, cls):
                        names = name.split('.')
                        lora_module_names.add(names[0] if len(names) == 1 else names[-1])

                if 'lm_head' in lora_module_names:  # needed for 16-bit
                    lora_module_names.remove('lm_head')
                return list(lora_module_names)

            def print_trainable_parameters(model, use_4bit=False):
                trainable_params = 0
                all_param = 0
                for _, param in model.named_parameters():
                    num_params = param.numel()
                    # if using DS Zero 3 and the weights are initialized empty
                    if num_params == 0 and hasattr(param, "ds_numel"):
                        num_params = param.ds_numel
                    all_param += num_params
                    if param.requires_grad:
                        trainable_params += num_params
                if use_4bit:
                    trainable_params /= 2
                logger.info(f"all params: {int(all_param):,d} || trainable params: {int(trainable_params):,d} "
                            f"|| trainable %: {100 * trainable_params / all_param}")

            plm_name_or_path = self.config['plm_name_or_path']
            logger.info(f'Start loading pretrained model from {plm_name_or_path}')
            bits = self.config['peft'].get('bits', 4)
            model = AutoModelForMaskedLM.from_pretrained(plm_name_or_path,
                                                         quantization_config=create_bnb_config(bits=bits),
                                                         # load_in_8bit=(bits == 8),
                                                         # load_in_4bit=(bits == 4),
                                                         # torch_dtype=torch.bfloat16,
                                                         device_map="auto")
            # Using the prepare_model_for_kbit_training method from PEFT
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model.supports_gradient_checkpointing)

            # Create peft model with lora module names
            modules = find_all_linear_names(model)
            peft_config = create_peft_config(modules)
            logger.info(f'Creating peft model with lora config: {peft_config}')
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            # print_trainable_parameters(model, use_4bit=(bits == 4))
            tokenizer = AutoTokenizer.from_pretrained(plm_name_or_path)
            return model, tokenizer

        def _load_train_val_datasets(self, tokenizer):
            def create_seq_mutator(mut_config=None):
                if mut_config:
                    mut_config = copy.deepcopy(mut_config)
                    mutator_type = mut_config.pop('type')
                    if mutator_type == 'uniform':
                        return UniformAASeqMutator(**mut_config)
                    elif mutator_type == 'calis':
                        return CalisImmunogenicAASeqMutator(**mut_config)
                    else:
                        raise ValueError(f'Unsupported seq mutator type: {mutator_type}')
                return None

            config = self.config['data']
            EpitopeTargetDataset.FN_DATA_CONFIG = config.get('config', '../config/data-test.json')

            ds = EpitopeTargetDataset.from_key(config['data_key'])
            train_ds, val_ds = ds.train_test_split(test_size=config['val_size'], shuffle=True)

            epitope_seq_mutator = None
            target_seq_mutator = None
            if 'seq_mutators' in config:
                epitope_seq_mutator = create_seq_mutator(config['seq_mutators'].get('epitope'))
                target_seq_mutator = create_seq_mutator(config['seq_mutators'].get('target'))

            data_collator = EpitopeTargetMaskedLMCollator(tokenizer=tokenizer,
                                                          epitope_seq_mutator=epitope_seq_mutator,
                                                          target_seq_mutator=target_seq_mutator,
                                                          max_epitope_len=ds.max_epitope_len,
                                                          max_target_len=ds.max_target_len,
                                                          seq_format=config.get('seq_format',
                                                                                '{epitope_seq}{target_seq}'))
            return train_ds, val_ds, data_collator

        def _create_callbacks(self):
            config = self.config['trainer'].get('callbacks')
            callbacks = []
            if config:
                for cb_config in config:
                    cb_config = copy.deepcopy(cb_config)
                    cb_cls = self.CALLBACK_CLS_MAP.get(cb_config.pop('type'))
                    callbacks.append(cb_cls(**cb_config))
            return callbacks

    FN_CONFIG = '../config/exp.json'
    _exp_configs = None

    def __init__(self, config=None):
        self.config = self._update_task_configs(copy.deepcopy(config))

    def create_task(self, task_key=None):
        config = self.get_task_config(task_key)
        if config['type'] == 'mlm_finetune':
            return Experiment.MLMFinetuneTask(config=config)
        else:
            raise ValueError(f'Unsupported task type: {config["type"]}')

    def get_task_config(self, task_key):
        return self.config['taskflow'][task_key]

    def get_task_config_keys(self):
        return self.config['taskflow'].keys()

    def _update_task_configs(self, config=None):
        output_dir = config['output_dir']
        for task_key, task_config in config['taskflow'].items():
            task_config['key'] = task_key
            task_config['output_dir'] = f'{output_dir}/{task_key}'
            CollectionUtils.recursive_replace_substr(task_config, '{task_output_dir}', task_config['output_dir'])
        return config

    @classmethod
    def from_key(cls, key=None, reload=False):
        return Experiment(config=cls.load_config(key, reload))

    @classmethod
    def load_config(cls, key=None, reload=False):
        if (cls._exp_configs is None) or reload:
            cls._exp_configs = FileUtils.json_load(cls.FN_CONFIG)
        return cls._exp_configs[key]


class ExperimentTest(BaseTest):
    def setUp(self):
        super().setUp()
        self.exp_key = 'testexp'
        self.exp = Experiment.from_key(self.exp_key)

    def test_mlm_finetune_task_config(self):
        task_key = 'mlm_finetune'
        task = self.exp.create_task(task_key)
        self.assertTrue(isinstance(task, Experiment.MLMFinetuneTask))
        expected_output_dir = f"{self.exp.config['output_dir']}/{task_key}"
        self.assertEqual(task.config['output_dir'], expected_output_dir)
        self.assertEqual(task.config['trainer']['args']['output_dir'], expected_output_dir)
        self.assertEqual(task.config['trainer']['args']['logging_dir'], f'{expected_output_dir}/logs')


if __name__ == '__main__':
    unittest.main()
