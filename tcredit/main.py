from argparse import ArgumentParser
import logging
import torch
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tcredit.data import EpitopeTargetDataset


# Logger
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger('gentcr')


def generate_datasets(args):
    logger.info(f'Start generate_datasets with args: {args}')

    EpitopeTargetDataset.FN_DATA_CONFIG = args.data_config
    for data_key in args.data.split(','):
        logger.info(f'Generating dataset for {data_key}')
        try:
            ds = EpitopeTargetDataset.from_key(data_key, args=args)
            logger.info(f'Done to generate data for {data_key}, the number of data: {len(ds)}')
        except Exception as e:
            logger.error(f'Failed to generate data for {data_key}: {e}')

def create_peft_model(args):
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
        print(f"all params: {int(all_param):,d} || trainable params: {int(trainable_params):,d} "
              f"|| trainable %: {100 * trainable_params / all_param}")

    logger.info(f'Start create_peft_model with args: {args}')
    model = AutoModelForMaskedLM.from_pretrained(args.plm_name_or_path,
                                                 # quantization_config=create_bnb_config(bits=args.bits),
                                                 # load_in_8bit=True,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    # Using the prepare_model_for_kbit_training method from PEFT
    # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=model.supports_gradient_checkpointing)

    # Create peft model with lora module names
    modules = find_all_linear_names(model)
    peft_config = create_peft_config(modules)
    logger.info(f'Creating peft model with lora config: {peft_config}')
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    # print_trainable_parameters(model, use_4bit=(args.bits == 4))

    # Save the model
    logger.info(f'Saving the model and tokenizer to {args.outdir}')
    model.save_pretrained(args.outdir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name_or_path)
    tokenizer.save_pretrained(args.outdir)
    logger.info(f'Done to create_peft_model.')

    # Merge the model with LORA adaptor
    # model = AutoPeftModel.from_pretrained(args.outdir, device_map="auto", torch_dtype=torch.bfloat16)
    # model = model.merge_and_unload()
    # model.save_pretrained(args.outdir, safe_serialization=True)


def main():
    parser = ArgumentParser('gentcr')
    parser.add_argument('--log_level', type=str, default='DEBUG')
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'generate_datasets'
    sub_parser = subparsers.add_parser('generate_datasets')
    sub_parser.set_defaults(func=generate_datasets)
    sub_parser.add_argument('--data_config', type=str, default='../config/data.json')
    sub_parser.add_argument('--data', type=str, default='immunecode')
    sub_parser.add_argument('--n_workers', type=int, default=1)

    # Arguments for sub command 'create_peft_model'
    sub_parser = subparsers.add_parser('create_peft_model')
    sub_parser.set_defaults(func=create_peft_model)
    sub_parser.add_argument('--plm_name_or_path', type=str, default='facebook/esm2_t33_650M_UR50D')
    sub_parser.add_argument('--outdir', type=str, default='../output/peft_esm2_t33_650M_UR50D')

    # # Arguments for sub command 'run_train'
    # sub_parser = subparsers.add_parser('run_train')
    # sub_parser.set_defaults(func=run_train)
    # sub_parser.add_argument('--fn_source', type=str, default='output/laion256_latent.pt')
    # sub_parser.add_argument('--chunk_size', type=int, default=127399)
    # sub_parser.add_argument('--z_shape', type=tuple, default=(4, 32, 32))
    # sub_parser.add_argument('--d_cond', type=int, default=768)
    # sub_parser.add_argument('--latent_scaling_factor', type=float, default=0.18215)
    # sub_parser.add_argument('--batch_size', type=int, default=16)
    # sub_parser.add_argument('--accelerator', type=str, default='gpu')
    # sub_parser.add_argument('--devices', type=int, default=2)
    # # sub_parser.add_argument('--strategy', type=str, default='deepspeed_stage_3')
    # sub_parser.add_argument('--accumulate_grad_batches', type=int, default=2)
    # sub_parser.add_argument('--precision', type=int, default=16)
    # sub_parser.add_argument('--max_epochs', type=int, default=30)
    # sub_parser.add_argument('--n_workers', type=int, default=1)
    # sub_parser.add_argument('--filepath', type=str, default='output/best_ldm.ckpt')
    #
    # # Arguments for sub command 'gen_sample'
    # prompts = ["Call for Applications AFMA 2020",
    #            "Only a strong woman can work for AT&T | AT&T Shirt",
    #            "Deadlines... T Shirt",
    #            "Girl holding blank board Stock Photo",
    #            "Rocket Space Dog Costume",
    #            "Girl holding blank board Stock Photo"]
    # sub_parser = subparsers.add_parser('gen_samples')
    # sub_parser.set_defaults(func=gen_samples)
    # sub_parser.add_argument('--ckpt', type=str, default='output/best_ldm.ckpt/model.ckpt')
    # sub_parser.add_argument('--sampler', type=str, default='ddpm')
    # sub_parser.add_argument('--autoencoder', type=str, default='kl-f8')
    # sub_parser.add_argument('--z_shape', type=tuple, default=(4, 32, 32))
    # sub_parser.add_argument('--latent_scaling_factor', type=float, default=0.18215)
    # sub_parser.add_argument('--clip', type=str, default='openai/clip-vit-large-patch14')
    # sub_parser.add_argument('--prompts', type=int, default=prompts)
    # sub_parser.add_argument('--d_cond', type=int, default=768)
    # sub_parser.add_argument('--n_steps', type=int, default=1000)

    args = parser.parse_args()
    print(f'Logging level: {args.log_level}')
    logger.setLevel(args.log_level)
    args.func(args)

if __name__ == '__main__':
    main()
