from transformers.models.megatron_gpt2.convert_megatron_gpt2_checkpoint import convert_megatron_checkpoint, recursive_print
import sys
import os
import zipfile
import torch
import argparse
from transformers import AutoTokenizer, GPT2Config, PreTrainedTokenizerFast
from tokenizers.processors import TemplateProcessing
from tokenizers import Tokenizer

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                    os.path.pardir)))
def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    args = parser.parse_args()

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint)

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    ds_args = input_state_dict.get("args", None)

    # Read the config, or default to the model released by NVIDIA.
    if args.config_file == "":
        if ds_args is not None:
            if ds_args.swiglu:
                activation_function = "swiglu"
            elif ds_args.bias_gelu_fusion:
                activation_function = "gelu_fast"
            elif ds_args.openai_gelu:
                activation_function = "gelu_new"
            else:
                activation_function = "gelu"
        else:
            # in the very early days this used to be "gelu_new"
            activation_function = "gelu_new"

        # Spell out all parameters in case the defaults change.
        config = GPT2Config(
            vocab_size=128,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function=activation_function,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=3,
            eos_token_id=4,
        )
    else:
        config = GPT2Config.from_json_file(args.config_file)

    config.architectures = ["GPT2LMHeadModel"]

    # Convert.
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # Add tokenizer class info to config
    # see https://github.com/huggingface/transformers/issues/13906)
    if ds_args is not None:
        tokenizer_type = ds_args.tokenizer_type
        if tokenizer_type == "GPT2BPETokenizer":
            tokenizer_model_name = "gpt2"
        elif tokenizer_type == "PretrainedFromHF":
            tokenizer_model_name = ds_args.tokenizer_name_or_path
        else:
            raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    else:
        tokenizer_model_name = "gpt2"
    
    # ## init tokenizer ##
    # progen_tokenizer = create_tokenizer_custom(dict_path)
    # # add bos and eos
    # progen_tokenizer.post_processor = TemplateProcessing(
    #     single="1 $A 2",
    #     special_tokens=[("1", 3), ("2", 4)],
    # ) 

    # progen_tokenizer = PreTrainedTokenizerFast(tokenizer_object=progen_tokenizer)
    # progen_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    # progen_tokenizer.add_special_tokens({'bos_token': '1'})        
    # progen_tokenizer.add_special_tokens({'eos_token': '2'})  

    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    # tokenizer_class = type(tokenizer).__name__
    # config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(basename)

    # Save tokenizer based on args
    # print(f"Adding {tokenizer_class} tokenizer files")
    # tokenizer.save_pretrained(basename)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)

    return basename

if __name__ == '__main__':
    basename = main()
    print('Convert done, now check the converted checkpoint')
    # basename = '/root/ZJLAB_Progen_13B'
    from models import GPT2LMHeadModel
    zj_progen = GPT2LMHeadModel.from_pretrained(basename).cuda()

    progen_tokenizer = create_tokenizer_custom(os.path.join(basename, 'tokenizer.json'))
    # add bos and eos
    # progen_tokenizer.post_processor = TemplateProcessing(
    #     single="1 $A 2",
    #     special_tokens=[("1", 3), ("2", 4)],
    # ) 

    progen_tokenizer = PreTrainedTokenizerFast(tokenizer_object=progen_tokenizer)
    progen_tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    progen_tokenizer.add_special_tokens({'bos_token': '1'})        
    progen_tokenizer.add_special_tokens({'eos_token': '2'})  

    tokens = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            target = torch.tensor(progen_tokenizer.encode(tokens)).to(zj_progen.device)
            output = zj_progen(target, labels=target)

    print(output.loss)