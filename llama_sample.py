"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import argparse
import pandas as pd
import numpy as np

from transformers import (
    LlamaForCausalLM, LlamaTokenizer
)
from peft import PeftModel
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from llama_finetune import (
    get_crystal_string,   
    MAX_LENGTH
)
from templating import make_swap_table

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

def prepare_model_and_tokenizer(args):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)
    
    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")
    
    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
def unconditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    prompts = []
    for _ in range(args.num_samples):
        prompt = "Below is a description of a bulk material. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < args.num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif")
            except Exception as e:
                print(e)
                continue

            outputs.append({
                "gen_str": gen_str,
                "cif": cif_str,
                "model_name": args.model_name,
            })

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

condition_templates = {
    "pretty_formula": "The chemical formula is {pretty_formula}. ",
    "e_above_hull": "The energy above the convex hull is {e_above_hull}. ",
    "spacegroup.number": "The spacegroup number is {spacegroup.number}. ",
}

def conditional_sample(args):
    model, tokenizer = prepare_model_and_tokenizer(args)

    conditions_data = pd.read_csv(args.conditions_file)[
        ["e_above_hull", "pretty_formula", "spacegroup.number"]
    ].drop_duplicates()
    conditions_data = conditions_data.sample(args.num_samples, replace=False).to_dict(orient="records")

    conditions = args.conditions.split(",")

    prompts = []
    for d in conditions_data:
        prompt = "Below is a description of a bulk material. "
        for c in conditions:
            prompt += condition_templates[c].format(**d)

        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < args.num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+args.batch_size]
        batch_conditions = conditions[len(outputs):len(outputs)+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=args.temperature, 
            top_p=args.top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, _conditions in zip(gen_strs, batch_prompts, batch_conditions):
            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif") #double check valid cif string
            except Exception as e:
                print(e)
                continue

            sample = {
                "gen_str": gen_str,
                "cif": cif_str,
                "model_name": args.model_name,
            }
            sample.update(_conditions)
            outputs.append(sample)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

def infill_sample(args, start_crystal_cif=None):
    model, tokenizer = prepare_model_and_tokenizer(args)

    if start_crystal_cif is None:
        df = pd.read_csv(args.infill_file)
        idx = np.random.randint(len(df))
        start_crystal_cif = df['cif_str'][idx]

    print("Start crystal cif:")
    print(start_crystal_cif)

    prompts = []
    species_to_remove_list = []
    masked_crystal_strs = []
    for _ in range(args.num_samples):

        prompt = (
            'Below is a partial description of a bulk material where one '
            'element has been replaced with the string "[MASK]":\n'
        )

        structure = Structure.from_str(start_crystal_cif, fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)
        species_to_remove_list.append(species_to_remove)

        crystal_string = get_crystal_string(start_crystal_cif)

        partial_crystal_str = crystal_string.replace(
            species_to_remove, "[MASK]"
        )
        masked_crystal_strs.append(partial_crystal_str)

        prompt = prompt + partial_crystal_str + "\n"

        prompt += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        prompts.append(prompt)
 
    assert args.batch_size == 1, "Batch size must be 1 for infill sampling"

    swap_table = make_swap_table(args.infill_constraint_tolerance)

    outputs = []
    for i in range(0, args.num_samples, args.batch_size):
        batch_prompts = prompts[i:i+args.batch_size]
        species_to_remove_batch = species_to_remove_list[i:i+args.batch_size]
        masked_crystals = masked_crystal_strs[i:i+args.batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        possible_elems = [str(s) for s in swap_table[species_to_remove_batch[0]]]

        kwargs = {
            "do_sample": True,
            "max_new_tokens": 10,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        if args.infill_do_constraint:
            kwargs["bad_words_ids"] = [tokenizer.encode(s) for s in possible_elems]

        generate_ids = model.generate(
            **batch,
            **kwargs,
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, species_to_remove, masked_crystal in zip(
            gen_strs, batch_prompts, species_to_remove_batch, masked_crystals
        ):
            new_element = gen_str.replace(prompt, "").split("\n")[0]
            
            print(f"Swap {species_to_remove} with {new_element}")

            orig_crys_str = masked_crystal.replace("[MASK]", species_to_remove)
            new_crys_str = masked_crystal.replace("[MASK]", new_element)

            try:
                new_cif = parse_fn(new_crys_str)
                _ = Structure.from_str(new_cif, fmt="cif") #double check valid cif string
                original_cif = parse_fn(orig_crys_str)
            except Exception as e:
                print(e)
                continue

            sample = {
                "original_element": species_to_remove,
                "new_element": new_element,
                "original_crystal": original_cif,
                "new_crystal": new_cif,
                "model_name": args.model_name,
            }
            outputs.append(sample)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="llm_samples.csv")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--conditions", type=str, default="pretty_formula")
    parser.add_argument("--conditions_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_do_constraint", type=int, default=0)
    parser.add_argument("--infill_constraint_tolerance", type=float, default=0.1)
    args = parser.parse_args()

    if ".csv" in args.out_path:
        out_path = args.out_path
    else:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(args.out_path, f"samples_{i}.csv") 
        args.out_path = out_path

    if args.conditions_file:
        conditional_sample(args)
    elif args.infill_file:
        infill_sample(args)
    else:
        unconditional_sample(args)
