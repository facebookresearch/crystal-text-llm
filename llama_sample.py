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
from hf_finetune import MAX_LENGTH

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def llama2_model_string(model_size, chat):
    chat = "chat-" if chat else ""
    return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

def parse_fn(gen_str):
    # print(gen_str.split("\n"))
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

def prepare_model_and_tokenizer(model_path, model_name):
    llama_options = model_name.split("-")[1:]
    is_chat = len(llama_options) == 3
    model_size = llama_options[1]
    model_string = llama2_model_string(model_size, is_chat)

    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, model_path, device_map="auto")

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def get_crystal_string(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atom_ids, frac_coords)
        ])

    return crystal_str

def unconditional_sample(
    model_path, 
    model_name,
    num_samples,
    out_path,
    temperature=1.0, 
    top_p=1.0,
    batch_size=1,
):
    model, tokenizer = prepare_model_and_tokenizer(model_path, args.model_name)

    prompts = []
    for _ in range(num_samples):
        prompt = "Below is a description of a bulk material. "
        # prompt += "The chemical formula is YbCuBi. "
        # prompt += "The energy above the convex hull is 0.0. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+batch_size]

        # print(batch_prompts)

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
            # padding=True,
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=temperature, 
            top_p=top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            print(gen_str)

            material_str = gen_str.replace(prompt, "")

            # print(material_str)
            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif") #double check valid cif string
            except Exception as e:
                print(e)
                continue

            print(cif_str)

            outputs.append({
                "gen_str": gen_str,
                "cif": cif_str,
                # "composition_tag": composition_str,
                "model_name": model_name,
            })

    # print(1/0)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

def conditional_sample(
    model_path, 
    model_name,
    num_samples,
    out_path,
    temperature=1.0, 
    top_p=1.0,
    batch_size=1,
):
    model, tokenizer = prepare_model_and_tokenizer(model_path, args.model_name)

    conditions = pd.read_csv("/data/home/ngruver/ocp-modeling-dev/llm/mp_w_tags/test.csv")[
        ["e_above_hull", "pretty_formula", "spacegroup.number"]
    ].drop_duplicates()
    conditions = conditions.sample(num_samples, replace=False).to_dict(orient="records")

    prompts = []
    for d in conditions:
        prompt = "Below is a description of a bulk material. "
        prompt += f"The chemical formula is {d['pretty_formula']}. "
        # prompt += "The energy above the convex hull is 0.0. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+batch_size]
        batch_conditions = conditions[len(outputs):len(outputs)+batch_size]

        # print(batch_prompts)

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
            # padding=True,
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=temperature, 
            top_p=top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt, _conditions in zip(gen_strs, batch_prompts, batch_conditions):
            print(gen_str)

            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif") #double check valid cif string
            except Exception as e:
                print(e)
                continue

            print(cif_str)

            sample = {
                "gen_str": gen_str,
                "cif": cif_str,
                "model_name": model_name,
            }
            sample.update(_conditions)
            outputs.append(sample)

    # print(1/0)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)

def infill_sample(
    model_path, 
    model_name,
    num_samples,
    out_path,
    start_crystal_cif=None,
    temperature=1.0, 
    top_p=1.0,
    batch_size=1,
):
    llama_options = args.model_name.split("-")[1:]
    is_chat = len(llama_options) == 3
    model_size = llama_options[1]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)

    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )
    
    model = PeftModel.from_pretrained(model, model_path, device_map="auto")

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    if start_crystal_cif is None:
        df = pd.read_csv("/private/home/ngruver/ocp-modeling-dev/llm/mp_training_data/train.csv")
        idx = np.random.randint(len(df))
        start_crystal_cif = df['cif_str'][idx]

    print(start_crystal_cif)

    prompts = []
    for _ in range(num_samples):

        prompt = (
            'Below is a partial description of a bulk material where one '
            'element has been replaced with the string "[MASK]":\n'
        )

        structure = Structure.from_str(start_crystal_cif, fmt="cif")
        species = [str(s) for s in structure.species]
        species_to_remove = random.choice(species)

        crystal_string = get_crystal_string(start_crystal_cif)

        partial_crystal_str = crystal_string.replace(
            species_to_remove, "[MASK]"
        )

        prompt = prompt + partial_crystal_str + "\n"

        prompt += (
            "Generate an element that could replace [MASK] in the bulk material:\n"
        )

        prompts.append(prompt)
 
    outputs = []
    for i in range(0, num_samples, batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # print(batch_prompts)

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
            # padding=True,
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        # generate_ids = model.generate(
        #     **batch,
        #     do_sample=True,
        #     max_new_tokens=500,
        #     temperature=temperature, 
        #     top_p=top_p, 
        # )

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=5,
            num_beams=10,
            early_stopping=True
        )

        # print(generate_ids)
        # print(tokenizer.eos_token_id)

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            # print(gen_str)

            new_element = gen_str.replace(prompt, "").split("\n")[0]
            
            print(f"Swap {species_to_remove} with {new_element}")

            # # print(material_str)
            # try:
            #     cif_str = parse_fn(material_str)
            #     _ = Structure.from_str(cif_str, fmt="cif") #double check valid cif string
            # except Exception as e:
            #     print(e)
            #     continue

            # print(cif_str)

            # outputs.append({
            #     "gen_str": gen_str,
            #     "cif": cif_str,
            #     # "composition_tag": composition_str,
            #     "model_name": model_name,
            # })

    # print(1/0)

    df = pd.DataFrame(outputs)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--out_path", type=str, default="llm_samples.csv")
    # parser.add_argument("--sample_per_condition", type=int, default=)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--w_conditions", type=int, default=0)
    args = parser.parse_args()

    if ".csv" in args.out_path:
        out_path = args.out_path
    else:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(args.out_path, f"samples_{i}.csv") 

    if args.w_conditions:
        conditional_sample(
            args.model_path, 
            args.model_name,
            args.num_samples,
            out_path,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        unconditional_sample(
            args.model_path, 
            args.model_name,
            args.num_samples,
            out_path,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # infill_sample(
    #     args.model_path, 
    #     args.model_name,
    #     args.num_samples,
    #     out_path,
    #     temperature=args.temperature,
    #     top_p=args.top_p,
    # )