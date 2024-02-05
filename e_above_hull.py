"""
Build a PatchedPhaseDiagram from all MP ComputedStructureEntries for calculating
DFT-ground truth convex hull energies.
"""

import warnings
import tempfile

from pymatgen.core import Structure
import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PatchedPhaseDiagram
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
from tqdm import tqdm

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.sets import MPRelaxSet

import tqdm
import pandas as pd
from m3gnet.models import Relaxer
from pymatgen.core.structure import Structure

from crystal import cif_str_to_crystal

def m3gnet_relaxed_energy(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    relaxer = Relaxer()  # This loads the default pre-trained model
    relax_results = relaxer.relax(structure)
    final_structure = relax_results['final_structure']
    final_energy_per_atom = float(relax_results['trajectory'].energies[-1])
    return final_energy_per_atom, final_structure

from timeout_decorator import timeout

@timeout(30)
def call_m3gnet_relaxed_energy(cif_str):
    return m3gnet_relaxed_energy(cif_str)

def label_energies(filename):
    df = pd.read_csv(filename)
    
    new_df = []
    for r in tqdm.tqdm(df.to_dict('records')):
        cif_str = r['cif']

        crystal = cif_str_to_crystal(cif_str)
        if crystal is None or not crystal.valid:
            continue

        structure = Structure.from_str(cif_str, fmt="cif")
        if len(structure) == 1:
            continue

        try:
            e, relaxed_s = call_m3gnet_relaxed_energy(cif_str)
            r['m3gnet_relaxed_energy'] = e
            r['m3gnet_relaxed_cif'] = relaxed_s.to(fmt="cif")
        except Exception as e:
            continue

        new_df.append(r)

    new_df = pd.DataFrame(new_df)
    new_filename = filename.replace(".csv","") + "_m3gnet_relaxed_energy.csv"
    new_df.to_csv(new_filename)






def generate_CSE(structure, m3gnet_energy):
    # Write VASP inputs files as if we were going to do a standard MP run
    # this is mainly necessary to get the right U values / etc
    b = MPRelaxSet(structure)
    with tempfile.TemporaryDirectory() as tmpdirname:
        b.write_input(f"{tmpdirname}/", potcar_spec=True)
        poscar = Poscar.from_file(f"{tmpdirname}/POSCAR")
        incar = Incar.from_file(f"{tmpdirname}/INCAR")
        clean_structure = Structure.from_file(f"{tmpdirname}/POSCAR")

    # Get the U values and figure out if we should have run a GGA+U calc
    param = {"hubbards": {}}
    if "LDAUU" in incar:
        param["hubbards"] = dict(zip(poscar.site_symbols, incar["LDAUU"]))
    param["is_hubbard"] = (
        incar.get("LDAU", True) and sum(param["hubbards"].values()) > 0
    )
    if param["is_hubbard"]:
        param["run_type"] = "GGA+U"

    # Make a ComputedStructureEntry without the correction
    cse_d = {
        "structure": clean_structure,
        "energy": m3gnet_energy,
        "correction": 0.0,
        "parameters": param,
    }

    # Apply the MP 2020 correction scheme (anion/+U/etc)
    cse = ComputedStructureEntry.from_dict(cse_d)
    _ = MaterialsProject2020Compatibility(check_potcar=False).process_entries(
        cse,
        clean=True,
    )

    # Return the final CSE (notice that the composition/etc is also clean, not things like Fe3+)!
    return cse

def get_e_above_hull(fn):
    data_path = f"/private/home/ngruver/ocp-modeling-dev/llm/2023-07-13-mp-computed-structure-entries.json.gz"

    print(f"Loading MP ComputedStructureEntries from {data_path}")
    df = pd.read_json(data_path)

    #filter to only df entries that contain the substring "GGA" in the column 'index'
    df = df[df['index'].str.contains("GGA")]
    print(len(df))

    mp_computed_entries = [ComputedEntry.from_dict(x) for x in tqdm(df.entry) if 'GGA' in x['parameters']['run_type']]
    mp_computed_entries = [entry for entry in mp_computed_entries if not np.any(['R2SCAN' in a.name for a in entry.energy_adjustments])]

    ppd_mp = PatchedPhaseDiagram(mp_computed_entries, verbose=True)

    df = pd.read_csv(fn)

    new_df = []
    for d in tqdm(df.to_dict(orient="records")):
        try:
            structure = Structure.from_str(d["m3gnet_relaxed_cif"], fmt="cif")
            energy = d["m3gnet_relaxed_energy"]

            cse = generate_CSE(structure, energy)
            e_above_hull = ppd_mp.get_e_above_hull(cse, allow_negative=True)

            d["e_above_hull"] = e_above_hull
            new_df.append(d)
        except Exception as e:
            print(e)
            continue

    new_df = pd.DataFrame(new_df)
    new_df.to_csv("e_above_hull_llama_70b.csv", index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="data/llm/relaxations/relaxations.csv")
    args = parser.parse_args()

    #suppress tensorflow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    label_energies(args.filename)




    warnings.filterwarnings("ignore")
    get_e_above_hull("")