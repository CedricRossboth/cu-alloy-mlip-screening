import argparse
import logging
import multiprocessing as mp
import os
import pickle
from random import sample
import numpy as np
import pandas as pd
from ase.io import read
from ase.optimize import BFGS, LBFGS
import matplotlib.pyplot as plt
import torch
from time import time

from fairchem.core.common.utils import setup_logging
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.data.oc.core import Slab, Adsorbate, Bulk, AdsorbateSlabConfig
from fairchem.data.oc.utils import DetectTrajAnomaly
from fairchem.data.oc.utils.vasp import write_vasp_input_files

from ts_calc import Reaction
from ts_calc import OCPNEB
from ts_calc.autoframe import AutoFrameCoulping, is_edge_list_respected
from data import ADSORBATE_PKL_PATH, BULK_PKL_PATH, COUPLING_REACTION_DB_PATH


class SafeBFGS(BFGS):
    def step(self):
        # Perform the normal BFGS step
        super().step()

        # Check the maximum force (fmax) after the step
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1)).max()

        if np.isnan(fmax) or np.isinf(fmax) or fmax == 0:
            print("NaN or Inf or 0 detected in forces! Stopping optimization.")
            self.max_steps = self.nsteps

class SafeLBFGS(LBFGS):
    def step(self):
        # Perform the normal LBFGS step
        super().step()

        # Check the maximum force (fmax) after the step
        forces = self.atoms.get_forces()
        fmax = np.sqrt((forces**2).sum(axis=1)).max()

        if np.isnan(fmax) or np.isinf(fmax):
            print("NaN or Inf detected in forces! Stopping optimization.")
            self.max_steps = self.nsteps

#Reactions = ["*CH*CH + *CH*CH -> *CHCHCH*CH", "*CH*CH + *CHCH2 -> *CHCHCHCH2", "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2"]
Reactions = [#"*CH + *CH -> *CH*CH", 
            "*CH2 + *CH -> *CH2CH", "*CH3 + *CH -> *CH3CH"]
Distances_dict = {"*CH*CH + *CH*CH -> *CHCHCH*CH": [(5.0, 3.0)], 
                "*CH*CH + *CHCH2 -> *CHCHCHCH2": [(6.0, 4.0)],
                "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2": [(8.0, 6.0)],
                "*CH + *CH -> *CH*CH": [(4.0, 2.5)],
                "*CH2 + *CH -> *CH2CH": [(4.0, 3.0)],
                "*CH3 + *CH -> *CH3CH": [(4.0, 3.0)]}

Products_dict = {
    "*CH + *CH -> *CH*CH": "C2H2",
    "*CH2 + *CH -> *CH2CH": "C2H3",
    "*CH3 + *CH -> *CH3CH": "C2H4",
    "*CH*CH + *CH*CH -> *CHCHCH*CH": "C4H4",
    "*CH*CH + *CHCH2 -> *CHCHCHCH2": "C4H5",
    "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2": "C4H6"
}
interpolation_method = "idpp"
idpp = True  # whether to use idpp for NEB initial path generation

#Optional
# from x3dase.x3d import X3D
# from x3dase.visualize import view_x3d_n

#Set random seed
#import numpy as np
#np.random.seed(42) # originally # np.random.seed(22)


def parse_neb_info(neb_frames: list, calc, conv: bool, fig_path: str):
    """
    At the conclusion of the ML NEB, this function processes the important
    results and adds them to the entry dictionary.

    Args:
        neb_frames (list[ase.Atoms]): the ML relaxed NEB frames
        calc: the ocp ase Atoms calculator
        conv (bool): whether or not the NEB achieved forces below the threshold within
            the number of allowed steps
    """
    def title_energy_profile(path, activation_energy):
        import re
        pattern = r"results/([A-Za-z0-9]+)/bulk_([A-Za-z0-9]+)/slab_\d+_(\d+)(?:_([a-zA-Z]+))?/\d+\.png"
        match = re.search(pattern, path)
        if match:
            pathway, composition, miller, position = match.groups()
            if position is None:
                position = "top"  # default if missing

            # Convert numbers to subscript for LaTeX
            def latex_subscript(text):
                return ''.join([c if not c.isdigit() else f"$_{c}$" for c in text])

            pathway_latex = latex_subscript(pathway)
            composition_latex = latex_subscript(composition)

            return f"Energy profile for the {position} ({miller}) {composition_latex} surface: {pathway_latex} pathway\nActivation energy: {activation_energy:.2f} eV"
        return None

    def get_activation_energy_and_indices(es):
        def local_extrema(arr):
            arr = np.array(arr)
            local_mins = []
            local_maxs = []

            for i in range(1, len(arr)-1):
                if arr[i] < arr[i-1] and arr[i] <= arr[i+1]:
                    local_mins.append(i)
                elif arr[i] > arr[i-1] and arr[i] >= arr[i+1]:
                    local_maxs.append(i)
            
            # Handle plateaus at the end
            if arr[-1] < arr[-2]:
                local_mins.append(len(arr)-1)
            elif arr[-1] > arr[-2]:
                local_maxs.append(len(arr)-1)
            
            # Optionally include first element if needed
            if arr[0] < arr[1]:
                local_mins.insert(0, 0)
            elif arr[0] > arr[1]:
                local_maxs.insert(0, 0)
            
            return np.array(local_mins), np.array(local_maxs)
        # Find local maximas and minimas in energy profile
        min_energy_local_indices, max_energy_local_indices = local_extrema(es)
        
        # Holds the largest energy barrier from a local minima from all minimas.
        max_energy_from_local_minimas = 0
        if min_energy_local_indices.size > 0:
            # Loops over the local minimas to see which has the biggest energy barrier to cross.
            for min_energy_local_index in min_energy_local_indices:
                # Find closest local maximum to the current local minima considered. By default -1.
                max_energy_local_index = len(es)-1
                if max_energy_local_indices.size > 0:
                    for local_maxima in max_energy_local_indices: 
                        if local_maxima - min_energy_local_index > 0:
                            max_energy_local_index = local_maxima
                            break

                # finds the energy barrier by iterating between each energy point from the local minima to the closest local maximum.
                activation_energy_from_local_minima = es[max_energy_local_index] - es[min_energy_local_index]
                
                # Finds the largest barrier caused by internal barriers
                if activation_energy_from_local_minima > max_energy_from_local_minimas:
                    max_energy_from_local_minimas = activation_energy_from_local_minima
                    min_index = min_energy_local_index
                    max_index = max_energy_local_index

        # Defines the energy barrier as either the largest value or the largest jump from a low lying energy.
        activation_energy = max(max(es), max_energy_from_local_minimas)

        if max(es) == activation_energy:
            min_index = 0
            max_index = np.argmax(es)
 
        return activation_energy, min_index, max_index


    e_along_traj = []
    for frame in neb_frames:
        frame.calc = calc
        e_along_traj.append(frame.get_potential_energy())

    # Plot the reaction coordinate
    e_along_traj = [e - e_along_traj[0] for e in e_along_traj] 

    barrier_height, i_min, i_max = get_activation_energy_and_indices(e_along_traj)

    # plot horizontal lines that show where the energy barrier is considered.
    plt.plot(e_along_traj, marker='o', markersize=4, color='k')
    # Get current x-limits after plotting
    xmin, xmax = plt.gca().get_xlim()
    plt.xlim(xmin, xmax)

    if i_min == i_max:
        plt.hlines(y=e_along_traj[i_min], xmin=xmin, xmax=i_min, linestyles='--', colors='blue', label='Barrierless')
    else:
        # Horizontal line at minimum energy
        plt.hlines(y=e_along_traj[i_min], xmin=xmin, xmax=i_min, linestyles='--', colors='blue', label='lower bound of energy barrier')
        # Horizontal line at activation energy level
        plt.hlines(y=e_along_traj[i_max], xmin=xmin, xmax=i_max, linestyles='--', colors='red', label='upper bound of energy barrier')

    plt.xlabel('Reaction coordinate')
    plt.ylabel('Energy (eV)')
    plt.title(title_energy_profile(fig_path, barrier_height))
    plt.legend()
    plt.gca().set_facecolor('#f9f9f9')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.savefig(fig_path)
    plt.close()

    E_rxn = e_along_traj[-1] - e_along_traj[0]

    if barrier_height <= 0.1 or barrier_height <= E_rxn + 0.1:
        barrierless = True
        ts_idx = None
    else:
        barrierless = False
        ts_idx = e_along_traj.index(max(e_along_traj))

    results = {}
    results["E_a_ml"] = barrier_height
    results["E_rxn_ml"] = E_rxn
    results["converged_ml"] = conv
    results["barrierless_ml"] = barrierless
    results["transition_state_idx"] = ts_idx

    return results


def ts_calc_one_bulk(bulk, reaction, calc, args):
    logging.info(f"Start NEB calc for bulk {bulk.atoms.symbols}")
    reaction_str = reaction.reaction_str_from_db
    if reaction_str in Products_dict:
        product = Products_dict[reaction_str]
    else:
        product = f"unknown_product_{reaction.reaction_str_from_db}"
    bulk_path = os.path.join(args.output_dir, f"{product}/bulk_{bulk.atoms.symbols}")
    slab_pkl_path = os.path.join(bulk_path, "slabs.pkl")
    os.makedirs(bulk_path, exist_ok=True)

    # check whether this bulk was finished
    if os.path.exists(os.path.join(bulk_path, "bulk_results.csv")):
        df1 = pd.read_csv(os.path.join(bulk_path, "bulk_results.csv"))
        df2 = pd.read_csv(os.path.join(bulk_path, "bulk_summary.csv"))
        bulk_results = df1.to_dict(orient='records')
        bulk_summary = df2.to_dict(orient='records')
        logging.warning(f"Read previously saved results of bulk {bulk.atoms.symbols} in {bulk_path}")
        return bulk_results, bulk_summary

    try:
        all_slabs = pickle.load(open(slab_pkl_path, "rb"))
        logging.info(f"Get {len(all_slabs)} slabs for bulk {bulk.atoms.symbols} from {slab_pkl_path}")
    except:
        # Grab the bulk and cut the slab we are interested in
        slab_111 = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,1,1))
        slab_100 = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,0,0))
        all_slabs = slab_111 + slab_100
        logging.info(f"Get {len(all_slabs)} slabs for bulk {bulk.atoms.symbols}")
        """
        logging.info(f"Get {len(all_slabs)} slabs for bulk {bulk.src_id}")
        if len(all_slabs) > 25:
            all_slabs = bulk.get_slabs(max_miller=2)
            logging.info(f"Get {len(all_slabs)} slabs for bulk {bulk.src_id} at max_miller=2")
            if len(all_slabs) > 25:
                all_slabs = sample(all_slabs, 25)
        """
        with open(slab_pkl_path, 'wb') as f:
            pickle.dump(all_slabs, f)
            f.flush()

    bulk_results = []
    bulk_summary = []
    for idx, slab in enumerate(all_slabs):
        millers = "".join(str(m) for m in slab.millers)
        top = "top" if slab.top else "bottom"
        slab_path = os.path.join(bulk_path, f"slab_{idx:0>3d}_{millers}_{top}")
        os.makedirs(slab_path, exist_ok=True)
        try:
            _results, _summary = ts_calc_one_slab(slab, reaction, calc, slab_path, args)
            print(f"Finished calc of Slab {idx}, millers={millers}, top={slab.top}, bulk formula={str(bulk.atoms.symbols)}", flush=True)
        except Exception as e:
            print(f"Slab {idx}, millers={millers}, top={slab.top}, bulk formula={str(bulk.atoms.symbols)}: {e}", flush=True)
            continue
        bulk_results.extend(_results)
        bulk_summary.append(_summary)
    
    # Process the bulk results
    df1 = pd.DataFrame(bulk_results)
    df2 = pd.DataFrame(bulk_summary)
    df1.to_csv(os.path.join(bulk_path, "bulk_results.csv"), index=False)
    df2.to_csv(os.path.join(bulk_path, "bulk_summary.csv"), index=False)

    return bulk_results, bulk_summary


def ts_calc_one_slab(slab, reaction, calc, opath, args):

    # check whether this slab was finished
    if os.path.exists(os.path.join(opath, "slab_results.csv")):
        df1 = pd.read_csv(os.path.join(opath, "slab_results.csv"))
        df2 = pd.read_csv(os.path.join(opath, "slab_summary.csv"))
        slab_results = df1.to_dict(orient='records')
        summary = df2.to_dict(orient='records')[0]
        logging.warning(f"Read previously saved results of slab in {opath}")
        return slab_results, summary

    # Perform adsorption site enumeration, num_sites=100 in AdsorbML
    reactant1_configs = AdsorbateSlabConfig(
        slab=slab, adsorbate=reaction.reactant1_ads,
        mode=args.placement_mode,
        num_sites=args.num_sites).atoms_list
    reactant2_configs = AdsorbateSlabConfig(
        slab=slab, adsorbate=reaction.reactant2_ads,
        mode=args.placement_mode,
        num_sites=args.num_sites).atoms_list
    product1_configs = AdsorbateSlabConfig(
        slab=slab, adsorbate=reaction.product1_ads,
        mode=args.placement_mode,
        num_sites=args.num_sites).atoms_list

    t2 = time()
    # Relax the reactant1 systems

    reactant1_energies = []
    for idx, config in enumerate(reactant1_configs):
        try:
            config.calc = calc
            opt = SafeBFGS(config, logfile=None)
            opt.run(fmax=args.fmax, steps=args.opt_max_steps)
            reactant1_energies.append(config.get_potential_energy())
        except Exception as e:
            print(f"Error relaxing reactant1 config: {e}", flush=True)
        if idx % 10 == 0:
            print(f"Relaxed {idx} reactant1 configurations", flush=True)
    print(f"Relaxed {len(reactant1_configs)} reactant1 configurations", flush=True)

    # Relax the reactant2 systems
    reactant2_energies = []
    for idx, config in enumerate(reactant2_configs):
        try:
            config.calc = calc
            opt = SafeBFGS(config, logfile=None)
            opt.run(fmax=args.fmax, steps=args.opt_max_steps)
            reactant2_energies.append(config.get_potential_energy())
        except Exception as e:
            print(f"Error relaxing reactant2 config: {e}", flush=True)
        if idx % 10 == 0:
            print(f"Relaxed {idx} reactant2 configurations", flush=True)
    print(f"Relaxed {len(reactant2_configs)} reactant2 configurations", flush=True)

    # Relax the product1 systems
    product1_energies = []
    new_product1_configs = []
    for idx, config in enumerate(product1_configs):
        try:
            #ini_cf = config.copy()
            config.calc = calc
            opt = SafeBFGS(config, logfile=None)
            #converged = 
            opt.run(fmax=args.fmax, steps=args.opt_max_steps)
            """
            # Check for anomolous behavior
            dt = DetectTrajAnomaly(
                ini_cf,
                config,
                config.get_tags(),
            )
            detector_bool = all(
                [
                    not dt.is_adsorbate_intercalated(),
                    not dt.is_adsorbate_desorbed(),
                    not dt.has_surface_changed(),
                ]
            )

            if (
                converged
                and is_edge_list_respected(config, reaction.edge_list_final)
                and detector_bool
            ):
                product1_energies.append(config.get_potential_energy())
                new_product1_configs.append(config.copy())
            """
            product1_energies.append(config.get_potential_energy())
        except Exception as e:
            print(f"Error relaxing product1 config: {e}", flush=True)
        if idx % 10 == 0:
            print(f"Relaxed {idx} product1 configurations", flush=True)
    print(f"Relaxed {len(product1_configs)} product1 configurations", flush=True)


    with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
        f.write(f"Total time for all relaxations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}: {time() - t2:.2f}s\n")
        f.flush()
    print(f"Total time for all relaxations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}: {time() - t2:.2f}s", flush=True)

    dist_max, r_react_max = Distances_dict[reaction.reaction_str_from_db][0]

    # Enumerate NEB initial structures
    af = AutoFrameCoulping(
        reaction = reaction,
        reactant1_systems = reactant1_configs,
        reactant1_energies = reactant1_energies,
        reactant2_systems = reactant2_configs,
        reactant2_energies = reactant2_energies,
        product_systems = product1_configs,
        product_energies = product1_energies,
        dist_max=dist_max,
        r_react_max=r_react_max,
        r_react_min=1,
    )

    t3 = time()
    frame_sets, mapping_idxs = af.get_neb_frames(
        calc, n_frames=args.neb_images,
        n_initial_frames=5, n_final_frames_per_initial=1,
        idpp=idpp
    )
    with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
        f.write(f"""Time for initial NEB frames collection for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}, {interpolation_method}, dist max = {dist_max} and r react max = {r_react_max}: {time() - t3:.2f}s
{len(frame_sets)} initial NEB paths obtained.\n""")
        f.flush()
    print(f"""Time for initial NEB frames collection for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}, {interpolation_method}, dist max = {dist_max} and r react max = {r_react_max}: {time() - t3:.2f}s
{len(frame_sets)} initial NEB paths obtained.""", flush=True)
    
    t4 = time()
    slab_results, summary = [], {}
    try:
        # Run NEB calculations
        if len(frame_sets) == 0:
            with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {args.neb_k}: 0s
0 optimized NEB paths obtained, 0 converged.\n""")
                f.flush()
            print(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {args.neb_k}: 0s
0 optimized NEB paths obtained, 0 converged.\n""", flush=True)
            traj_path = os.path.join(opath, "0.traj")
            _results = {}
            _results["E_a_ml"] = np.nan
            _results["E_rxn_ml"] = np.nan
            _results["converged_ml"] = False
            _results["barrierless_ml"] = None
            _results["transition_state_idx"] = np.nan

            _results['bulk_comp'] = slab.bulk.atoms.symbols
            #_results['bulk_mpid'] = slab.bulk.src_id
            _results['slab_millers'] = " ".join(str(m) for m in slab.millers)
            _results['slab_shift'] = slab.shift
            _results['slab_top'] = slab.top
            _results['neb_id'] = 0
            _results['neb_traj'] = traj_path
            slab_results.append(_results)
            summary = slab_results[0]
            df1 = pd.DataFrame(slab_results)
            df2 = pd.DataFrame([summary])
            df1.to_csv(os.path.join(opath, "slab_results.csv"), index=False)
            df2.to_csv(os.path.join(opath, "slab_summary.csv"), index=False)

            return slab_results, summary
        
        else:
            for idx, frame_set in enumerate(frame_sets):
                torch.cuda.empty_cache()
                traj_path = os.path.join(opath, f"{idx}.traj")
                fig_path = os.path.join(opath, f"{idx}.png")

                try:
                    neb = OCPNEB(
                        frame_set,
                        checkpoint_path=args.ckpt_path,
                        k=args.neb_k,
                        batch_size=args.neb_batch_size,
                        cpu=args.cpu,
                    )
                    optimizer = SafeLBFGS(neb, trajectory=traj_path)
                    conv = optimizer.run(fmax=args.fmax + args.delta_fmax_climb, steps=200)
                    if conv:
                        print(f"Climbing the NEB with fmax = {args.fmax}")
                        neb.climb = True
                        conv = optimizer.run(fmax=args.fmax, steps=300)
                    if conv:
                        print("The NEB has succesfully converged.")
                    _results = parse_neb_info(frame_set, calc, conv, fig_path)

                except RuntimeError:
                    logging.warning(f"RuntimeError in NEB {idx} — possibly NaN forces")
                    conv = False  # ensure conv exists
                    _results = parse_neb_info(frame_set, calc, conv, fig_path)
                except Exception as e:
                    logging.warning(f"Error with NEB {idx}: {e}")
                    _results = {}
                    _results["E_a_ml"] = np.nan
                    _results["E_rxn_ml"] = np.nan
                    _results["converged_ml"] = False
                    _results["barrierless_ml"] = None
                    _results["transition_state_idx"] = np.nan

                _results['bulk_comp'] = slab.bulk.atoms.symbols
                #_results['bulk_mpid'] = slab.bulk.src_id
                _results['slab_millers'] = " ".join(str(m) for m in slab.millers)
                _results['slab_shift'] = slab.shift
                _results['slab_top'] = slab.top
                _results['neb_id'] = idx
                _results['neb_traj'] = traj_path
                slab_results.append(_results)

        # get the NEB with lowest energy barrier
        e_list = [_r["E_a_ml"] for _r in slab_results]
        e_list = np.array(e_list)
        try:
            summary = slab_results[np.nanargmin(e_list)]

        except Exception as e:
            logging.error(f"Error occurred while processing slab results: {e}")

        # Process the slab results
        df1 = pd.DataFrame(slab_results)
        df2 = pd.DataFrame([summary])
        df1.to_csv(os.path.join(opath, "slab_results.csv"), index=False)
        df2.to_csv(os.path.join(opath, "slab_summary.csv"), index=False)
        valid_count = len(df1[~np.isnan(df1['E_a_ml'])])
        converged_count = len(df1[df1['converged_ml']])

        with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
            f.write(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}: {time() - t4:.2f}s
{valid_count} optimized NEB paths obtained, {converged_count} converged.\n""")
            f.flush()
        print(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, top={slab.top}: {time() - t4:.2f}s
{valid_count} optimized NEB paths obtained, {converged_count} converged.""", flush=True)
    except Exception as e:
        logging.error(f"Error occurred while processing NEB results: {e}")

    return slab_results, summary


def parse_args():
    parser = argparse.ArgumentParser(description="Sample adsorbate and bulk surface(s)")

    # input databases
    parser.add_argument(
        "--bulk_db",
        type=str,
        default=BULK_PKL_PATH,
        help="Underlying db for bulks (.pkl)"
    )

    parser.add_argument(
        "--adsorbate_db",
        type=str,
        default=ADSORBATE_PKL_PATH,
        help="Underlying db for adsorbates (.pkl)",
    )

    parser.add_argument(
        "--reaction_db",
        type=str,
        default=COUPLING_REACTION_DB_PATH,
        help="Underlying db for adsorbates (.pkl)",
    )

    # MLIP model and TS calculation args
    parser.add_argument(
        "--model",
        type=str,
        default="EquiformerV2-31M-S2EF-OC20-All+MD",
        help="The name of pretrained model for TS calculations"
    )

    parser.add_argument(
        "--model_ckpt_dir",
        type=str,
        default="./ocp_checkpoints",
        help="The path for model checkpoints"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Using CPU to run TS calculations",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="The fmax for opt and neb calculations"
    )

    parser.add_argument(
        "--opt_max_steps",
        type=int,
        default=200,
        help="The max steps for opt and neb calculations"
    )

    parser.add_argument(
        "--num_sites",
        type=int,
        default=50,
        help="for adsorption site enumeration"
    )

    parser.add_argument(
        "--max_miller",
        type=int,
        default=2,
        help="Max miller indices to consider for generating surfaces",
    )

    parser.add_argument(
        "--placement_mode",
        type=str,
        default="random_site_heuristic_placement",
        choices=["random", "heuristic", "random_site_heuristic_placement"],
        help="The method for surface site sampling and adsorbate placement on each site",
    )

    parser.add_argument(
        "--neb_images",
        type=int,
        default=10,
        help="Number of NEB interpolation points (images)"
    )

    parser.add_argument(
        "--neb_k",
        type=float,
        default=1.0,
        help="NEB spring constant(s) in eV/Ang."
    )

    parser.add_argument(
        "--neb_batch_size",
        type=int,
        default=10,
        help="batch size for NEB calculations"
    )

    parser.add_argument(
        "--delta_fmax_climb",
        type=float,
        default=0.4,
        help="initial fmax for NEB calculations"
    )

    # specification option A: provide one set of indices
    parser.add_argument(
        "--reaction_index", type=int, default=0, help="Reaction index (int)"
    )
    parser.add_argument(
        "--reaction_str", type=str,
        default=None,
        help="Reaction string"
    )
    parser.add_argument(
        "--bulk_mpid",
        type=str,
        default=None,
        help="Material project ID of Bulk",
    )

    # specification option B: provide one set of indices
    parser.add_argument(
        "--bulk_indices_file",
        type=str,
        default=None,
        help="file containing bulk indices (mpid)",
    )

    parser.add_argument(
    "--bulk_atoms_file",
    type=str,
    default=None,
    help="file containing bulk atoms (mpid)",
)

    # output
    parser.add_argument("--output_dir", type=str, help="Root directory for outputs")

    # other options

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling/random sites generation",
    )
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Random seed for sampling/random sites generation",
    )
    parser.add_argument(
        "--no_vasp",
        action="store_true",
        default=False,
        help="Do not write out POTCAR/INCAR/KPOINTS for adslabs",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Log detailed info"
    )
    # parser.add_argument(
    #     "--workers",
    #     type=int,
    #     default=10,
    #     help="Number of workers for multiprocessing when given a file of indices",
    # )

    args = parser.parse_args()

    # check that all needed args are supplied
    # if args.indices_file:
    #     if not (args.random_placements or args.heuristic_placements):
    #         parser.error("Must choose either or both of random or heuristic placements")
    #     if args.random_placements and (
    #         args.random_sites is None or args.random_sites <= 0
    #     ):
    #         parser.error("Must specify number of sites for random placements")
    # elif args.bulk_indices_file:
    #     assert args.precomputed_slabs_dir is not None
    # else:
    #     if (
    #         args.adsorbate_index is None
    #         or args.bulk_index is None
    #         or args.surface_index is None
    #     ):
    #         parser.error("Must provide a file or specify all material indices")

    return args


if __name__ == "__main__":
    """
    This script creates adsorbate+surface placements and saves them out. An
    indices_file is required, which contains the adsorbate, bulk, and surface
    index desired for placement. Alternatively, if a bulk_indices_file is
    provided with bulk indices, slabs will be precomputed and saved to the
    provided directory.
    """
    t_start = time()
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = model_name_to_local_file(args.model, local_cache=args.model_ckpt_dir)
    args.ckpt_path = ckpt_path
    calc = OCPCalculator(checkpoint_path=ckpt_path, cpu=args.cpu, seed=args.seed)

    all_results = []
    all_summary = []
    for reaction_str in Reactions:
        reaction = Reaction(
            reaction_db_path=args.reaction_db,
            adsorbate_db_path=args.adsorbate_db,
            reaction_str_from_db=reaction_str,
        )

        if args.bulk_atoms_file:
            # with open(args.bulk_indices_file, "r") as f:
            #     bulk_indices = f.read().splitlines()
            with open(args.bulk_atoms_file, "rb") as f:
                bulk_atoms_list = pickle.load(f)
            print(f"Running bulks from {args.bulk_atoms_file} for the reaction {reaction_str}......", flush=True)
            with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"Running bulks from {args.bulk_atoms_file} for the reaction {reaction_str}......\n")
                f.flush()

            t0 = time()
            for bulk_data in bulk_atoms_list:
                bulk = Bulk(bulk_atoms=bulk_data["atoms"])
                try:
                    t1 = time()
                    bulk_results, bulk_summary = ts_calc_one_bulk(bulk, reaction, calc, args)
                    with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
                        f.write(f"Finished bulk = {str(bulk.atoms.symbols)} in {time() - t1:.2f}s\n\n")
                        f.flush()
                    print(f"Finished bulk = {str(bulk.atoms.symbols)} in {time() - t1:.2f}s", flush=True)
                    all_results.extend(bulk_results)
                    all_summary.append(bulk_summary)

                except Exception as e:
                    with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
                        f.write(f"Error with bulk chemical formula={str(bulk.atoms.symbols)}: {e}\n\n")
                        f.flush()
                    print(f"Error with bulk chemical formula={str(bulk.atoms.symbols)}: {e}", flush=True)
                    continue

            with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"Total time for all bulks: {time() - t0:.2f}s\n\n\n")
                f.flush()
            print(f"Total time for all bulks: {time() - t0:.2f}s", flush=True)

            try:
                # Process the all results
                df1 = pd.DataFrame(all_results)
                df2 = pd.DataFrame(all_summary)  
                df1.to_csv(os.path.join(args.output_dir, f"all_results_{args.job_id:0>2d}.csv"), index=False)
                df2.to_csv(os.path.join(args.output_dir, f"all_summary_{args.job_id:0>2d}.csv"), index=False)
            except Exception as e:
                print(all_results, all_summary, type(all_results), type(all_summary), flush=True)
                logging.error(f"Error occurred while processing all results: {e}")

            print(f"Finished! All {len(bulk_atoms_list)} bulks were calculated!", flush=True)

        elif args.bulk_indices_file:
            # with open(args.bulk_indices_file, "r") as f:
            #     bulk_indices = f.read().splitlines()
            bulk_indices = pd.read_csv(args.bulk_indices_file)["bulk_mpid"].to_list()
            logging.info(f"Running bulk idxs from {args.bulk_indices_file}......")

            all_results = []
            all_summary = []
            for _mpid in bulk_indices:
                bulk = Bulk(bulk_src_id_from_db=_mpid, bulk_db_path=args.bulk_db)
                try:
                    bulk_results, bulk_summary = ts_calc_one_bulk(bulk, reaction, calc, args)
                    all_results.extend(bulk_results)
                    all_summary.extend(bulk_summary)
                except:
                    logging.warning(f"Error with bulk mpid={bulk.src_id}")
                    continue

            # Process the all results
            df1 = pd.DataFrame(all_results)
            df2 = pd.DataFrame(all_summary)
            df1.to_csv(os.path.join(args.output_dir, f"all_results_{args.job_id:0>2d}.csv"), index=False)
            df2.to_csv(os.path.join(args.output_dir, f"all_summary_{args.job_id:0>2d}.csv"), index=False)

            logging.info(f"Finished! All {len(bulk_indices)} bulks were calculated!")

        elif args.bulk_mpid:
            bulk = Bulk(bulk_src_id_from_db=args.bulk_mpid, bulk_db_path=args.bulk_db)
            bulk_results, bulk_summary = ts_calc_one_bulk(bulk, reaction, calc, args)
            # Process the all results
            df1 = pd.DataFrame(bulk_results)
            df2 = pd.DataFrame(bulk_summary)
            df1.to_csv(os.path.join(args.output_dir, "bulk_results.csv"), index=False)
            df2.to_csv(os.path.join(args.output_dir, "bulk_summary.csv"), index=False)
            logging.info(f"Finished! Bulk {args.bulk_mpid} was calculated!")
        else:
            print("ERROR")
    with open(f"logs/time_info_{args.job_id:0>2d}.txt", "a") as f:
        f.write(f"Total time for this job: {time() - t_start:.2f}s\n")
        f.flush()
    print(f"Total time for this job: {time() - t_start:.2f}s", flush=True)
