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

        if np.isnan(fmax) or np.isinf(fmax) or fmax == 0:
            print("NaN or Inf or 0 detected in forces! Stopping optimization.")
            self.max_steps = self.nsteps


Initial_neb_method = ['idpp']
Reactions = ["*CH*CH + *CH*CH -> *CHCHCH*CH", "*CH*CH + *CHCH2 -> *CHCHCHCH2", "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2"]
#Distances_dict = {"*CH*CH + *CH*CH -> *CHCHCH*CH": [(5.0, 3.0), (6.0, 4.0), (7.0, 5.0)], "*CH*CH + *CHCH2 -> *CHCHCHCH2": [(6.0, 4.0), (7.0, 5.0), (8.0, 6.0)], "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2": [(8.0, 6.0), (9.0, 7.0), (10.0, 8.0)]}
Distances_dict = {"*CH*CH + *CH*CH -> *CHCHCH*CH": [(5.0, 3.0)], "*CH*CH + *CHCH2 -> *CHCHCHCH2": [(6.0, 4.0)], "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2": [(8.0, 6.0)]}
#neb_ks = [0.1, 0.5, 1.0]
neb_ks = [1.0]


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
    e_along_traj = []
    for frame in neb_frames:
        frame.calc = calc
        e_along_traj.append(frame.get_potential_energy())
    barrier_height = max(e_along_traj) - e_along_traj[0]
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

    # plot neb figure
    es = [e - e_along_traj[0] for e in e_along_traj]
    plt.plot(es)
    plt.xlabel("frame number")
    plt.ylabel("relative energy [eV]")
    plt.title(f"Ea = {max(es):1.2f} eV")
    plt.savefig(fig_path)
    plt.clf()

    return results


def ts_calc_one_bulk(bulk, reaction, calc, args):
    logging.info(f"Start NEB calc for bulk {bulk.atoms.symbols}")
    if reaction_str == "*CH*CH + *CH*CH -> *CHCHCH*CH":
        product = "C4H4"
    elif reaction_str == "*CH*CH + *CHCH2 -> *CHCHCHCH2":
        product = "C4H5"
    elif reaction_str == "*CHCH2 + *CHCH2 -> *CH2*CH*CH*CH2":
        product = "C4H6"
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
        all_slabs = [slab for slab in all_slabs if slab.top]
        logging.info(f"Get {len(all_slabs)} slabs for bulk {bulk.atoms.symbols} from {slab_pkl_path}")
    except:
        # Grab the bulk and cut the slab we are interested in
        slab_111 = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,1,1))
        if len(slab_111) == 2:
            slab_111 = [slab for slab in slab_111 if slab.top]
        slab_100 = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=(1,0,0))
        if len(slab_100) == 2:
            slab_100 = [slab for slab in slab_100 if slab.top]
        all_slabs = slab_111 #+ slab_100
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

    bulk_results = []
    bulk_summary = []
    for idx, slab in enumerate(all_slabs):
        millers = "".join(str(m) for m in slab.millers)
        slab_path = os.path.join(bulk_path, f"slab_{idx:0>3d}_{millers}")
        os.makedirs(slab_path, exist_ok=True)
        try:
            _results, _summary = ts_calc_one_slab(slab, reaction, calc, slab_path, args)
            print(f"Finished calc of Slab {idx}, millers={millers}, bulk formula={str(bulk.atoms.symbols)}", flush=True)
        except Exception as e:
            print(f"Error with Slab {idx}, millers={millers}, bulk formula={str(bulk.atoms.symbols)}: {e}", flush=True)
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
    """
    # check whether this slab was finished
    if os.path.exists(os.path.join(opath, "slab_results.csv")):
        df1 = pd.read_csv(os.path.join(opath, "slab_results.csv"))
        df2 = pd.read_csv(os.path.join(opath, "slab_summary.csv"))
        slab_results = df1.to_dict(orient='records')
        summary = df2.to_dict(orient='records')[0]
        logging.warning(f"Read previously saved results of slab in {opath}")
        return slab_results, summary
    """
    relaxed_structures_path = os.path.join(opath, "relaxed_structures")
    os.makedirs(relaxed_structures_path, exist_ok=True)
    slab_results_all = []
    summary_all = []

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
            if idx % 10 == 0:
                print(f"Relaxed {idx} reactant1 configurations", flush=True)
        except Exception as e:
            print(f"Error with Reactant1 config {idx}: {e}", flush=True)
            continue
    print(f"Relaxed {len(reactant1_configs)} reactant1 configurations", flush=True)
    pickle.dump(reactant1_configs, open(os.path.join(relaxed_structures_path, "reactant1_configs.pkl"), "wb"))
    pickle.dump(reactant1_energies, open(os.path.join(relaxed_structures_path, "reactant1_energies.pkl"), "wb"))

    # Relax the reactant2 systems
    reactant2_energies = []
    for idx, config in enumerate(reactant2_configs):
        try:
            config.calc = calc
            opt = SafeBFGS(config, logfile=None)
            opt.run(fmax=args.fmax, steps=args.opt_max_steps)
            reactant2_energies.append(config.get_potential_energy())
            if idx % 10 == 0:
                print(f"Relaxed {idx} reactant2 configurations", flush=True)
        except Exception as e:
            print(f"Error with Reactant1 config {idx}: {e}", flush=True)
            continue
    print(f"Relaxed {len(reactant2_configs)} reactant2 configurations", flush=True)
    pickle.dump(reactant2_configs, open(os.path.join(relaxed_structures_path, "reactant2_configs.pkl"), "wb"))
    pickle.dump(reactant2_energies, open(os.path.join(relaxed_structures_path, "reactant2_energies.pkl"), "wb"))

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
            if idx % 10 == 0:
                print(f"Relaxed {idx} product1 configurations", flush=True)
        except Exception as e:
            print(f"Error with Reactant1 config {idx}: {e}", flush=True)
            continue
    print(f"Relaxed {len(product1_configs)} product1 configurations", flush=True)
    pickle.dump(product1_configs, open(os.path.join(relaxed_structures_path, "product1_configs.pkl"), "wb"))
    pickle.dump(product1_energies, open(os.path.join(relaxed_structures_path, "product1_energies.pkl"), "wb"))

    with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
        f.write(f"Total time for all relaxations for {str(slab.bulk.atoms.symbols)} {slab.millers}: {time() - t2:.2f}s\n")
    print(f"Total time for all relaxations for {str(slab.bulk.atoms.symbols)} {slab.millers}: {time() - t2:.2f}s", flush=True)

    for interpolation_method in Initial_neb_method:
        interpolation_method_path = os.path.join(opath, f"{interpolation_method}")
        os.makedirs(interpolation_method_path, exist_ok=True)

        if interpolation_method == "idpp":
            idpp = True
        elif interpolation_method == "linear":
            idpp = False

        reactant1_configs = pickle.load(open(os.path.join(relaxed_structures_path, "reactant1_configs.pkl"), "rb"))
        reactant1_energies = pickle.load(open(os.path.join(relaxed_structures_path, "reactant1_energies.pkl"), "rb"))

        reactant2_configs = pickle.load(open(os.path.join(relaxed_structures_path, "reactant2_configs.pkl"), "rb"))
        reactant2_energies = pickle.load(open(os.path.join(relaxed_structures_path, "reactant2_energies.pkl"), "rb"))

        product1_configs = pickle.load(open(os.path.join(relaxed_structures_path, "product1_configs.pkl"), "rb"))
        product1_energies = pickle.load(open(os.path.join(relaxed_structures_path, "product1_energies.pkl"), "rb"))

        for dist_pair in Distances_dict[reaction_str]:
            dist_max, r_react_max = dist_pair
            distance_path = os.path.join(interpolation_method_path, f"{dist_max}_{r_react_max}")
            os.makedirs(distance_path, exist_ok=True)

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
            with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"""Time for initial NEB frames collection for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max} and r react max = {r_react_max}: {time() - t3:.2f}s
{len(frame_sets)} initial NEB paths obtained.\n""")
            print(f"""Time for initial NEB frames collection for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max} and r react max = {r_react_max}: {time() - t3:.2f}s
{len(frame_sets)} initial NEB paths obtained.""", flush=True)
            # If any initial NEB paths obtained follow, else continue for another distance.
            try:
                if len(frame_sets) == 0:
                    for neb_k in neb_ks:
                        with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                            f.write(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {neb_k}: 0s
0 optimized NEB paths obtained, 0 converged.\n""")
                        print(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {neb_k}: 0s
0 optimized NEB paths obtained, 0 converged.\n""", flush=True)
                    continue
                
                initial_neb_path = os.path.join(distance_path, "initial_NEBs.pkl")
                try:
                    with open(initial_neb_path, "wb") as f:
                        pickle.dump(frame_sets, f, protocol=pickle.HIGHEST_PROTOCOL)
                except (OSError, pickle.PickleError) as e:
                    logging.error(f"Failed to save NEB frames to {initial_neb_path}: {e}")
                    raise  # Optional: re-raise or handle gracefully


                for neb_k in neb_ks:
                    neb_k_path = os.path.join(distance_path, f"neb_k_{neb_k}")
                    os.makedirs(neb_k_path, exist_ok=True)
                    initial_neb_path = os.path.join(distance_path, "initial_NEBs.pkl")
                    try:
                        with open(initial_neb_path, "rb") as f:
                            frame_sets = pickle.load(f)
                    except (EOFError, OSError, pickle.PickleError) as e:
                        logging.error(f"Failed to load NEB frames from {initial_neb_path}: {e}")
                        frame_sets = []  # or handle differently

                    # Run NEB calculations
                    t4 = time()
                    slab_results, summary = [], {}
                    for idx, frame_set in enumerate(frame_sets):
                        torch.cuda.empty_cache()
                        traj_path = os.path.join(neb_k_path, f"{idx}.traj")
                        fig_path = os.path.join(neb_k_path, f"{idx}.png")

                        try:
                            neb = OCPNEB(
                                frame_set,
                                checkpoint_path=args.ckpt_path,
                                k=neb_k,
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
                    slab_results_all += slab_results
                    summary_all.append(summary)
                    df1 = pd.DataFrame(slab_results)
                    df2 = pd.DataFrame([summary])
                    df1.to_csv(os.path.join(neb_k_path, "slab_results.csv"), index=False)
                    df2.to_csv(os.path.join(neb_k_path, "slab_summary.csv"), index=False)
                    valid_count = len(df1[~np.isnan(df1['E_a_ml'])])
                    converged_count = len(df1[df1['converged_ml']])

                    with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                        f.write(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {neb_k}: {time() - t4:.2f}s
{valid_count} optimized NEB paths obtained, {converged_count} converged.\n""")
                    print(f"""Total time for all NEB calculations for {str(slab.bulk.atoms.symbols)} {slab.millers}, {interpolation_method}, dist max = {dist_max}, r react max = {r_react_max} and k = {neb_k}: {time() - t4:.2f}s
{valid_count} optimized NEB paths obtained, {converged_count} converged.""", flush=True)
            except Exception as e:
              logging.error(f"Error occurred while processing NEB results: {e}")

    return slab_results_all, summary_all


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
        default=0.5,
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
    args = parse_args()
    setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = model_name_to_local_file(args.model, local_cache=args.model_ckpt_dir)
    args.ckpt_path = ckpt_path
    calc = OCPCalculator(checkpoint_path=ckpt_path, cpu=args.cpu, seed=args.seed)

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
            logging.info(f"Running bulks from {args.bulk_atoms_file} for the reaction {reaction_str}......")
            with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"Running bulks from {args.bulk_atoms_file} for the reaction {reaction_str}......\n")

            all_results = []
            all_summary = []
            t0 = time()
            for bulk_data in bulk_atoms_list[5:6]:
                bulk = Bulk(bulk_atoms=bulk_data["atoms"])
                try:
                    t1 = time()
                    bulk_results, bulk_summary = ts_calc_one_bulk(bulk, reaction, calc, args)
                    with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                        f.write(f"Finished bulk = {str(bulk.atoms.symbols)} in {time() - t1:.2f}s\n\n")
                    print(f"Finished bulk = {str(bulk.atoms.symbols)} in {time() - t1:.2f}s", flush=True)
                    all_results.extend(bulk_results)
                    all_summary.extend(bulk_summary)

                except:
                    with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                        f.write(f"Error with bulk chemical formula={str(bulk.atoms.symbols)}\n")
                    logging.warning(f"Error with bulk chemical formula={str(bulk.atoms.symbols)}")
                    continue

            with open(f"logs_test/time_info_{args.job_id:0>2d}.txt", "a") as f:
                f.write(f"Total time for all bulks: {time() - t0:.2f}s\n\n\n")
            print(f"Total time for all bulks: {time() - t0:.2f}s", flush=True)

            # Process the all results
            df1 = pd.DataFrame(all_results)
            df2 = pd.DataFrame(all_summary)
            df1.to_csv(os.path.join(args.output_dir, f"all_results_{args.job_id:0>2d}.csv"), index=False)
            df2.to_csv(os.path.join(args.output_dir, f"all_summary_{args.job_id:0>2d}.csv"), index=False)

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
