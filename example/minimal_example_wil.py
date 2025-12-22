#  uv run example/minimal_example_wil.py
"""
VERIFIED FOR VERSION 2.1.3

Minimal example script to demonstrate functionality of py-xl-sindy package.

This script generates synthetic data from a mujoco simulation of a cartpole.
It then uses the py-xl-sindy package to identify a model from the data.

The identified model is then simulated and compared to the original data.

This minimal example is a bit more modular allowing :
- two types of catalog (Lagrange and Classical)
- different regression methods
- different system and friction

"""

import numpy as np
import sympy as sp 
import xlsindy
from generate_trajectory import generate_mujoco_trajectory,generate_theoretical_trajectory
from xlsindy.logger import setup_logger
from xlsindy.optimization import lasso_regression
import time
import matplotlib.pyplot as plt
import os
import sys 
import importlib
import tyro
from pydantic import BaseModel
from typing import List
import hashlib
import json
from scipy import interpolate

def mujoco_transform(pos, vel, acc):

    return -pos, -vel, -acc

def inverse_mujoco_transform(pos, vel, acc):
    if acc is not None:
        return -pos, -vel, -acc
    else:
        return -pos, -vel, None
    
logger = setup_logger(__name__,level="DEBUG")

class Args(BaseModel):

    random_seed: List[int] = [10]
    """the random seed for the simulation"""
    batch_number: int = 2
    """the number of trajectory to generate"""
    max_time: float = 10.0
    """the maximum time of the trajectory"""
    initial_position: List[float] = [0, 0, 0,0.0]
    """the initial position of the trajectory"""
    initial_condition_randomness: List[float] = [0.1]
    """the randomness of the initial condition"""
    forces_scale_vector: List[float] = [2.0, 2.0]
    """the scale of the external forces"""
    data_ratio: float = 20.0
    """the ratio of data to catalog size"""
    validation_time: float = 15.0
    """the time of the validation trajectory"""
    noise_level: float = 0
    """the noise level of the trajectory"""
    experiment_system: str = "double_pendulum_pm"
    """the experiment system to use"""
    damping_coefficients: List[float] = [-1.5, -1.5]
    """the damping coefficients for each coordinate"""
    catalog_lenght: int = 20
    """the length of the catalog to use"""
    simulation_mode: str = "mixed"  

def main(args: Args):

    exp_uid = hashlib.md5(args.model_dump_json().encode()).hexdigest()[:8]

    ## 0. Import the metacode for the chosen experiment system

    # Add the experiment system folder to the path
    folder_path = os.path.join(os.path.dirname(__file__),"mujoco_align_data", args.experiment_system)
    sys.path.append(folder_path)
    # Import the metacode xlsindy_gen from the experiment system folder
    xlsindy_gen = importlib.import_module("xlsindy_gen")

    try:
        xlsindy_component = xlsindy_gen.xlsindy_component
    except AttributeError:
        raise AttributeError(
            "xlsindy_gen.py should contain a function named xlsindy_component"
        )

    try:
        mujoco_transform = xlsindy_gen.mujoco_transform
    except AttributeError:
        mujoco_transform = None

    try:
        inverse_mujoco_transform = xlsindy_gen.inverse_mujoco_transform
    except AttributeError:
        inverse_mujoco_transform = None

    num_coordinates, time_sym, symbols_matrix, catalog, xml_content, extra_info = (
        xlsindy_component( random_seed=args.random_seed, damping_coefficients=args.damping_coefficients,mode=args.simulation_mode,sindy_catalog_len=args.catalog_lenght)  # type: ignore
    )

    ideal_solution_vector = extra_info.get("ideal_solution_vector", None)
    if ideal_solution_vector is None:
        raise ValueError(
            "xlsindy_gen.py should return an ideal_solution_vector in the extra_info dictionary"
        )

    ## End import metacode 

    rng = np.random.default_rng(args.random_seed)


    # Create the mujoco trajectory
    (simulation_time_t, 
    simulation_qpos_t, 
    simulation_qvel_t, 
    simulation_qacc_t, 
    force_vector_t,
    _) = generate_mujoco_trajectory(
        num_coordinates,
        args.initial_position,
        args.initial_condition_randomness,
        args.random_seed,
        args.batch_number,
        args.max_time,
        xml_content,
        args.forces_scale_vector,
        mujoco_transform,
        inverse_mujoco_transform
    )

    # Add noise
    simulation_qpos_t += rng.normal(loc=0, scale=args.noise_level, size=simulation_qpos_t.shape)*np.linalg.norm(simulation_qpos_t)/simulation_qpos_t.shape[0]
    simulation_qvel_t += rng.normal(loc=0, scale=args.noise_level, size=simulation_qvel_t.shape)*np.linalg.norm(simulation_qvel_t)/simulation_qvel_t.shape[0]
    simulation_qacc_t += rng.normal(loc=0, scale=args.noise_level, size=simulation_qacc_t.shape)*np.linalg.norm(simulation_qacc_t)/simulation_qacc_t.shape[0]
    force_vector_t += rng.normal(loc=0, scale=args.noise_level, size=force_vector_t.shape)*np.linalg.norm(force_vector_t)/force_vector_t.shape[0]

    # Use a fixed ratio of the data in respect with catalog size
    catalog_size = catalog.catalog_length
    
    # Sample uniformly n samples from the imported arrays
    n_samples = int(catalog_size * args.data_ratio)
    total_samples = simulation_qpos_t.shape[0]

    if n_samples < total_samples:

        # Evenly spaced sampling (deterministic, uniform distribution)
        sample_indices = np.linspace(0, total_samples - 1, n_samples, dtype=int)
        
        # Apply sampling to all arrays
        simulation_qpos_t = simulation_qpos_t[sample_indices]
        simulation_qvel_t = simulation_qvel_t[sample_indices]
        simulation_qacc_t = simulation_qacc_t[sample_indices]
        force_vector_t = force_vector_t[sample_indices]
        
        logger.info(f"Sampled {n_samples} points uniformly from {total_samples} total samples")
    else:
        logger.info(f"Using all {total_samples} samples (requested {n_samples})")

    logger.info("Starting mixed regression")

    start_time = time.perf_counter()

    pre_knowledge_indices = np.nonzero(args.forces_scale_vector)[0] + catalog.starting_index_by_type("ExternalForces")
    pre_knowledge_mask = np.zeros((catalog.catalog_length,))
    pre_knowledge_mask[pre_knowledge_indices] = 1.0

    solution, exp_matrix = xlsindy.simulation.regression_mixed(
        theta_values=simulation_qpos_t,
        velocity_values=simulation_qvel_t,
        acceleration_values=simulation_qacc_t,
        time_symbol=time_sym,
        symbol_matrix=symbols_matrix,
        catalog_repartition=catalog,
        external_force=force_vector_t,
        regression_function=lasso_regression,
        pre_knowledge_mask=pre_knowledge_mask,
    )

    end_time = time.perf_counter()

    regression_time = end_time - start_time

    logger.info(f"Regression completed in {end_time - start_time:.2f} seconds")

    # Use the result to generate validation trajectory

    threshold = 1e-2  # Adjust threshold value as needed
    solution = np.where(np.abs(solution)/np.linalg.norm(solution) < threshold, 0, solution)

    model_acceleration_func_np, valid_model = (
        xlsindy.dynamics_modeling.generate_acceleration_function(
            solution, 
            catalog,
            symbols_matrix,
            time_sym,
            lambdify_module="numpy",
        )
    )



    (simulation_time_v, 
    simulation_qpos_v, 
    simulation_qvel_v, 
    simulation_qacc_v, 
    force_vector_v,
    _) = generate_mujoco_trajectory(
        num_coordinates,
        args.initial_position,
        args.initial_condition_randomness,
        args.random_seed+[0], # Ensure same seed as for data generation
        1,
        args.validation_time,
        xml_content,
        args.forces_scale_vector,
        mujoco_transform,
        inverse_mujoco_transform
    )

    print("""(simulation_time_v, 
simulation_qpos_v, 
simulation_qvel_v, 
simulation_qacc_v, 
force_vector_v,)""", (simulation_time_v.shape, 
    simulation_qpos_v.shape, 
    simulation_qvel_v.shape, 
    simulation_qacc_v.shape, 
    force_vector_v.shape,))

    if valid_model:
        (simulation_time_vs, 
         simulation_qpos_vs, 
         simulation_qvel_vs, 
         simulation_qacc_vs, 
         force_vector_vs,
         _) = generate_theoretical_trajectory(
             num_coordinates,
             args.initial_position,
             args.initial_condition_randomness,
             args.random_seed+[0], # Ensure same seed as for data generation
             1,
             args.validation_time,
             solution,
             catalog,
             time_sym,
             symbols_matrix,
             args.forces_scale_vector,
         )
    else:
        logger.warning("Model is not valid, skipping validation trajectory generation")
    
    labels = catalog.label()
    coeffs = np.array(solution)

    tolerance = 1e-6
    for j in range(coeffs.shape[1]):
        print(f"\n==== Learned equation for coordinate {j} (qddot_{j}) ====")
        for i, c in enumerate(coeffs[:, j]):
            if abs(c) > tolerance:
                print(f"{c:+.3e} * {labels[i]}")


    # Create a figure with 4 subplots stacked vertically
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('Trajectory Comparison: Mujoco vs. Theoretical', fontsize=16)

    # --- 1. Plot Position Data ---
    axes[0].plot(simulation_time_v, simulation_qpos_v, label='Mujoco Simulation')
    if valid_model:
        axes[0].plot(simulation_time_vs, simulation_qpos_vs, label='Theoretical Simulation', linestyle='--')
    axes[0].set_title('Position vs. Time')
    axes[0].set_ylabel('Position')
    axes[0].legend()
    axes[0].grid(True)

    # --- 2. Plot Velocity Data ---
    axes[1].plot(simulation_time_v, simulation_qvel_v, label='Mujoco Simulation')
    if valid_model:
        axes[1].plot(simulation_time_vs, simulation_qvel_vs, label='Theoretical Simulation', linestyle='--')
    axes[1].set_title('Velocity vs. Time')
    axes[1].set_ylabel('Velocity')
    axes[1].legend()
    axes[1].grid(True)

    # --- 3. Plot Acceleration Data ---
    axes[2].plot(simulation_time_v, simulation_qacc_v, label='Mujoco Simulation')
    if valid_model:
        axes[2].plot(simulation_time_vs, simulation_qacc_vs, label='Theoretical Simulation', linestyle='--')
    axes[2].set_title('Acceleration vs. Time')
    axes[2].set_ylabel('Acceleration')
    axes[2].legend()
    axes[2].grid(True)

    # --- 4. Plot Force Data ---
    axes[3].plot(simulation_time_v, force_vector_v, label='Mujoco Force')
    if valid_model:
        axes[3].plot(simulation_time_vs, force_vector_vs, label='Theoretical Force', linestyle='--')
    axes[3].set_title('Force vs. Time')
    axes[3].set_ylabel('Force')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend()
    axes[3].grid(True)

    # Improve layout to prevent labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to make room for the suptitle

    # Display the plots
    plt.savefig(f"test/trajectory_comparison_{exp_uid}.png")
    plt.close(fig)

    ### PLOTS WIL ###
    print("simulation_qpos_v =", simulation_qpos_v)
    plt.clf() 
    n = simulation_qpos_v.shape[0]
    plt.plot(simulation_qpos_v[:,0], simulation_qpos_v[:,1]) #color=[(i, 0, 1 - i) for i in np.arange(n) / n])
    plt.savefig(f"test/wil_joint_angles{exp_uid}.png")


    L1 = 0.2
    L2 = 0.275

    theta1 = np.unwrap(simulation_qpos_v[:, 0])
    theta2 = np.unwrap(simulation_qpos_v[:, 1])

    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)

    x2 = x1 + L2 * np.sin(theta1 + theta2)
    y2 = y1 - L2 * np.cos(theta1 + theta2)
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.scatter(x1[0], y1[0], color='red')
    ax2.plot(x1, y1, color='blue', label='test_elbow')
    ax2.scatter(x2[0], y2[0], color='red')
    ax2.plot(x2, y2, color='orange', label="test_ef")

    if valid_model:
        theta1 = np.unwrap(simulation_qpos_vs[:, 0])
        theta2 = np.unwrap(simulation_qpos_vs[:, 1])

        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)

        x2 = x1 + L2 * np.sin(theta1 + theta2)
        y2 = y1 - L2 * np.cos(theta1 + theta2)

        ax2.plot(x1, y1, color='blue', label='theoretical_elbow', linestyle='--')
        ax2.plot(x2, y2, color='orange', label="theoretical_ef", linestyle="--")

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.legend()
    ax2.set_title("End-effector path")
    ax2.grid(True)
    fig2.savefig(f"test/wil_{exp_uid}.png")
    plt.close(fig2)

    ###

    # Compute error between mujoco and theoretical simulations
    if valid_model:
        # Determine the common time range
        max_validation_end_time = min(simulation_time_v[-1,0], simulation_time_vs[-1,0])
        
        # Filter to common time range
        mask_v = simulation_time_v.flatten() <= max_validation_end_time
        mask_vs = simulation_time_vs.flatten() <= max_validation_end_time
        
        time_v_filtered = simulation_time_v[mask_v,0]
        qpos_v_filtered = simulation_qpos_v[mask_v,:]
        
        time_vs_filtered = simulation_time_vs[mask_vs,0]
        qpos_vs_filtered = simulation_qpos_vs[mask_vs,:]
        
        # Interpolate theoretical data onto mujoco time points
        qpos_vs_interpolated = np.zeros_like(qpos_v_filtered)
        for i in range(num_coordinates):
            interp_func = interpolate.interp1d(
                time_vs_filtered, 
                qpos_vs_filtered[:, i], 
                kind='cubic', 
                fill_value='extrapolate'
            )
            qpos_vs_interpolated[:, i] = interp_func(time_v_filtered)
        
        # Compute error (RMSE)
        error = np.sqrt(np.mean((qpos_v_filtered - qpos_vs_interpolated)**2))
        
        logger.info(f"Position RMSE between Mujoco and Theoretical: {error:.6f}")
    else:
        error = float('inf')
        max_validation_end_time = 0.0
        logger.warning("Model invalid, setting error to infinity")
    
    # Save results to JSON
    result_dict = args.model_dump()
    result_dict['error'] = error
    result_dict['max_validation_end_time'] = float(max_validation_end_time)
    result_dict['regression_time'] = regression_time
    result_dict['valid_model'] = valid_model
    
    os.makedirs("test", exist_ok=True)
    json_path = f"test/results_{exp_uid}.json"
    with open(json_path, 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    logger.info(f"Results saved to {json_path}")

if __name__ == "__main__":

    args = tyro.cli(Args)
    main(args)
        
        
