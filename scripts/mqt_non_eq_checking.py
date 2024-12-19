"""Standalone script to perform non-equivalence checking on classes of benchmark circuits in
mqtbench and save the results in a csv file.
"""

import logging
import multiprocessing
import math
import random

from multiprocessing import Process

import pandas as pd

from mqt import qcec
from mqt.bench import get_benchmark
from mqt.qcec import Configuration
from mqt.qcec.configuration import augment_config_from_kwargs

from keys import IBMQ_API
from qiskit import transpile

import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService

# set random seed for reproducibility
random.seed(0)

# ---------------------------------------------------
# setup the logging both to stdout and file
# ---------------------------------------------------
logger = logging.getLogger(__name__)

def verify_benchmark(benchmark_name: str, n_qubits: int,
                     ibm_backend: qiskit_ibm_runtime.ibm_backend.IBMBackend,
                     config: dict, result_dict: dict):
    """Construct the circuits and run verification using qcec.

    Args:
        benchmark_name (str): Name of the benchmark circuit.
        n_qubits (int): No. of qubits to use for the circuit size.
        ibm_backend (qiskit_ibm_runtime.ibm_backend.IBMBackend): Fake or actual IBM backend to
            transpile the circuit to.
        config (dict): Configuration dictionary for qcec.verify().
        result_dict (dict): Result dictionary to save the results of equivalence checking.
    """
    print(f'Running verification on {benchmark_name} with {n_qubits} qubits.')

    # load the benchmark from mqtbench
    circ = get_benchmark(benchmark_name, level='indep', circuit_size=n_qubits)

    # IMPORTANT: for mqtbench circuits, in order for the verification to work propoerly, we need to
    # first remove any and then add measurements to all qubits at the end of the circuit
    circ.remove_final_measurements()
    circ.measure_all()

    # transpile to a HW backend
    circ_hw = transpile(circ, ibm_backend)

    # remove a random gate from the circuit
    removed = False
    while not removed:
        rand_idx = random.choice(range(sum(circ_hw.count_ops().values())))

        # don't randomly remove measurements and barriers
        if circ_hw.data[rand_idx].name not in ['measure', 'barrier']:
            del circ_hw.data[rand_idx]
            removed = True

    # verify equivalence
    result = qcec.verify(circ, circ_hw, config)

    # return the times and the results
    log_string = f'name: {benchmark_name}, n_qubits: {n_qubits}, ' +\
        f'preprocessing: {result.preprocessing_time}' +\
        f', check: {result.check_time}, equivalence: {result.equivalence}'

    print(log_string)

    # add the results to the result_dict to return from the subprocess
    result_dict['n_qubits'] = n_qubits
    result_dict['check_time'] = result.check_time
    result_dict['result'] = result.equivalence
    result_dict['started_simulations'] = result.started_simulations
    result_dict['performed_simulations'] = result.performed_simulations


if __name__ == '__main__':
    # benchmark name
    benchmark_name = 'qft'

    # setup the range for the n_qubits to explore
    min_qubits, max_qubits = 2, 40
    qubit_step_size = 2

    # string to identify this experiment
    id_string = f'{benchmark_name}_min_{min_qubits}_max_{max_qubits}_step_{qubit_step_size}'

    # setup basic logging to a file
    logging.basicConfig(
        filename=f'logs/mqtbench_{id_string}_non_eq_checking_results.log',
        level=logging.INFO
    )

    # setup the configuration for mqt qcec
    config = Configuration()
    config_dict = {
        # application
        "alternating_scheme": "proportional",
        "simulation_scheme": "proportional",

        # execution
        "run_zx_checker": False,
        # TODO(abhi) set this if need to use internal qcec timeout (note this does not always work!)
        # "timeout": 60.,

        # functionality
        "check_partial_equivalence": False,

        # optimizations
        "elide_permutations": False,
        "fuse_single_qubit_gates": False,
        "reconstruct_swaps": False,
        "reorder_operations": False,
        "transform_dynamic_circuit": False,

        # simulation
    }

    # update the configuration with the dictionary
    augment_config_from_kwargs(config, config_dict)

    # setup the hardware backend to use
    service = QiskitRuntimeService(channel="ibm_quantum", token=IBMQ_API,
                                   instance="ibm-q/open/main")
    ibm_backend = service.backend("ibm_sherbrooke")
    ext_timeout = 3600.

    benchmark_results_dict = {'n_qubits': [], 'check_time': [], 'result': [],
                              'started_simulations': [], 'performed_simulations': []}
    n_steps = math.floor((max_qubits-min_qubits)/qubit_step_size)

    for i, n_qubits in enumerate(range(min_qubits, max_qubits+1, qubit_step_size)):
        print(f'{i}/{n_steps}')

        # initialize the dictionary to save the result of the equivalence checking
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        prc = Process(
            target=verify_benchmark,
            kwargs={'benchmark_name': benchmark_name, 'n_qubits': n_qubits,
                    'ibm_backend': ibm_backend, 'config': config,
                    'result_dict': result_dict},
            name=f'process_verify_{benchmark_name}'
        )
        prc.start()
        prc.join(timeout=ext_timeout)
        prc.terminate()

        # add the process results if they exist to the benchmark_result_dict
        if all(key in result_dict.keys() for key in benchmark_results_dict.keys()):
            for key in benchmark_results_dict.keys():
                benchmark_results_dict[key].append(result_dict[key])

        if prc.exitcode is None:
            print(f'verification for {benchmark_name}_{n_qubits} unsucessful in {ext_timeout} sec.')

    # save the result of the experiments as csv to disk
    print(result_dict, benchmark_results_dict)
    result_df = pd.DataFrame.from_dict(benchmark_results_dict)
    result_df.to_csv(f'results/mqtbench_{id_string}_non_eq_checking_results.csv')