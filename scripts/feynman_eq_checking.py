import logging
import multiprocessing
import random
import sys

from multiprocessing import Process

import pandas as pd

from mqt import qcec
from mqt.qcec import Configuration
from mqt.qcec.configuration import augment_config_from_kwargs

from keys import IBMQ_API
from qiskit import QuantumCircuit, transpile

import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService

# set random seed for reproducibility
random.seed(0)

# ---------------------------------------------------
# setup the logging both to stdout and file
# ---------------------------------------------------
logger = logging.getLogger(__name__)


def verify_benchmark(benchmark_name: str, ibm_backend: qiskit_ibm_runtime.ibm_backend.IBMBackend,
                     config: dict, result_dict: dict):
    """Construct the circuits and run verification using qcec.

    Args:
        benchmark_name (str): Name of the benchmark circuit.
        ibm_backend (qiskit_ibm_runtime.ibm_backend.IBMBackend): Fake or actual IBM backend to
            transpile the circuit to.
        config (dict): Configuration dictionary for qcec.verify().
        result_dict (dict): Result dictionary to save the results of equivalence checking.
    """
    print(f'Running verification on {benchmark_name}!')
    # load the qasm file
    qasm_path = "../feynman/benchmarks/qasm/" + benchmark_name + ".qasm"
    circ = QuantumCircuit.from_qasm_file(qasm_path)

    # IMPORTANT: add measurements to the end of the circuit
    circ.measure_all()

    # transpile to a HW backend
    circ_hw = transpile(circ, ibm_backend)

    # verify equivalence
    result = qcec.verify(circ, circ_hw, config)

    # return the times and the results
    log_string = f'name: {benchmark_name}, preprocessing: {result.preprocessing_time}' +\
        f', check: {result.check_time}, equivalence: {result.equivalence}'

    print(log_string)

    # add the results to the result_dict to return from the subprocess
    result_dict['name'] = benchmark_name
    result_dict['check_time'] = result.check_time
    result_dict['result'] = result.equivalence


if __name__ == '__main__':
    # setup basic logging to a file
    logging.basicConfig(filename='logs/feynman_eq_checking_results.log', level=logging.INFO)

    # load the csv containing benchmark circuit properties
    circ_info_df = pd.read_csv('feynman_benchmark_properties.csv')
    circ_info_df = circ_info_df.sort_values(by='n_gates_original')

    # extract the benchmark names from the df
    benchmark_names = circ_info_df.name
    benchmark_names = benchmark_names.to_list()

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

    benchmark_results_dict = {'name': [], 'check_time': [], 'result': []}

    for i, benchmark_name in enumerate(benchmark_names):
        print(f"{i}/{len(benchmark_names)}")

        # initialize the dictionary to save the result of the equivalence checking
        manager = multiprocessing.Manager()
        result_dict = manager.dict()

        prc = Process(
            target=verify_benchmark,
            kwargs={'benchmark_name': benchmark_name, 'ibm_backend': ibm_backend, 'config': config,
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
            print(f'verification for {benchmark_name} unsucessful in {ext_timeout} seconds.')

    # save the result of the experiments as csv to disk
    print(result_dict, benchmark_results_dict)
    result_df = pd.DataFrame.from_dict(benchmark_results_dict)
    result_df.to_csv("results/feynman_eq_checking_results.csv")
