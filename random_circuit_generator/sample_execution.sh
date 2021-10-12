#! sh

python_dir="python39"
output_folder="../random_circuits"
n_qubits=64 # Number of qubits that the circuit will assume (and hence operate with)
qgatefrac=0.8 # float in the [0,1] interval stating the percentage of 2-qubit gates
size=3000 # Total number of gates used in the circuit

${python_dir} ./random_circuit_generator.py -o ${output_folder} -q ${n_qubits} -f ${qgatefrac} -s ${size}
