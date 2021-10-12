#!/usr/bin/python

from openql import openql as ql
import random
import os
import pickle
import sys, getopt
import numpy as np


def random_circuit(qubits, size, two_qubit_fraction, initialization='h', mirrored=True, seed=None):
	"""
	Args:
		qubits: Number of circuit qubits.
		gate_load: The fraction of busy cycles overall.
		gate_domain: The set of gates to choose from, with a specified arity.
		two_qubit_fraction: Fraction of the gate_load that corresponds to two_qubit gates. (Note that a two qubit gate introduces 4 times the load of a single qubit gate, since they require double the qubits for double the amount of time.)

	Raises:
		ValueError:
			* gate_load is not in (0, 1).
			* gate_domain is empty.
			* qubits is an int less than 1 or an empty sequence.

	Returns:
		The randomly generated Circuit.
	"""

	random.seed(a=seed)

	if isinstance(qubits, int):
		if qubits < 1:
			raise ValueError('qubits must be a >=1 integer.')
	else:
		raise ValueError('qubits must be a >=1 integer.')

	gates_1qb = ['x', 'x45', 'x90', 'xm45', 'xm90', 'y', 'y45', 'y90', 'ym45', 'ym90']
	gates_1qb_with_idle = gates_1qb + ['idle']
	gates_2qb = ['cz']
	qubit_list = list(range(qubits))
	gate_list = []

	if initialization == 'h':
		for qubit in qubit_list:
			gate_list.append(('x', (qubit,)))
			gate_list.append(('ym90', (qubit,)))

	elif initialization == 'random':
		for qubit in qubit_list:
			gate_list.append((random.choice(gates_1qb), (qubit,)))

	elif initialization == 'random_with_idle':
		for qubit in qubit_list:
			_gate = random.choice(gates_1qb_with_idle)
			if _gate == "idle":
				continue
			gate_list.append((_gate, (qubit,)))

	elif initialization == 'x':
		for qubit in qubit_list:
			gate_list.append(('x', (qubit,)))

	if mirrored:
		ngates = round(size / 2)
	else:
		ngates = size

	print("Size: ", size)
	print("Number of qubits: ", qubits)
	print("Two-qubit gate fraction: ", two_qubit_fraction)

	for i in range(ngates):
		dice = random.random()
		if dice < two_qubit_fraction:
			gate = 'cz'
			operands = tuple(random.sample(qubit_list, 2))
			gate_list.append((gate, operands))
		else:
			gate = random.choice(gates_1qb)
			operand = (random.choice(qubit_list),)
			gate_list.append((gate, operand))

	# Now we mirror the circuit!!! Append the conjugate gates at the end
	inverse_dict = {'x': 'x', 'x45': 'xm45', 'x90': 'xm90', 'xm45': 'x45', 'xm90': 'x90', 'y': 'y', 'y45': 'ym45',
					'y90': 'ym90', 'ym45': 'y45', 'ym90': 'y90', 'cz': 'cz'}
	if mirrored:
		for gate in reversed(gate_list):
			gate_list.append((inverse_dict[gate[0]], gate[1]))

	return tuple(gate_list)


def get_openql_script(circ_name, qubits, circ):
	beginning = """
from openql import openql as ql
import os
import argparse

def circuit(config_file, new_scheduler='yes', scheduler='ALAP', uniform_sched= 'no', sched_commute = 'yes', mapper='minextend', moves='yes', maptiebreak='first', initial_placement='no', output_dir_name='random_output', optimize='no', measurement=True, log_level='LOG_WARNING'):
	output_dir = output_dir_name
	ql.initialize()
	# uses defaults of options in mapper branch except for output_dir and for maptiebreak
	ql.set_option('output_dir', output_dir)     # this uses output_dir set above
	ql.set_option('maptiebreak', maptiebreak)       # this makes behavior deterministic to cmp with golden
													# and deviates from default

	ql.set_option('log_level', log_level)
	ql.set_option('optimize', optimize)
	ql.set_option('mapmaxalters', '10')
	ql.set_option('scheduler_heuristic', 'random')
	ql.set_option('use_default_gates', 'no')
	ql.set_option('generate_code', 'no')

	ql.set_option('decompose_toffoli', 'no')
	ql.set_option('scheduler', scheduler)
	ql.set_option('scheduler_uniform', uniform_sched)
	ql.set_option('scheduler_commute', sched_commute)
	ql.set_option('scheduler_commute_rotations', 'yes')
	ql.set_option('prescheduler', 'yes')
	ql.set_option('cz_mode', 'manual')
	ql.set_option('print_dot_graphs', 'no')
	
	ql.set_option('clifford_premapper', 'yes')
	ql.set_option('clifford_postmapper', 'no')
	ql.set_option('mapper', 'no')
	ql.set_option('mapinitone2one', 'yes')
	ql.set_option('mapassumezeroinitstate', 'yes')
	ql.set_option('initialplace', initial_placement)
	ql.set_option('initialplace2qhorizon', '0')
	ql.set_option('mapusemoves', moves)
	ql.set_option('mapreverseswap', 'yes')
	ql.set_option('mappathselect', 'random')
	ql.set_option('maplookahead', 'noroutingfirst')
	ql.set_option('maprecNN2q', 'no')
	ql.set_option('mapselectmaxlevel', '0')
	ql.set_option('mapselectmaxwidth', 'min')
	#ql.set_option('scheduler_post179', new_scheduler)
	ql.set_option('write_report_files', 'no')
	
	# platform  = ql.Platform('platform_none', config_file)
	platform  = ql.Platform('starmon', config_file)
	num_circuits = 1\n"""

	middle = "	num_qubits = " + str(qubits) + "\n"
	middle += "	p = ql.Program('" + circ_name + "', platform, num_qubits)\n"
	middle += "	k = ql.Kernel('" + circ_name + "', platform, num_qubits)\n"

	bulk = ""
	for gate in circ:
		bulk += "	k.gate('" + gate[0] + "', " + str(gate[1]) + ")\n"

	end = """
	if measurement:
		for q in range(num_qubits):
			k.gate('measure', [q])

	p.add_kernel(k)
	p.compile()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='OpenQL compilation of a Quantum Algorithm')
	parser.add_argument('config_file', help='Path to the OpenQL configuration file to compile this algorithm')
	parser.add_argument('--new_scheduler', nargs='?', default='yes', help='Scheduler defined by Hans')
	parser.add_argument('--scheduler', nargs='?', default='ASAP', help='Scheduler specification (ASAP (default), ALAP, ...)')
	parser.add_argument('--uniform_sched', nargs='?', default='no', help='Uniform shceduler actication (yes or no)')
	parser.add_argument('--sched_commute', nargs='?', default='yes', help='Permits two-qubit gates to be commutable')
	parser.add_argument('--mapper', nargs='?', default='base', help='Mapper specification (base, minextend, minextendrc)')
	parser.add_argument('--moves', nargs='?', default='no', help='Let the use of moves')
	parser.add_argument('--maptiebreak', nargs='?', default='random', help='')
	parser.add_argument('--initial_placement', nargs='?', default='no', help='Initial placement specification (yes or no)')
	parser.add_argument('--out_dir', nargs='?', default='compiler_output', help='Folder name to store the compilation')
	parser.add_argument('--measurement', nargs='?', default=True, help='Add measurement to all the qubits in the end of the algorithm')
	args = parser.parse_args()
	try:
		circuit(args.config_file, args.new_scheduler, args.scheduler, args.uniform_sched, args.sched_commute, args.mapper, args.moves, args.maptiebreak, args.initial_placement, args.out_dir)
	except TypeError:
		print('\\nCompiled, but some gate is not defined in the configuration file. \\nThe gate will be invoked like it is.')
		raise"""

	return beginning + middle + bulk + end


# %%
file_counter = {}  # keys: (qubit_number, size, qubit_fraction), value = count


def start_file_counter(input_dir):
	global file_counter

	files = [name for name in os.listdir(input_dir) if
			 os.path.isfile(os.path.join(input_dir, name)) and (".py" in name)]
	name = ""
	for file in files:

		extension_index = file.rindex(".")
		file = file[:extension_index]

		underscore = [i for i in range(len(file)) if file.startswith('_', i)]
		q = file[underscore[0] + 2:underscore[1]]
		s = file[underscore[1] + 2:underscore[2]]
		twoqbf = file[underscore[2] + 5:underscore[3]]
		count = int(file[underscore[3] + 1:])
		if file_counter.get((q, s, twoqbf), 0) < count:
			file_counter[(q, s, twoqbf)] = count

# %%
def save_random_circ_list(save_dir, circ_list, qubits, size, two_qubit_fraction):
	global file_counter
	two_qubit_fraction = str(round(two_qubit_fraction, 2)).replace('.', '')
	qubits = str(qubits)
	size = str(size)
	file_pattern = "random_q" + qubits + "_s" + size + "_2qbf" + two_qubit_fraction
	number = file_counter.get((qubits, size, two_qubit_fraction), 0)
	for circ in circ_list:
		number += 1
		circ_name = file_pattern + "_" + str(number)
		with open(os.path.join(save_dir, circ_name + '.py'), 'w') as fopen:
			fopen.writelines(get_openql_script(circ_name, qubits, circ))

	file_counter[(qubits, size, two_qubit_fraction)] = number
	return circ_name + '.py'

def main(qubits,size,fraction):
	output_dir = "random_circuits"

	initialization = "random"
	mirrored = False
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	circ_list = set([random_circuit(qubits, size, fraction, initialization=initialization, mirrored=mirrored)])
	start_file_counter(output_dir)
	filename=save_random_circ_list(output_dir, circ_list, qubits, size - qubits, (fraction * size) / (size - qubits))
	return filename

if __name__ == "__main__":
	argv=sys.argv[1:]
	try:
		opts, args = getopt.getopt(argv, "ho:q:f:s:", ["odir=", "qubits=", "2qgatefrac=", "size"])
	except getopt.GetoptError:
		print('random_circuit_generator.py -o <output_dir> -q <qubits> -f <two_qubit_gate_fraction> -s <size>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('random_circuit_generator.py -o <output_dir> -q <qubits> -f <two_qubit_gate_fraction> -s <size>')
			sys.exit()
		elif opt in ("-o", "--odir"):
			output_dir = arg
		elif opt in ("-q", "--qubits"):
			qubits = int(arg)
		elif opt in ("-f", "--2qgatefrac"):
			fraction = float(arg)
		elif opt in ("-s", "--size"):
			size = int(arg)
	main(qubits,size,fraction)
