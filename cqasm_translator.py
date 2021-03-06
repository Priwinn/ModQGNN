###############################
# Supported translations:     #
# - QASM to cQASM 1.0         #
# - OPENQASM 2.0 to cQASM 1.0 #
#                             #
# Requirements:               #
# - Python3                   #
###############################

# TODO:
# - Use libraries for parsing
# - Find better way to identify file types

import argparse
import os
import sys
import csv
import re

#from qiskit.qasm import Qasm
#from qiskit.qasm.qasmparser import QasmParser

# Constant definitions
curdir = os.path.dirname(__file__)

# Hardcoded dictionary for translating gates
dictionary = {
    "i": "i",
    "id": "i",
    "cx": "cnot",
    "cnot": "cnot",
    "sdg": "sdag",
    "sdag": "sdag",
    "tdg": "tdag",
    "tdag": "tdag",
    "x": "x",
    "y": "y",
    "z": "z",
    "t": "t",
    "h": "h",
    "s": "s",
    "rz": "rz",
    "u1": "rz",
    "u2":"X90",
    "rx": "rx",
    "ry": "ry",
    "cz": "cz",
    "p":"rz",
    "toffoli": "toffoli",
    "ccx": "toffoli",
    "prep0": "prep_z",
    "prepz": "prep_z",
    "prepx": "prep_x",
    "prepy": "prep_y",
    "reset": "prep_z",
    "swap": "swap",
    "measure": "measure",
    "measure_z" :"measure_z",
    "measure_y" :"measure_y",
    "measure_x" :"measure_x",
    "rx(pi*0.5)": "X90",
    "rx(pi*-0.5)": "mX90",
    "ry(pi*0.5)": "Y90",
    "ry(pi*-0.5)": "mY90",
    "rz(pi*0.5)": "s",
    "cu1": "CR",
    "CR": "CR",
    "CRk": "CRk",
    "c-X": "c-X",
    "c-Y": "c-Y",
    "c-Z": "c-Z",
    "not": "not",
    "reg": "q"
    }

def is_qasm(lines):
    return ("# Circuit generated by QLib" in lines[0])

def is_openqasm(lines):
    for line in lines:
        if "OPENQASM" in line:
            return True
    return False


'''
Translate QASM gates to cQASM
'''
def qasm2cqasm(input_path, gates_buffer, lines):
    num_qubits = 0
    qubits_dict = []

    for line in lines:
        line = line.strip()

        # .qubit(s) XX
        qubits_match = re.findall(r'^\.qubit[s]? (\d+)$', line)
        if (qubits_match):
            num_qubits = int(qubits_match[0])
            continue

        # qubit qXX
        qubit_def_match = re.findall(r'^qubit (\w+)$', line)
        if (qubit_def_match):
            qubits_dict.append(qubit_def_match[0])
            continue

        # gate qXX qXX
        match = re.findall(r'^(\w+)((?: \w+)*)$', line)
        if (not match):
            continue

        gate = match[0][0].lower()
        operands = match[0][1].split()

        if gate in dictionary:
            try:
                indentation = "  "
                converted_gate = dictionary[gate]

                # For the CP gates, we have to catch and arrange the angle in the proper place
                #if gate is "cp":
                #    angle_str = operands[-1]

                # Convert operands from string to int
                converted_operands = []
                for qubit in operands:
                    converted_operands.append(str(qubits_dict.index(qubit)))

                angle_str = ""

                gates_buffer.append(
                    indentation
                    + converted_gate
                    + " "
                    + "q["
                    + "], q[".join(converted_operands)
                    + "]"
                    + angle_str
                    + "\n"
                )

            except IOError as err:
                print("IO ERROR: {0}".format(err))
                return -1
            except ValueError:
                print(
                   "ERROR while translating: \n"
                    + line + "\n"
                    + "The algorithm asks for an undeclared qubit in file: " + input_path
                    + "\n"
                )
                print("Qubit declared")
                print(qubits_dict)

                return -1
            except:
                print("UNEXPECTED ERROR: ", sys.exc_info()[0])
                return -1
        else:
            print("UNTRANSLATABLE GATE: `" + gate + "` in " + input_path)
            return -1

    return num_qubits

'''
Translate OPENQASM gates to cQASM
'''
def openqasm2cqasm(input_path, gates_buffer, lines):
    max_qubit_index = 0
    indentation = "  "
    
    #qasm = Qasm(input_path)
    #res = qasm.parse().qasm()
    #print(res)
    #return

    for line in lines:
        line = line.strip()
        operands = []

        # Ignore match
        ignore_match = re.findall(r'^(qreg|creg|barrier|//)(.*)$', line)
        if (ignore_match):
            continue
        
        # Find classical registers    
        c_reg = re.findall(r'(c)\[(\d+)\]*;', line)
        if c_reg:
            i = line.rfind("-")
            line = line[:i-1] + ";"

        #Find unitary gates
        u_match = re.findall(r'(u\d?)+(\((\-?\w*\*?\/?-?\d*\.?\d*\,*)*\))+', line)
        if u_match and 'u1' not in line:
            i = line.rfind(")")
            j = line.rfind("(")
            angles = line[j+1:i].split(",")
            angle_strs = []
            for angle_str in angles:
                if ("pi" in angle_str):
                    angle_strg = angle_str.replace("pi","3.14")
                    if "*" in angle_str: 
                        angle_split = angle_strg.split("*")
                        if ('/' in angle_split[1]):
                            angle_split2 = angle_split[1].split("/")
                            angle_calc = (float(angle_split[0]) * float(angle_split2[0])) / float(angle_split2[1])
                        else: 
                            angle_calc = float(angle_split[0]) * float(angle_split[1])
                    elif "/" in angle_str:
                        angle_split = angle_strg.split("/")
                        angle_calc = float(angle_split[0]) / float(angle_split[1])
                    else:
                        angle_calc = float(angle_strg)
                    angle_str = str(angle_calc)
                angle_strs.append(angle_str)
            if ('u2' not in line):
                line = "rz" + line[i+1:-1] + ", " + angle_strs[2] + ";" + "ry" + line[i+1:-1] + ", " + angle_strs[0] + ";" + "rz" + line[i+1:-1] + ", " + angle_strs[1]+ ";"
            else:
                line = "rz" + line[i+1:-1] + ", " + angle_strs[1] + ";" + "ry" + line[i+1:-1] + ", " + "1.57" + ";" + "rz" + line[i+1:-1] + ", " + angle_strs[0]+ ";"

            split_lines = line.split(";")
            for split_line in split_lines:
                gates_buffer.append(indentation + split_line + "\n")

            for op in operands:
                if (int(op) > max_qubit_index):
                    max_qubit_index = int(op)

        # gate q[XX], q[XX]
        match = re.findall(r'(^(\w+)(\(.*\))*\s+(?:\,?\s*(q||bits||reg||qr||c||a||b||carry||in||out||bound||delta||dist||ji_a||ji_b||ji_d||ji_k||I00||I11||I0g||I1g||v||m)\[(\d+)\])*)+;$', line)
        if (not match):    
            continue
            
        #operands_match = re.findall(r'\,?\s*(q||bits||reg)\[(\d+)\]', line)
        operands_match = re.findall(r'\[(\d+)\]', line)
        for op_match  in operands_match:
            operands.append(op_match)
        
        if (not u_match):
            gate = match[0][1].lower()

            if gate in dictionary:
                converted_gate = dictionary[gate]
                angle_str = ""
                if match[0][2]:
                    angle_str = match[0][2].replace("(", "").replace(")", "")
                    if ("pi" in angle_str):
                        angle_strg = angle_str.replace("pi","3.14")
                        if "*" in angle_str:
                            angle_split = angle_strg.split("*")
                            if ('/' in angle_split[1]):
                                angle_split2 = angle_split[1].split("/")
                                angle_calc = (float(angle_split[0]) * float(angle_split2[0])) / float(angle_split2[1])
                            else: 
                                angle_calc = float(angle_split[0]) * float(angle_split[1])
                        elif "/" in angle_strg:
                           angle_split = angle_strg.split("/")
                           angle_calc = float(angle_split[0]) / float(angle_split[1])
                        else:
                            angle_calc = float(angle_strg) 
                        angle_str = ", " + str(angle_calc)
                    else:
                        angle_str = ", " + angle_str


                gates_buffer.append(
                    indentation
                    + converted_gate
                    + " "
                    + "q["
                    + "], q[".join(operands)
                    + "]"
                    + angle_str
                    + "\n"
                )

                for op in operands:
                    if (int(op) > max_qubit_index):
                        max_qubit_index = int(op)
            else:
                print("UNTRANSLATABLE GATE: `" + gate + "` in " + input_path)
                return -1                 

    # Return the number of qubits
    return max_qubit_index + 1

'''
Translate file(s) to cQASM
'''
def translate(input_path, output_path, recursive = False, save_path = None):
    # Validate params
    if (recursive and os.path.isdir(input_path)):
        if (not os.path.isdir(output_path)):
            print("ERROR: the ouput path is not a folder")
            return

        translate_dir(input_path, output_path, save_path)
        return
    elif (not os.path.isfile(input_path)):
        print("ERROR: the input path is not a file")
        return

    # Get basename
    basename = os.path.basename(input_path)
    if (os.path.isdir(output_path)):
        output_path = os.path.dirname(output_path) + basename

    # Check to not override input
    if (os.path.realpath(input_path) in os.path.realpath(output_path)):
        print("ERROR: input must be different from the output")
        return

    print("Translating " + basename + "...")

    with open(input_path, 'r') as input_file:
        with open(output_path, 'w') as output_file:
            # Translate gates
            gates_buffer = []
            lines = input_file.readlines()
            if (is_openqasm(lines)):
                # OPENQASM to cQASM
                #source = "RevLib"
                type = "OPENQASM"
                num_qubits = openqasm2cqasm(input_path, gates_buffer, lines)
            elif (is_qasm(lines)):
                # QASM to cQASM
                #source = "QLib"
                type = "QASM"
                num_qubits = qasm2cqasm(input_path, gates_buffer, lines)
            else:
                print("ERROR: file " + input_path + " not recognized\n")
                return

            if (not num_qubits or num_qubits <= 0):
                print("ERROR: the " + type + " file can not be converted\n")
                os.remove(output_path)
                return

            init_buffer = []

            # Add version
            init_buffer.append("version 1.0\n\n")

            # Add qubits defintion
            init_buffer.append("qubits " + str(num_qubits) + "\n\n")

            # Add kernel name
            # Note:
            # - Force kernel name to start with letter (cQASM spec)
            # - Change all dashes
            kernel_name = os.path.splitext(basename)[0].replace("-", "_")
            init_buffer.append("._" + kernel_name + "\n")

            output_file.writelines(init_buffer)
            output_file.writelines(gates_buffer)

            if (save_path):
                write_benchmark_to_db(
                    kernel_name, num_qubits, len(gates_buffer), type,
                    save_path
                )

            print("| " + kernel_name + " | " + str(num_qubits) + " | " + str(len(gates_buffer)) + "\n")

def translate_dir(input_path, output_path, save_path):
    # Validate params
    if (not os.path.isdir(input_path) or not os.path.isdir(output_path)):
        print("Input or output are not a directory")
        return

    print("Looking for .qasm files...")

    # Get all "qasm" files at input_path
    qasm_files = (filename for filename in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, filename)) and ".qasm" in filename
    )

    for filename in qasm_files:
        translate(
            os.path.join(input_path, filename),
            os.path.join(output_path, filename),
            False,
            save_path
        )

def write_benchmark_to_db(algorithm_name, num_qubits, num_gates, source, save_path):
    row = {
        "Algorithm": algorithm_name, "No. qubits": num_qubits,
        "No. gates": num_gates, "Source": source
    }
    with open(save_path, "a") as benchmarks_db:
        writer = csv.DictWriter(
            benchmarks_db,
            fieldnames=["Algorithm", "No. qubits", "No. gates", "Source"]
        )
        file_is_empty = (os.stat(save_path).st_size == 0)
        if (file_is_empty):
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
	# Get params
    parser = argparse.ArgumentParser(description='OPENQASM translator to cQASM')

    parser.add_argument('input',
        help='the path to the input OPENQASM file or folder')
    parser.add_argument('output',
        help='the path to the output cQASM file or folder')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write-to',
        help='the path to the CSV file to save the data')
    group.add_argument('--append-to',
        help='the path to the CSV file to save the data')

    args = parser.parse_args()

    # Validate params
    save_path = args.append_to or args.write_to
    if (save_path):
        if (os.path.isdir(save_path)):
            save_path = os.path.join(save_path, "benchmarks.csv")

        if (not args.append_to):
            os.remove(save_path)

    # Do the translation
    translate(args.input, args.output, True, save_path)
