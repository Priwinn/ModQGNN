import os
from random_circuit_generator.random_circuit_generator import main as rcg
from qasm_parser import parse
import sparse
import subprocess

def main(q, s, f, n, out_path):
    for _ in range(n):
        filename = rcg(q, s, f)
        subprocess.Popen(['python',f'random_circuits/{filename}','config_quantum/test_mapper_surf_100.json']).wait()
        compiler_path = os.path.join(os.curdir, 'compiler_output')
        slices = parse(os.path.join(compiler_path, filename.strip('.py')+'_scheduled.qasm'))
        sparse.save_npz(os.path.join(os.curdir, out_path, filename[:-3] + '.npz'), slices)

def parse_folder(path):
    for file in os.listdir(path):
        if '.py' in file:
            print(f'{path}/{file}')
            subprocess.Popen(['python',f'{path}/{file}','config_quantum/test_mapper_100_not_constrained.json']).wait()
            compiler_path = 'compiler_output'
            slices = parse(os.path.join(compiler_path, file.strip('.py')+'_scheduled.qasm'))
            sparse.save_npz(os.path.join(os.curdir, path, file[:-3] + '.npz'), slices)

if __name__ == '__main__':
    #Random
    main(100, 5000, 0.4, 1000,'random02')
    #Folder parsing
    # parse_folder('real_circuits/qft_circuits')
