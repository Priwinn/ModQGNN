import os
from random_circuit_generator.random_circuit_generator import main as rcg
from qasm_parser import parse
import sparse
import subprocess

def main(q, s, f, n):
    for _ in range(n):
        filename = rcg(q, s, f)
        subprocess.Popen(['python',f'random_circuits/{filename}','config/test_mapper_surf_100.json']).wait()
        compiler_path = os.path.join(os.curdir, 'compiler_output')
        slices = parse(os.path.join(compiler_path, filename.strip('.py')+'_last.qasm'))
        sparse.save_npz(os.path.join(os.curdir, 'random_circuits', filename[:-3] + '.npz'), slices)


if __name__ == '__main__':
    main(100, 5000, 0.8, 1000)
