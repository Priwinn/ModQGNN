import numpy
import sparse
import re


def find_qbits(line):
    qbits = re.findall('\[[\d]*\]', line)
    for i, q in enumerate(qbits):
        qbits[i] = int(q[1:-1])
    return qbits


def parse(path, self_loops=False, skip_repeat=False, empty_slices=False):
    slices = []
    with open(path) as f:
        cycle = 0
        while True:
            line = f.readline()
            if line[0] == '.':
                line = line.split('_')
                n_qbits = int(line[1][1:])
                break
        while line:
            line = f.readline().strip(' ')
            try:
                _ = line[0]
            except IndexError:
                break
            data = []
            col_idx = []
            row_idx = []
            if line[0] == '{':
                while line[0] != '}':
                    line = f.readline().strip(' ')
                    qbits = find_qbits(line)
                    if len(qbits) == 1 and self_loops:
                        col_idx.append(qbits[0])
                        row_idx.append(qbits[0])
                        data.append(1)
                    if len(qbits) == 2:
                        col_idx.append(qbits[0])
                        row_idx.append(qbits[1])
                        col_idx.append(qbits[1])
                        row_idx.append(qbits[0])
                        data.append(1)
                        data.append(1)
            else:
                qbits = find_qbits(line)
                if 'skip' in line:
                    skip_times = int(re.findall('[\d]+', line)[0])
                    if skip_repeat:
                        for _ in range(skip_times):
                            slices.append(slices[-1])
                    cycle += skip_times
                if len(qbits) == 1 and self_loops:
                    col_idx.append(qbits[0])
                    row_idx.append(qbits[0])
                    data.append(1)
                if len(qbits) == 2:
                    col_idx.append(qbits[0])
                    row_idx.append(qbits[1])
                    col_idx.append(qbits[1])
                    row_idx.append(qbits[0])
                    data.append(1)
                    data.append(1)
            if data:
                slices.append(sparse.COO([row_idx, col_idx], data, shape=[n_qbits, n_qbits]))
            else:
                continue
            cycle += 1
    return sparse.stack(slices)


if __name__ == '__main__':
    parse('random_circuits/compiler_output/random_q64_s2936_2qbf082_1_last.qasm')
