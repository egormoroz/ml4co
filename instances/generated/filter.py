import multiprocessing as mp
from datetime import datetime
from pathlib import Path
import time
import gzip
import shutil
import sys
import glob, os

from dataclasses import dataclass

from pyscipopt import Model


@dataclass
class Inst:
    orig: str
    clones: list


def myprint(*args, **kwargs):
    print(f'[{datetime.now()}]', end=' ')
    print(*args, **kwargs)
    sys.stdout.flush()

def cp_compress(src_path, dst_path):
    with open(src_path, 'rb') as src, gzip.open(dst_path, 'wb') as dst:
        dst.writelines(src)

# worker (hopefully) completes a round
def worker_routine(args):
    idx, params, d, inq, outq, working = args

    m = Model()
    m.hideOutput()

    myprint(f'[ worker {idx} ] started')
    while working.value != 0:
        orig, inst = inq.get()

        m.readProblem(inst)
        m.setParams(params)
        m.optimize()

        if m.getNSols() > 0:
            outq.put((orig, inst))
        else:
            outq.put((orig, None))

    myprint(f'[ worker {idx} ] finished')
    

def maintain_queues(d, inq, outq, outdir):
    feasibles = dict(zip(d.keys(), [0] * len(d)))
    total, done, k = 0, 0, 0
    round_done = set()

    for name, inst in d.items():
        total += len(inst.clones)
        inq.put((name, inst.clones.pop(0)))
    
    while done < total:
        orig_name, clone_path = outq.get()
        done += 1

        def find_next(exclude):
            min_name, min_feasibles = None, 1000000
            for name, n in feasibles.items():
                if n < min_feasibles and d[name].clones and not (name in exclude):
                    min_feasibles = n
                    min_name = name
            return min_name, min_feasibles

        min_name, min_feasibles = find_next(round_done)
        if min_name is None:
            min_name, min_feasibles = find_next(set())

        if clone_path:
            if len(round_done) >= len(d):
                round_done.clear()
                k += 1
            round_done.add(orig_name)

            if orig_name in feasibles:
                feasibles[orig_name] += 1

            myprint(f'[{done}/{total}] [{k}:{len(round_done)}/{len(d)}] '
                    f'{clone_path} ({orig_name}) {min_feasibles} ({min_name})')
            sys.stdout.flush()
            
            clone_name = clone_path.partition('/')[-1]
            cp_compress(clone_path, f'{outdir}/{clone_name}.gz')

        inq.put((min_name, d[min_name].clones.pop(0)))
        if len(d[min_name].clones) == 0:
            del d[min_name]
            del feasibles[min_name]



def main():
    with open('splist.txt') as f:
        names = [i.partition('.')[0] for i in f]

    d = {}
    for name in names:
        orig = f'origs/{name}.mps.gz'
        assert os.path.isfile(orig)

        clones = glob.glob(f'clones/{name}*')
        d[name] = Inst(orig=orig, clones=sorted(clones))

    outdir = 'feas'
    Path(outdir).mkdir(exist_ok=True)
    params = { 'limits/solutions': 1, 'limits/time': 600 }

    m = mp.Manager()

    inq = m.Queue()
    outq = m.Queue()

    working = m.Value('b', 1)

    n = 8

    with mp.Pool(n) as pool:
        args = [(i, params, d, inq, outq, working) for i in range(n)]
        results = pool.map_async(worker_routine, args)

        maintain_queues(d, inq, outq, outdir)
        working.value = 0

        pool.close()
        pool.join()

    myprint('done')


if __name__ == '__main__':
    main()

