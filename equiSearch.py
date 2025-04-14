#!/usr/bin/env python
import argparse
import numpy as np
import os
from pathlib import Path
import ray
from equivariant import pocketHomology

# Define the parser
parser = argparse.ArgumentParser(description='Take user-defined run-time \
                                                arguments')

# Declare an argument, and using the default value if the argument isn't given
parser.add_argument('pdbdir', help='Directory of candidate scaffold PDBs')
parser.add_argument('outputdir', help='Output directory for rosetta files, \
                                    scorefile, etc.')
parser.add_argument('target', help='Path to PDB of target fpocket output')
parser.add_argument('-n', '--num', dest='num_spheres', default=106,
                metavar='N', help='Number of alpha spheres in target pocket.')
parser.add_argument('-t', '--threads', dest='threads', default=1,
                metavar='M', help='Proportion of cpu threads to use in \
                                    multiprocessing. If set to 0, multi\
                                    processing is turned off. Defaults to all \
                                    available threads for use on hpc resources.')

# Now, parse the command line arguments and 
# store the values in the arg variable
args = parser.parse_args()

# initialize all options to corresponding variables
pdbdir = Path(args.pdbdir)
outputdir = Path(args.outputdir)
target = Path(args.target)
n_spheres = int(args.num_spheres)
multiprocessing = int(args.threads)

aliases = Path('aliases.yaml')

searcher = pocketHomology(
    target,
    pdbdir,
    outputdir,
    n_spheres,
    aliases=aliases,
)

batched = np.array([p for p in pdbdir.glob('*.pdb') if len(p.stem) == 4])
print(batched)
#batched = np.array_split(np.array(pdbdir.glob('*.pdb')), multiprocessing)

if multiprocessing > 1:    
    @ray.remote
    def pocket_search(pocket_searcher: object,
                      structures: List[str]) -> None:
        pocket_searcher.batch(structures)

    # initialize ray
    ray.init()
    
    # setup futures
    futures = [pocket_search.remote(searcher, pdbs) for pdbs in batched]
    
    # run ray
    _ = ray.get(futures)

else:
    # run in serial on one thread
    for structure in batched.flatten():
        searcher.search(structure)

if os.path.exists('checkpoint.chk'):
    os.remove('checkpoint.chk')
