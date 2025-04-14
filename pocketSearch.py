#!/usr/bin/env python
import argparse
import os
from pathlib import Path
import ray
from search import pocketSearcher

# Define the parser
parser = argparse.ArgumentParser(description='Take user-defined run-time \
                                                arguments')

# Declare an argument, and using the default value if the argument isn't given
parser.add_argument('pdbdir', help='Directory of candidate scaffold PDBs')
parser.add_argument('outputdir', help='Output directory for rosetta files, \
                                    scorefile, etc.')
parser.add_argument('targetdir', help='Directory containing target catalytic \
                                    pocket conformations')
parser.add_argument('-a', '--alpha', dest='alpha', default=0.80,
                metavar='A', help='Percentage cutoff for minimum number of \
                                    alpha spheres in fpocket, default behavior \
                                    is 0.80 of target structure')
parser.add_argument('-c', '--cutoff', dest='cutoff', default=1.50,
                metavar='C', help='Percentage cutoff for maximum number of \
                                    alpha spheres in fpocket,default behavior \
                                    is 1.50 of target structure')
parser.add_argument('-f', '--filter', dest='filt', default=0.7,
                metavar='F', help='Minimum shared %% identity')
parser.add_argument('-ht','--hits', dest='hilt', default=1,
                metavar='H', help='Minimum number of %% identity hits')
parser.add_argument('-s','--screen', dest='screen', default=0.5,
                metavar='S', help='Short screen filter to determine whether \
                                    translational sampling occurs')
parser.add_argument('-cp','--checkpoint', dest='checkpoint', default=None,
                metavar='P', help='The name of checkpoint file to read in order \
                                    to restart a run')
parser.add_argument('-mp', '--multiprocessing', dest='mp', default=1,
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
targetdir = Path(args.targetdir)
alpha = float(args.alpha)
cutoff = float(args.cutoff)
min_intersect = float(args.filt)
min_hits = int(args.hilt)
screen = float(args.screen)
checkpoint = Path(args.checkpoint) if args.checkpoint is not None else args.checkpoint
multiprocessing = int(args.mp)

aliases = Path('aliases.yaml')

searcher = pocketSearcher(
    pdbdir,
    outputdir,
    targetdir,
    alpha,
    cutoff,
    min_intersect,
    min_hits,
    screen,
    checkpoint,
    aliases
)

searcher.preprocess()

if multiprocessing:
    @ray.remote
    def pocket_search(pocket_searcher: object,
                      structure: str) -> None:
        pocket_searcher.search(structure)

    # initialize ray
    ray.init()
    
    # setup futures
    futures = [pocket_search.remote(searcher, pdb) for pdb in searcher.remaining_pdbs]
    
    # run ray
    _ = ray.get(futures)

else:
    # run in serial on one thread
    for structure in searcher.remaining_pdbs:
        searcher.search(structure)

if os.path.exists('checkpoint.chk'):
    os.remove('checkpoint.chk')
