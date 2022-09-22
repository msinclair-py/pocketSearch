#!/usr/bin/env python
import argparse, os
import ray
from utils import *

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
parser.add_argument('-cp','--checkpoint', dest='checkpoint', default=False,
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
pdbdir = args.pdbdir
outputdir = args.outputdir
targetdir = args.targetdir
alpha = float(args.alpha)
cutoff = float(args.cutoff)
min_intersect = float(args.filt)
min_hits = int(args.hilt)
screen = float(args.screen)
checkpoint = args.checkpoint
multiprocessing = int(args.mp)

# make sure directories have correct formatting
pdbdir = check_format(pdbdir)
outputdir = check_format(outputdir)
targetdir = check_format(targetdir)

# run preprocessing
preprocessed = preprocess(checkpoint, pdbdir, targetdir, alpha, cutoff, outputdir)
tracker, t, s, short_sample, vol = preprocessed

# initialize scorefile
gen_scorefile(outputdir)

# run pocketSearch
if multiprocessing:
    # wrap with ray to enable multiprocessing
    pocket_search = ray.remote(pocket_search)
    
    # initialize ray
    ray.init()
    
    # obtain parameters for ray
    shared = [outputdir, pdbdir, targetdir, t, s, short_sample, 
                min_intersect, vol, screen, min_hits]
    params = [[i, structure] + shared for i, structure in enumerate(tracker)]

    # setup futures
    futures = [pocket_search.remote(*par) for par in params]
    
    # run ray
    _ = ray.get(futures)

else:
    # run in serial on one thread
    for i, structure in enumerate(tracker):
        pocket_search(i, structure, outputdir, pdbdir, targetdir, t, s, 
                        short_sample, min_intersect, vol, screen, min_hits) 

postprocessing(outputdir)

if os.path.exists('checkpoint.chk'):
    os.remove('checkpoint.chk')
