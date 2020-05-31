#!/usr/bin/env python
import os, glob, argparse
from utils import *
from coordinate_manipulation import Coordinates

# Define the parser
parser = argparse.ArgumentParser(description='Take user-defined run-time arguments')

# Declare an argument, and using the default value if the argument isn't given
parser.add_argument('inputdir', help='Directory of PDBs to be tested against target')
parser.add_argument('outputdir', help='Output directory')
parser.add_argument('initialdir', 
		help='Directory containing target pocket conformations')
parser.add_argument('-m', '--mode',  dest='mode', nargs='?',
		choices=['random','skip'], metavar='M', default='skip',
		help='Mode of action, choices: skip (default), random download')
parser.add_argument('-a', '--alpha', dest='alpha', default=80,
		metavar='A', help='Cutoff for minimum number of alpha spheres in fpocket')
parser.add_argument('-c', '--cutoff', dest='cutoff', default=150,
		metavar='C', help='Cutoff for maximum number of alpha spheres in fpocket')
parser.add_argument('-f', '--filter', dest='filt', default=0.7,
		metavar='F', help='Minimum shared %% identity')
parser.add_argument('-ht','--hits', dest='hilt', default=1,
		metavar='H', help='Minimum number of %% identity hits')
parser.add_argument('-r','--rand', dest='rand', default=100,
		metavar='R', help='Number of random structure to download, if random mode selected')
parser.add_argument('-s','--screen', dest='screen', default=0.3,
		metavar='S', help='Short screen filter to determine whether \
						translational sampling occurs')

# Now, parse the command line arguments and store the values in the arg variable
args = parser.parse_args()

# initialize all options to corresponding variables
inputdir = args.inputdir
outputdir = args.outputdir
initialdir = args.initialdir
alpha = int(args.alpha)
cutoff = int(args.cutoff)
min_intersect = float(args.filt)
min_hits = int(args.hilt)
rand = int(args.rand)
screen = float(args.screen)

# make sure directories have correct formatting
inputdir = checkFormat(inputdir)
outputdir = checkFormat(outputdir)
initialdir = checkFormat(initialdir)

# Different modes of pdb handling
if args.mode=='random':
	randomPDBs(inputdir,rand,outputdir)
elif args.mode=='sublist':
	print('Which sublist? (1-400)')
	num=input()
	sublist(num)
elif args.mode!='skip':
	print("No mode was selected for pdb acquisition, default is to skip downloading pdbs")

# rename everything to be consistent with typical naming schemes
formatPDBs(inputdir)

# clean up pdbs, removing cofactors and renumbering where necessary
for unclean in glob.glob(f'{inputdir}*.pdb'):
	clean(unclean)

# generate pockets
find_pockets(inputdir,alpha)

name_array=[os.path.basename(name) for name in glob.glob(f'{inputdir}*_out/*_out.pdb')]

for entry in name_array:
	writePockets(inputdir,entry,cutoff)

# center and align pockets
prealigned = []
for name in glob.glob(f'{inputdir}*.pocket*.pdb'):
	stem = os.path.basename(name).split('.')
	prealigned.append([stem[0],stem[1]])

tracker = []
for entry in prealigned:
	print(entry)
	tracker.append(f'{entry[0]}.{entry[1]}')
	coordsystem = Coordinates(inputdir,entry[0],pnum=entry[1])
	coords = coordsystem.getCoords()
	centered = coordsystem.center(coords)
	vector = coordsystem.principal(centered)
	aligned = coordsystem.align(centered,vector)
	coordsystem.makePDB(aligned)

# generate surf files and run VASP on pockets
# check to see if VASP scores and VASP pockets directory exists
if not os.path.exists(f'{outputdir}VASP'):
	os.mkdir(f'{outputdir}VASP')
if not os.path.exists(f'{outputdir}VASP/scores'):
	os.mkdir(f'{outputdir}VASP/scores')
if not os.path.exists(f'{outputdir}VASP/pockets'):
	os.mkdir(f'{outputdir}VASP/pockets')

# conformational sampling for initial screen
shortSample = genShortSample(initialdir)
s = len(shortSample)

# target pocket volume
vol = float([line.split() for line in open(f'{initialdir}vol.txt','r').readlines()][-1][-1])

for i,structure in enumerate(tracker):
	# dummy variables to guide conformational sampling
	tSamp = np.full(6,True)
	print(f'-----Running on: {structure}-----')
	
	# get surf file
	gen_surfs(outputdir,inputdir,structure)
	
	# run short screen of intersect VASP on each structure
	#### Counter is fucked up now #####
	intersect(outputdir,initialdir,structure,0,s,
				shortSample,full=False)
	
	# get scores and update scorefile
	extractScore(outputdir,structure)
	originalVolume(outputdir,structure)
	genScorefile(outputdir,structure,min_intersect,vol)
	
	result, tSamp = screenCheck(outputdir, structure, screen*vol, tSamp)

	# only perform full intersect sampling if initial screen passed,
	# also performs only translations in directions that pass Samp
	if np.any(np.append(result,tSamp)):
		print(f'-----Full Screen on: {structure}-----')
		longSample = genLongSample(initialdir,shortSample,result,tSamp)
		intersect(outputdir,initialdir,structure,i*1800,
				  len(tracker)*1800,longSample)

		# extract each score
		extractScore(outputdir,structure)

		# append scorefile
		genScorefile(outputdir,structure,min_intersect,vol)

	# prep for ROSETTA
	rosetta_prep(outputdir,inputdir,min_intersect,min_hits,structure)

# move scored structures ### currently broken
#moveScoredStructures(outputdir,inputdir)
