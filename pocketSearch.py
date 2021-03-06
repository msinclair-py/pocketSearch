#!/usr/bin/env python
import os, glob, argparse
from utils import *
from coordinate_manipulation import Coordinates

# Define the parser
parser = argparse.ArgumentParser(description='Take user-defined run-time arguments')

# Declare an argument, and using the default value if the argument isn't given
parser.add_argument('pdbdir', help='Directory of PDBs to be tested against target')
parser.add_argument('outputdir', help='Output directory')
parser.add_argument('targetdir', 
		help='Directory containing target pocket conformations')
parser.add_argument('-m', '--mode',  dest='mode', nargs='?',
		choices=['random','skip'], metavar='M', default='skip',
		help='Mode of action, choices: skip (default), random download')
parser.add_argument('-a', '--alpha', dest='alpha', default=.80,
		metavar='A', help='Percentage cutoff for minimum number of alpha spheres in fpocket, \
							default behavior is .8 of target structure (80%).')
parser.add_argument('-c', '--cutoff', dest='cutoff', default=1.50,
		metavar='C', help='Percentage cutoff for maximum number of alpha spheres in fpocket, \
							default behavior is 1.5 of target structure (150%).')
parser.add_argument('-f', '--filter', dest='filt', default=0.7,
		metavar='F', help='Minimum shared %% identity, default behavior is .7 match (70%).')
parser.add_argument('-ht','--hits', dest='hilt', default=1,
		metavar='H', help='Minimum number of %% identity hits, default behavior is 1 hit.')
parser.add_argument('-r','--rand', dest='rand', default=100,
		metavar='R', help='Number of random structure to download, if random mode selected')
parser.add_argument('-s','--screen', dest='screen', default=0.5,
		metavar='S', help='Short screen filter to determine whether translational sampling occurs, \
							default screen percentage is .5 match (50%).')

# Now, parse the command line arguments and store the values in the arg variable
args = parser.parse_args()

# initialize all options to corresponding variables
pdbdir = args.pdbdir
outputdir = args.outputdir
targetdir = args.targetdir
alpha = float(args.alpha)
cutoff = float(args.cutoff)
min_intersect = float(args.filt)
min_hits = int(args.hilt)
rand = int(args.rand)
screen = float(args.screen)

# make sure directories have correct formatting
pdbdir = checkFormat(pdbdir)
outputdir = checkFormat(outputdir)
targetdir = checkFormat(targetdir)

# Different modes of pdb handling
if args.mode=='random':
	randomPDBs(pdbdir,rand,outputdir)
elif args.mode!='skip':
	print("No mode was selected for pdb acquisition, default is to skip downloading pdbs")

# rename everything to be consistent with typical naming schemes
formatPDBs(pdbdir)

# clean up pdbs, removing cofactors and renumbering where necessary
if not os.path.exists(f'{pdbdir}/original_pdbs/'):
	os.mkdir(f'{pdbdir}/original_pdbs/')

for unclean in glob.glob(f'{pdbdir}*.pdb'):
	protein = os.path.basename(unclean)
	if not os.path.exists(f'{pdbdir}original_pdbs/{protein[:-4]}_orig'):
		getInfo(protein,pdbdir)
		clean(unclean)

# get target pocket alpha sphere count for fpocket cutoff calculation
for target in glob.glob(f'{targetdir}*pocket*'):
	with open(target) as f:
		for i,l in enumerate(f):
			pass
		aspheres = i + 1

fpocket_min = int(aspheres * alpha)
fpocket_max = int(aspheres * cutoff)

# generate pockets
find_pockets(pdbdir,fpocket_min)

name_array=[os.path.basename(name) for name in glob.glob(f'{pdbdir}*_out/*_out.pdb')]

for entry in name_array:
	# write pockets that pass filters
	writePockets(pdbdir,entry,fpocket_max)

# identify cofactors that match each pocket that passes filter
identifyCofactors(pdbdir)

# center and align pockets
prealigned = []
for name in glob.glob(f'{pdbdir}*.pocket*.pdb'):
	# check if original pdb still there, if not it has been scored already
	# and this structure should be skipped
	stem = os.path.basename(name).split('.')
	if os.path.exists(f'{pdbdir}{stem[0]}.pdb'):
		prealigned.append([stem[0],stem[1]])

tracker = []
for entry in prealigned:
	print(entry)
	tracker.append(f'{entry[0]}.{entry[1]}')
	coordsystem = Coordinates(pdbdir,entry[0],pnum=entry[1])
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
shortSample = genShortSample(targetdir)
s = len(shortSample)
t = len(tracker)

# target pocket volume
vol = float([line.split() for line in open(f'{targetdir}vol.txt','r').readlines()][-1][-1])

for i,structure in enumerate(tracker):
	# dummy variables to guide conformational sampling
	tSamp = np.full(6,True)
	print(f'-----Running on: {structure}-----')
	
	# get surf file
	gen_surfs(outputdir,pdbdir,structure)
	
	# run short screen of intersect VASP on each structure
	intersect(outputdir,targetdir,structure,i,t,s,
				shortSample,full=False)
	
	# get scores and update scorefile
	extractScore(outputdir,structure)
	originalVolume(outputdir,structure)
	genScorefile(outputdir,pdbdir,structure,min_intersect,vol)
	
	result, tSamp = screenCheck(outputdir, structure, screen*vol, tSamp)

	# only perform full intersect sampling if initial screen passed,
	# also performs only translations in directions that pass Samp
	if np.any(np.append(result,tSamp)):
		print(f'-----Full Screen on: {structure}-----')
		longSample = genLongSample(targetdir,shortSample,result,tSamp)
		intersect(outputdir,targetdir,structure,i,t,
				  len(longSample),longSample)

		# extract each score
		extractScore(outputdir,structure)

		# append scorefile
		genScorefile(outputdir,pdbdir,structure,min_intersect,vol)

	# prep for ROSETTA
	rosetta_prep(outputdir,pdbdir,min_intersect,min_hits,structure)

	# remove surf/vasp files. they take up an enormous amount of storage
	# otherwise (~500Mb per structure ran)
	deleteSurfs(structure,outputdir)

	# move scored structures
	moveScoredStructures(outputdir,pdbdir)
