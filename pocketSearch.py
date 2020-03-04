#!/usr/bin/env python
import os, glob
from utils import *
import argparse, math, os, glob
from coordinate_manipulation import Coordinates

# Define the parser
parser = argparse.ArgumentParser(description='Take user-defined run-time arguments')

# Declare an argument, and using the default value if the argument isn't given
parser.add_argument('inputdir', help='Directory of PDBs to be tested against target')
parser.add_argument('outputdir', help='Output directory')
parser.add_argument('initialdir', 
		help='Directory containing target pocket conformations')
parser.add_argument('-m', '--mode',  dest='mode', nargs='?',
		choices=['random','sublist','skip'], metavar='M', default='skip',
		help='Mode of action, options include:random, sublist and skip')
parser.add_argument('-a', '--alpha', dest='alpha', default=80,
		metavar='A', help='Cutoff for minimum number of alpha spheres in fpocket')
parser.add_argument('-c', '--cutoff', dest='cutoff', default=150,
		metavar='C', help='Cutoff for maximum number of alpha spheres in fpocket')
parser.add_argument('-f', '--filter', dest='filt', default=0.7,
		metavar='F', help='Minimum shared %% identity')
parser.add_argument('-ht','--hits', dest='hilt', default=1,
		metavar='H', help='Minimum number of %% identity hits')
parser.add_argument('-r','--rand',dest='rand', default=100,
		metavar='R', help='Number of random structure to download, if random mode selected')

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
rename(inputdir)

#######currently doesnt work for some reason
# clean up pdbs, removing cofactors and renumbering where necessary
for i in glob.glob(f'{inputdir}*'):
	inp = f'python cleaner.py {i} A {inputdir}'
	os.system(inp)
	name = os.path.basename(i)[:-4]
	os.remove(i)
	if os.path.exists(f'{inputdir}{name}_A.pdb'):
		os.rename(f'{inputdir}{name}_A.pdb',f'{inputdir}{name}.pdb')


# generate pockets
find_pockets(inputdir,alpha)

name_array=[]
for name in glob.glob(f'{inputdir}*_out/*_out.pdb'):
	name_array.append(os.path.basename(name))

for entry in name_array:
	pock_array=[]
	pocket_id=[]
	uniq=[]
	with open(f'{inputdir}{entry[:-4]}/{entry}','r') as infile:
		for line in infile:
			if not "STP" in line:
				continue
			elif "STP" in line:
				pock_array.append(line)
				pocket_id.append(line.split()[5])
	
	uniq = unique(pocket_id)
	for i in range(0,len(uniq)):
		j=i+1
		a = pock_array.count(j)
		if a < cutoff:
			with open(f'{inputdir}{entry.split("_")[0]}.pocket{j}.pdb','w') as outfile:
				for line in pock_array:
					if line.split()[5] == uniq[i]:
						outfile.write(line)
					else:
						continue
		else:
			continue

# center and align pockets
prealigned=[]
for name in glob.glob(f'{inputdir}*.pocket*'):
	stem = os.path.basename(name).split('.')
	prealigned.append([stem[0],stem[1]])


###########################################use the class now##################
for entry in prealigned:
	print(entry)
	coordsystem = Coordinates(inputdir,entry[0],pnum=entry[1])
	coords = coordsystem.getCoords()
	centered = coordsystem.center(coords)
	vector = coordsystem.principal(centered)
	aligned = coordsystem.align(centered,vector)
	coordsystem.makePDB(aligned)

### generate surf files and run VASP on pockets
# check to see if VASP scores and VASP pockets directory exists
if not os.path.exists(f'{outputdir}VASP'):
	os.mkdir(f'{outputdir}VASP')
if not os.path.exists(f'{outputdir}VASP/scores'):
	os.mkdir(f'{outputdir}VASP/scores')
if not os.path.exists(f'{outputdir}VASP/pockets'):
	os.mkdir(f'{outputdir}VASP/pockets')

# get surf file of each structure
gen_surfs(outputdir,inputdir)

# run intersect VASP on each structure
intersect(outputdir,initialdir)

# extract each score
extract_score(outputdir)

# get the original volume of each pocket
original_volume(outputdir)

# generate scorefile
generate_scorefile(outputdir,initialdir,min_intersect)

# prep for ROSETTA
rosetta_prep(outputdir,inputdir,min_intersect,min_hits)
