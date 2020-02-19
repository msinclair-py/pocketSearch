#!/usr/bin/env python
import glob, subprocess, math
import numpy as np
import argparse
import utils
from coordinate_manipulation import Coordinates
from aliases import *

#this script is predicated on you already having the desired
#pocket chosen. you can input the pocket number for excision
#and further analysis. this is the same as the chain number
#in the pdb e.g. STP 14 would correspond to pocket 14
#NOTE: if you run fpocket with the -i 80 flag the target pocket
#for 1m1n is any of pockets 1-4 due to being a homotetramer
parser = argparse.ArgumentParser()#description=__doc__)
parser.add_argument('pdb', help='PDB of target protein')
parser.add_argument('initial', help='Directory containing target protein')
parser.add_argument('pocket', help='Residue ID of pocket from fpocket')
parser.add_argument('-r', '--r_angle', default=30, dest='r', metavar='R', 
		help='Degree of rotation about z')
parser.add_argument('-t', '--t_angle', default=10, dest='t', metavar='T',
		help='Degree to tilt')
parser.add_argument('-s', '--shift', default=2, dest='s', metavar='S',
		help='Angstroms to translate by')

args = parser.parse_args()

if '.pdb' in args.pdb:
	pdb = args.pdb.split('.')[0]
else:
	pdb = args.pdb

if args.initial[-1] != '/':
	initial = f'{args.initial}/'
else:
	initial = args.initial

pocknumber = args.pocket

if args.r:
	deg1 = args.r
else:
	deg1 = 30

if args.t:
	deg2 = args.t
else:
	deg2 = 10

if args.s:
	tr = args.s
else:
	tr = 2

#here are just some housekeeping variables, functions and 
#initialized arrays to be used in the following script
rotates = int(360/float(deg1))
conformations = [-30,0,30,150,180,210]
pocket=[]
vec_residues=[]

###body of code starts here; get the pocket from fpocket output
with open(f'{initial}{pdb}_out/{pdb}_out.pdb',
	'r') as infile:
	for line in infile:
		if "STP" in line:
			if line.split()[5] == pocknumber:
				pocket.append(line)
			else:
				continue
		else:
			continue

#create new pocket pdb structure
with open(f'{initial}{pdb}.pocket.pdb', 'w') as out:
	[out.write(line) for line in pocket]

coords = Coordinates(initial,pdb,deg1,deg2,tr,rotates,0)
initialcoords = coords.getCoords()
centered = coords.center(initialcoords)
vector = coords.principal(centered)
aligned = coords.align(centered,vector)
coords.conformation(aligned,conformations)

#pass each aligned structure through SURF to generate the pocket surface
for i in glob.glob(f'{initial}*conf*'):
	stuff=i.split('/')[2]
	conf=stuff.split('_')[1]
	rotation=stuff.split('_')[2]
	tilt=stuff.split('_')[3]
	tran=stuff.split('_')[4]
	flip=stuff.split('_')[5]
	args=(surf,'-surfProbeGen',i,f'{initial}{pdb}.{conf}.{rotation}.{tilt}.{tran}.{flip}.SURF',3,.5)
	str_args = [ str(x) for x in args ]
	subprocess.run(str_args)

#get the volume of the initial pocket for reference in the scorefile
vargs = (surf,'-surveyVolume',f'{initial}{pdb}.conf0.0.0.0.0.SURF')
str_vargs = [ str(x) for x in vargs ]
out = open(f'{initial}vol.txt','w')
subprocess.run(str_vargs, stdout=out)
