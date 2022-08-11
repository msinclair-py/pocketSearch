#!/usr/bin/env python
import argparse, glob, os, subprocess
import numpy as np
import ray
from typing import List
from coordinate_manipulation import Coordinates
from aliases import *

# this script is predicated on you already having the desired
# pocket chosen. you can input the pocket number for excision
# and further analysis. this is the same as the chain number
# in the pdb e.g. STP 14 would correspond to pocket 14
parser = argparse.ArgumentParser()
parser.add_argument('pdb', help='PDB of target protein (e.g. 1m1n or 1m1n.pdb)')
parser.add_argument('filepath', help='Directory containing all the fpocket outputs and original pdb')
parser.add_argument('pocketID', help='Residue ID of pocket from fpocket')
parser.add_argument('-r', '--r_angle', default=30., dest='r', metavar='R', 
		help='Degree of rotation about z')
parser.add_argument('-t', '--t_angle', default=10., dest='t', metavar='T',
		help='Degree to tilt')
parser.add_argument('-s', '--shift', default=2., dest='s', metavar='S',
		help='Angstroms to translate by')

args = parser.parse_args()

pdb = args.pdb.split('.')[0] if '.pdb' in args.pdb else args.pdb
fpath = args.filepath if args.filepath[-1] == '/' else f'{args.filepath}/'
pNum = int(args.pocketID)
deg1 = float(args.r)
deg2 = float(args.t)
tr = float(args.s)

# here are just some housekeeping variables, functions and 
# initialized arrays to be used in the following script
rotates = int(360/float(deg1))
conformations = [-30,0,30,150,180,210]
NConf = 350*len(conformations)

###body of code starts here; get the pocket from fpocket output
fp = f'{fpath}{pdb}_out/{pdb}_out.pdb'
pocket = [line for line in open(fp).readlines() \
			if 'STP' in line and line[23:29].strip() == f'{pNum}']

# create new pocket pdb structure
with open(f'{fpath}{pdb}.pocket.pdb', 'w') as out:
    for line in pocket:
        out.write(line)

# generate all the conformations
coords = Coordinates(fpath, pdb, deg1=deg1, deg2=deg2, translation=tr, rotate=rotates, mode=0)
coords.get_coords()

print('-----Centering and Aligning Coordinate System-----')
coords.center()
coords.principal()
coords.align()

print(f'-----Generating {NConf} Conformations-----')
coords.conformation(conformations)

# pass each conformation through SURF to generate the pocket surface
print(f'-----Generating {NConf} SURF Files-----')
all_conformations = glob.glob(f'{fpath}*_conf*')

@ray.remote
def generate_surface_files(structure: List[str]) -> None:
#    for c in structures:
    attributes=os.path.basename(structure)
    pdb, conf, rotation, tilt, tran, flip = attributes.split('_')
    
    args=(surf, '-surfProbeGen', structure, 
            f'{fpath}{pdb}.{conf}.{rotation}.{tilt}.{tran}.{flip}.SURF', 
            3, .5) # 3 and .5 are defaults for SURF parameters
    str_args = [ str(x) for x in args ]
    subprocess.run(str_args)

ray.init()
futures = [generate_surface_files.remote(struc) for struc in all_conformations]
_ = ray.get(futures)

# get the volume of the initial pocket for reference in the scorefile
vargs = (surf,'-surveyVolume',f'{fpath}{pdb}.conf0.0.0.0.0.SURF')
str_vargs = [ str(x) for x in vargs ]
out = open(f'{fpath}vol.txt','w')
subprocess.run(str_vargs, stdout=out)
