#!/usr/bin/env python
import argparse
from coordinate_manipulation import Coordinates
import glob
import os
import numpy as np
from pathlib import Path
import subprocess
import ray
from typing import Union
import yaml

PathLike = Union[str, Path]

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

pdb = Path(args.pdb).stem 
fpath = Path(args.filepath)
pNum = int(args.pocketID)
deg1 = float(args.r)
deg2 = float(args.t)
tr = float(args.s)

# unpack alias yaml
aliases = yaml.safe_load(open('aliases.yaml'))
surf = aliases['surf']

# here are just some housekeeping variables, functions and 
# initialized arrays to be used in the following script
rotates = int(360 / float(deg1))
conformations = [-30, 0, 30, 150, 180, 210]
NConf = 350 * len(conformations)

###body of code starts here; get the pocket from fpocket output
fp = fpath / f'{pdb}_out/{pdb}_out.pdb'
pocket = [line for line in open(fp).readlines()
          if 'STP' in line and line[23:29].strip() == f'{pNum}']

# create new pocket pdb structure
with open(fpath / f'{pdb}.pocket.pdb', 'w') as out:
    for line in pocket:
        out.write(line)

# generate all the conformations
coords = Coordinates(fpath, 
                     pdb, 
                     deg1=deg1, 
                     deg2=deg2, 
                     translation=tr, 
                     rotate=rotates, 
                     mode=0)
coords.get_coords()

print('-----Centering and Aligning Coordinate System-----')
coords.center()
coords.principal()
coords.align()

print(f'-----Generating {NConf} Conformations-----')
coords.conformation(conformations)

# pass each conformation through SURF to generate the pocket surface
print(f'-----Generating {NConf} SURF Files-----')
all_conformations = fpath.glob('*_conf*')

@ray.remote
def generate_surface_files(structure: PathLike) -> None:
    attributes = '.'.join(structure.name.split('_'))
    args=(surf, 
          '-surfProbeGen',
          structure, 
          fpath / f'{attributes}.SURF', 
          '3', 
          '.5') # 3 and .5 are defaults for SURF parameters
    subprocess.run(args)

ray.init()
futures = [generate_surface_files.remote(struc) for struc in all_conformations]
_ = ray.get(futures)

# get the volume of the initial pocket for reference in the scorefile
vargs = (surf, '-surveyVolume', fpath / f'{pdb}.conf0.0.0.0.0.SURF')
out = open(fpath / 'vol.txt','w')
subprocess.run(vargs, stdout=out)
