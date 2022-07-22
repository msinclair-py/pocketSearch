#!/usr/bin/env python
import os, glob, subprocess, random, Bio, shutil
import numpy as np
from scipy.spatial.distance import cdist
from aliases import fpocket,surf,vasp,pdblist
from Bio.PDB import PDBList

def randomPDBs(directory,rand,outdir):
	"""
	Download random PDBs from a supplied, curated list.
	Inputs:
		directory - PDB directory where PDBs will be downloaded
		rand - number of random PDBs to download
		outdir - output directory, list of downloaded PDBs generated here
	Outputs:
		Returns None; PDBs downloaded and list generated
	"""

	pdb1 = PDBList()
	PDB2list=[]
	array = np.genfromtxt(pdblist,dtype='unicode')
	
	samp = random.sample(range(0,len(array)), rand)
	for num in samp:
		PDB2list.append(array[num].strip())

	with open(f'{outdir}downloaded_structures.txt','w') as outfile:
		for i in PDB2list:
			pdb1.retrieve_pdb_file(i,pdir=directory,file_format='pdb')
			outfile.write(f'{i}\n')
	
	array = array[~np.isin(np.arange(array.size), samp)]
	with open(pdblist, 'w') as new_list:
		for entry in array:
			new_list.write(f'{entry}\n')


def checkFormat(directory):
	"""
	Reformats directory name for downstream consistency.
	Inputs:
		directory - directory to be checked
	Outputs:
		returns directory with last character '/'
	"""

	if directory[-1] != '/':
		directory = f'{directory}/'
	if not os.path.exists(directory):
		print('Directory doesn\'t exist')
	else:
		print('Directory exists')
	return directory


def formatPDBs(directory):
	"""
	Reformats file format for all files in 
	specified directory.
	Inputs:
		directory - directory to perform reformatting
	Outputs:
		returns None, all .ent files converted to .pdb
	"""

	for f in glob.glob(f'{directory}/*'):
		old = os.path.basename(f)
		if old[-4:] == '.ent':
			new = f'{old[3:-4]}.pdb'
			os.rename(f,f'{directory}{new}')


def getInfo(structure, directory):
	"""
    Obtain relevant information for each structure (protein name,
    experimental method, resolution, any cofactors).
    Inputs:
        structure - protein to extract info of
        directory - pdb directory where structure can be found
    Outputs:
        returns None, writes info to file in pdb directory
	"""
	
	# read file
	reader = []
	with open(f'{directory}{structure}','r') as f:
		for line in f:
			reader.append(line)
	
	# obtain title
	for line in reader:
		if 'COMPND' in line:
			if line[8:10] == ' 2':
				base = line.split(':')[-1]
				title = base.split(';')[0].strip()
		
	# obtain experimental info
	expArr = []
	for line in reader:
		if 'EXPDTA' in line:
			expArr.append(line)
		if 'RESOLUTION.' in line:
			expArr.append(line)

	if len(expArr) > 1:
		method = ' '.join(expArr[0].split()[1:])
		resolution = ' '.join(expArr[1].split()[-2:])
		exp_info = f'{method.strip()}: {resolution.strip()}'
	else:
		exp_info = 'NONE'

	# obtain cofactor info
	coflines = []
	for line in reader:
		if 'HETNAM' == line[:6]: 
			coflines.append(line)
	
	# go through extracted lines to get relevant cofactor information
	if coflines:
		multiples = 0
		cofs = []
	
		for i in range(len(coflines)):
			# identify the current cofactor
			current = coflines[i].split()[1]
		
			# if there are multiple lines the the 3-letter code will instead
			# be the line number of the current cofactor
			if len(current) < 2:
				multiples += 1
				# get entry index of current cofactor in the cofs array
				Idx = i - multiples
			
				# append cofactor name to existing entry
				joined = f'{cofs[Idx].split(":")[1].strip()}{"".join(coflines[i].split()[3:])}'
				cofs[Idx] = ': '.join([cofs[Idx].split(':')[0].strip(),joined,'Not present'])

			# first line of cofactor, just append to cofs
			else:
				cofactor = ' '.join(coflines[i].split()[2:])
				cofs.append(': '.join([current,cofactor,'Not present']))
	
		cofactors = '; '.join(cofs)
	
	else:
		cofactors = 'NONE'

	# write out all info to file
	if not os.path.exists(f'{directory}infofiles/'):
		os.mkdir(f'{directory}infofiles/')

	outfile = f'{directory}infofiles/{structure[:-4]}.info'
	with open(outfile, 'w') as out:
		out.write(f'{title}\n')
		out.write(f'{exp_info.strip()}\n')
		out.write(f'{cofactors.strip()}')                                 


def clean(structure): 
	"""
	PDBs straight from RCSB have a lot of information we don't care about.
	We need to isolate chain A and also filter out any alternative residues.
	It's ok to remove alt residues since we will be doing mutagenesis down the line.
	Finally for compatibility with downstream programs, atoms will be renumbered,
	starting from 1 with a simple mapping scheme.
	Inputs:
		struc - the particular structure to be cleaned, including filepath
	Outputs:
		returns None; writes out cleaned pdb over the old pdb file
	"""

	# get lines of atoms
	lines = [line for line in open(f"{structure}").readlines() if line.split()[0] == 'ATOM']
	# filter out all chains but chain A and also remove alternative residues
	filtered = [line for line in lines if line[21] == 'A' and line[22:26].strip().isdigit()]
	# only take highest occupancy atoms
	highest = [f'{line[:16]} {line[17:]}' for line in filtered if line[16] in [' ', 'A']]

	# renumber atoms beginning from 1
	# get all the resids for each line
	resid = np.array([line[22:26].strip() for line in highest]).astype(int)

	# need to map to new values
	uniqueValues = np.unique(resid)
	# perform mapping
	renumbered = [np.where(uniqueValues == x)[0][0] + 1 for x in resid]
	
	# put the new numbering back into each line, including TER line
	final = [f'{line[:22]}{renumbered[i]:>4}{line[26:]}' for i, line in enumerate(highest)]
    final = final + ['TER']

	# move original pdb file
	(dirname, filename) = os.path.split(structure)
	fpath = f'{dirname}/original_pdbs/'

	shutil.move(structure, f'{fpath}{filename[:-4]}_orig')

	with open(structure,'w') as outfile:
		for line in final:
			outfile.write(line)

    outfile.close()


def writePockets(directory, pock, maximum):
	"""
	Writes out pocket files according within the specified cutoffs.
	fpocket already filtered out any pockets with less than 80 dummy atoms,
	now pockets with over the max cutoff are excluded and all else are
	written into new pdbs.
	Inputs:
		directory - filepath of pdb file to test
		pock - pdb file to test for viable pockets
		maximum - cutoff filter for max dummy atoms
	Outputs:
		returns None, writes out files to directory
	"""
	
	infile = open(f'{directory}{pock[:-4]}/{pock}','r')
	pock_array = [line for line in infile if "STP" in line]
	pocketID = [int(line.split()[5]) for line in pock_array]

	uniq = np.unique(np.array(pocketID))
	for i in range(uniq.shape[0]):
		j=i+1
		a = pocketID.count(j)
		if a < maximum:
			outfile = open(f'{directory}{pock.split("_")[0]}.pocket{j}.pdb','w')
			[outfile.write(line) for line in pock_array if int(line[22:26].strip()) == uniq[i]]


def identifyCofactors(directory):
	"""
	Iterate through each generated pocket that passes filters to identify
	which, if any, cofactors would have inhabited the pocket. This is
	accomplished via a simple shortest distance algorithm on the pocket prior
	to centering and alignment.
	Inputs:
		directory - filepath to generated pockets
	Outputs:
		returns None, rewrites each .info file to reflect pocket matching each
			cofactor
	"""
	
	# generate a list of every pocket, skips over aligned pockets if this is
	# a restart run due to aligned pockets lacking the .pdb extension
	all_pockets = [ p for p in glob.glob(f'{directory}*pocket*.pdb') ]

	# go through each pocket to determine if it inhabits a cofactor's space
	for pocket in all_pockets:
		base = os.path.basename(pocket)
		pnum = base.split('.')[1]

		# location and name of corresponding infofile
		infodir = f'{directory}infofiles/'
		infofile = f'{infodir}{base[:4]}.info'
		
		# read whole infofile, specifically take the cofactor line for now
		infolines = [line for line in open(infofile).readlines()]
		cofactline = infolines[-1]
		
		if cofactline != 'NONE':
			# cofactor format is - 3 letter code: full name: pocket#/'Not present'; .....
			cofactors = [c.split(':')[0].strip() for c in cofactline.split(';')]

			# get pocket coordinates
			c = [[l[30:38].strip(),l[38:46].strip(),l[46:54].strip()] for l in open(pocket).readlines()]

			# extract all heteroatom coordinate lines from original pdb
			cof = [line for line in open(f'{directory}original_pdbs/{base[:4]}_orig').readlines() if line[:6] == 'HETATM']

			# generate array where 0 = cofactor not in pocket, 1 = cofactor in pocket
			# initially all cofactors set to 0
			cof_present = np.array([[c,0] for c in cofactors])
			for i, cofactor in enumerate(cofactors):
				# get specific cofactor coordinates
				c2 = [[l[30:38].strip(),l[38:46].strip(),l[46:54].strip()] for l in cof if l[17:20].strip() == cofactor]

				# measure minimum pairwise euclidean distance of cofactor and pocket coords
				d = np.min(np.min(cdist(c, c2, 'euclidean')))

				# cofactor occupacy of pocket defined as <1 angstrom, set array to 1
				if d < 1:
					cof_present[np.where(cof_present == cofactor)[0][0],1] = 1

			# identified array tracks any changes to make to infofile
			identified = []
			for i, cofactor in enumerate(cofactline.split(';')):
				# info contains just the info for the current cofactor
				info = [c.strip() for c in cofactor.split(':')]

				# if the entry for this cofactor in cof_present is '1', it occupies this pocket
				if cof_present[np.where(cof_present == info[0])[0][0],1] == '1':
					# if this cofactor HAS NOT been assigned a pocket, assign this one
					if info[-1] == 'Not present':
						info[-1] = pnum
					# if this cofactor HAS been assigned a pocket, append this one
					else:
						info[-1] = ', '.join([info[-1],pnum])
				identified.append(info)

			# generate the updated cofactor info line
			new_cofline = []
			for i in range(len(identified)):
				cur = ': '.join(identified[i])
				if i > 0:
					new_cofline = '; '.join([new_cofline, cur])
				else:
					new_cofline = cur

			# write out updated infofile
			with open(infofile,'w') as out:
				out.write(f'{infolines[0]}')
				out.write(f'{infolines[1]}')
				out.write(f'{new_cofline}')


def principal (array):
	"""
	Find the principal axis in a point cloud.
	Inputs:
		array - points to find principal axis of
	Outputs:
		axis1 - the principal axis of the points	
	"""

	inertia = array.T @ array
	e_values, e_vectors = np.linalg.eig(inertia)
	order = np.argsort(e_values)
	eval3,eval2,eval1 = e_values[order]
	axis3,axis2,axis1 = e_vectors[:,order].T
	return axis1


def align (array,a):
	"""
	Align a point cloud with its principal axis along
	the z-axis. This is done according to the formula:
	R = I + vx + vx^2/(1+c) where I is the identity 3x3,
	vx is a skew-symmetric of a1 x a2, and c is the dot 
	product a1*a2.
	Inputs:
		array - point cloud to align
		a - vector corresponding to principal axis
	Outputs:
		aligned - point cloud that has been aligned
	"""

	aligned=np.zeros((array.shape[0],array.shape[1])) 
	b = [0,0,1] 
	v = np.cross(a,b) 
	c = np.dot(a,b) 
	I = np.eye(3,3)
	vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
	R = I + vx + (vx @ vx)/(1+c)
	aligned = R @ array.T
	return aligned.T	


def center (array):
	"""
	Moves a point cloud so the geometric center is at the
	origin (0,0,0). Accomplished by subtracting each point
	by the geometric center of mass of the point cloud.
	Inputs:
		array - points to recenter
	Outputs:
		centered - array of recentered points
	"""

	centered=np.zeros((array.shape[0],array.shape[1]))
	com=array.mean(0)
	for i in range(array.shape[0]):
		centered[i] = array[i] - com
	return centered


def make_pdb (array,output):
	"""
	Writes a properly formatted PDB of an array of points.
	This appears to be a defunct function and will be culled
	in a later release.
	Inputs:
	Outputs:
	"""
	
	with open(f'{output}','w') as f:
		with open(f'{input}','r') as infile:
			i=0
			for line in infile:
				x = f'{array[i][0]:.3f}'
				y = f'{array[i][1]:.3f}'
				z = f'{array[i][2]:.3f}'
				f.write(f'{line[:27]}{x:>11}{y:>8}{z:>8} {line[55:]}')
				i+=1


def unique(list1): #######OBSOLETE#####
	"""
	Obtain only the unique entries of a list.
	Inputs:
		list1 - list to reduce to unique values
	Outputs:
		unique_list - list of unique values
	"""

	unique_list=[]
	for x in list1:
		if x not in unique_list:
			unique_list.append(x)
	return unique_list


def find_pockets(indir,alphas):
	"""
	Run fpocket for all pdbs in a given directory.
	Inputs:
		indir - directory containing structures to
				run fpocket on
		alphas - minimum alpha sphere cutoff, this
				is an option in fpocket
	Outputs:
		returns None, just runs fpocket and prints
			whether output directories contain files
	"""

	for i in glob.glob(f'{indir}*.pdb'):
		root = os.path.basename(i).split('.')[0]
		if not os.path.exists(f'{indir}{root}_out/'):
			args=(fpocket,'-f',i,'-i',alphas)
			str_args=[ str(x) for x in args ]
			subprocess.run(str_args)
	

def gen_surfs(outdir,indir,pocket):
	"""
	Run SURF to generate surf file of given pocket.
	Inputs:
		outdir - directory to output surf file to
		indir - filepath to pocket
		pocket - pdb ID and pocket number
	Outputs:
		returns None, runs SURF and outputs logfile
	"""

	pock = f'{indir}aligned.{pocket}'
	print(pocket)
	n = pocket.split('.')[0]
	p = pocket.split('.')[1]
	arg = (surf,'-surfProbeGen',pock,f'{outdir}VASP/pockets/{n}_{p}.SURF',3,.5)
	str_arg = [ str(x) for x in arg ]
	out = open(f'{outdir}VASP/pockets/surf.log','w')
	subprocess.run(str_arg, stdout=out)


def intersect(outdir,initial,pocket,snum,total,samples,catalogue,full=True):
	"""
	Run intersect VASP on given pocket.

	Inputs:
		outdir - output directory for VASP runs
		initial - directory containing target structures
				that will be intersected with given pocket
		pocket - pocket structure to be tested against target
		snum - structure number, for progress display
		total - total number of structures, for progress display
		samples - number of conformations being sampled for display
		catalogue - list of target structures to intersect
		full - kwarg that guides behavior of the function,
				when False additional analysis is performed
				to inform whether translations will occur in
				full sampling
	
	Outputs:
		None, runs VASP for user-specified catalogue
	"""
	count = 0
	struc = f'{outdir}VASP/pockets/{pocket.split(".")[0]}_{pocket.split(".")[1]}.SURF'
	n = f"{pocket.split('.')[0]}_{pocket.split('.')[1]}"

	# clean up outputs so that you can perform operations in
	# the VASP directory, otherwise too many files
	if not os.path.exists(f'{outdir}VASP/{n}/'):
		os.mkdir(f'{outdir}VASP/{n}/')

	for init in catalogue:
		count += 1
		if full:
			modulo = 25
		else:
			modulo = 5
		if (count%500==0):
			print(f'-----{snum+1}/{total} pockets to be sampled-----')
		if (count%modulo==0):
			print(f'{pocket}: {count}/{samples} conformations')

		fname = os.path.basename(init).split('.')
		conf = fname[1]
		rot = fname[2]
		tilt = fname[3]
		trans = fname[4]
		flip = fname[5]
		arg = (vasp,'-csg',struc,init,'I',f'{outdir}VASP/{n}/intersect.{n}.{conf}.{rot}.{tilt}.{trans}.{flip}.SURF',.5)
		str_arg = [ str(x) for x in arg ]
		out = open(f'{outdir}intersect.log','w')
		subprocess.run(str_arg, stdout=out)

	arg2 = (surf,'-surveyVolume',struc)
	str_arg2 = [ str(x) for x in arg2 ]
	out = open(f'{outdir}VASP/pockets/{n}.vol.txt','w')
	subprocess.run(str_arg2, stdout=out)


def genShortSample(targetDir):
	"""
	Generate the short sampling array.
	Inputs:
		targetDir - directory containing target structures
	Outputs:
		short - array of target structures that comprise the
				short screen
	"""

	short = glob.glob(f'{targetDir}*conf0.*.*.0.0.SURF')
	translations = glob.glob(f'{targetDir}*conf0.0.0.*.0.SURF')
	short = np.append(short,translations)
	short = np.unique(short)
	return short


def genLongSample(initialdir,short,_result,_tSamp):
	"""
	Generate the long sampling array. This is done by subtracting out
	the short array from the totality of structures. Also, the translational
	screen is applied to exclude any translations where necessary.
	Inputs:
		initialdir - directory containing target structures
		short - array of short screen structures
		_result - continue non-translated screening
		_tSamp - boolean array of translational sampling results
	Outpus:
		full - array of structures that comprise the full-length screen
	"""
	
	if _result:
		# remove the short sample from the full list
		full = glob.glob(f'{initialdir}*.SURF')
		full = np.setdiff1d(full,short)

		# remove appropriate translations from the full list
		for i,trans in enumerate(_tSamp):
			if not trans:
				translation = glob.glob(f'{initialdir}*conf*.*.*.{i+1}.*.SURF')
				full = np.setdiff1d(full,translation)

	else: # this means the non-translated screen failed
		full = np.array([])
		for i,trans in enumerate(_tSamp):
			if trans:
				translation = glob.glob(f'{initialdir}*conf*.*.*.{i+1}.*.SURF')
				full = np.append(full,translation)

	return full


def extractScore(outdir,pocket):
	"""
	Runs SURF to get the intersect score for all intersections performed
	on supplied pocket.
	Inputs:
		outdir - output directory where the scores will be put
		pocket - the pocket to obtain scores for
	Outputs
		returns None, runs SURF -surveyVolume to generate scores
	"""
	
	n, pock = pocket.split('.')[0], pocket.split('.')[1]
	for inter in glob.iglob(f"{outdir}VASP/{n}_{pock}/*.SURF"):
		p = os.path.basename(inter).split('.')
		na = p[1]
		co = p[2]
		ro = p[3]
		ti = p[4]
		tr = p[5]
		fl = p[6]
		arg = (surf,'-surveyVolume',inter)
		out = open(f'{outdir}VASP/{n}_{pock}/{na}_{co}_{ro}_{ti}_{tr}_{fl}_iscore.txt','w')
		str_arg = [ str(x) for x in arg ]
		subprocess.run(str_arg, stdout=out)


def screenCheck(outdir, pocket, cut, _tSamp):
	"""
	Check if a short screen was successful. Additionally, map out
	which translations to perform in full sampling.
	Inputs:
		outdir - output directory filepath for VASP scores
		pocket - which pocket to check screen on
		cut - volume cutoff for screening
		_tSamp = copy of tSamp boolean array, dictates trans sampling
	Outputs:
		result - whether short screen was successful or not
		_tSamp = updated copy of tSamp
	"""
	result = False
	n = pocket.split('.')[0]
	p = pocket.split('.')[1]

	# check small screen first
	sText = f'{outdir}VASP/{n}_{p}/{n}_{p}_conf0*_0_0_iscore.txt'
	values = np.array([float([lines.split() for lines in open(struc,'r')][-1][-1])
						for struc in glob.glob(sText)])
	if np.where(values > cut)[0].size > 0:
		result = True

	for i in range(len(_tSamp)):
		tInt = [line.split() for line in \
				open(f'{outdir}VASP/{n}_{p}/{n}_{p}_conf0_0_0_{i+1}_0_iscore.txt').readlines()][-1][-1]
		if float(tInt) < cut:
			_tSamp[i] = False

	return result, _tSamp


def originalVolume(outdir,p):
	"""
	Runs SURF -surveyVolume on supplied pocket.
	Inputs:
		outdir - directory where to output volume
		p - pocket to get original volume of
	Outputs:
		returns None, generates volume file for pocket
	"""

	v = f"{outdir}VASP/pockets/{p.split('.')[0]}_{p.split('.')[1]}.SURF"
	n = os.path.basename(v).split('.')[0]
	arg = (surf,'-surveyVolume',v)
	out = open(f'{outdir}VASP/pockets/{n}.vol.txt','w')
	str_arg = [ str(x) for x in arg ]
	subprocess.run(str_arg, stdout=out)


def genScorefile(outdir,pdbdir,struc,filt,vol):
	"""
	Generates scorefile. Checks if structure has been written to file before
	and outputs updated values if so. 
	Inputs:
		outdir - output directory containing filepath for scores and scorefile
		pdbdir - directory containing original pdbs and infofiles
		struc - pocket to append score to file
		filt - int% filter, used to count number of hits
		vol - volume of target pocket
	Outputs:
		returns None, appends values to scorefile
	"""

	print(struc)	
	pdb = struc.split('.')[0]
	pock = struc.split('.')[1]
	v = float([line.split() for line in open(f'{outdir}VASP/pockets/{pdb}_{pock}.vol.txt').readlines()][-1][-1])

	# list of all scorefiles to extract data from
	scores = [s for s in glob.glob(f'{outdir}VASP/{pdb}_{pock}/*_iscore.txt')]

	hitCounter = 0
	bestScore = 0
	curScore = 0
	
	print(f'-----Getting {pdb} {pock} Scores-----')
	# iterate through all scores and track hits and best int%
	for score in scores:
		curScore = float([lines.split() for lines in open(score).readlines()][-1][-1])
		if not bestScore or curScore > bestScore:
			bestScore = curScore
		if curScore/vol > filt:
			hitCounter+=1
		curScore = 0

	print('-----Updating Scorefile-----')
	header = 'PDB   Pock     Target Vol  Pock Vol   Int Vol  Int%  #hits'
	addl = 'Exp. Method                       Cofactor(s)                 Protein Name\n'
	header = '       '.join([header,addl])
	# read scorefile lines and check if structure has been reported before
	# if so we will inherit the current number of hits and update our counter
	if not os.path.exists(f'{outdir}score.txt'):
		with open(f'{outdir}score.txt','w') as sfile:
			sfile.write(header)
		sfile.close()
	
	currentScorefile = np.genfromtxt(f'{outdir}score.txt',delimiter=';',skip_header=1,dtype='unicode',autostrip=True)

	# if shape > 0 then array not empty; if size > 10 there is more than one entry.
	# this is relevant because the 1D array will throw IndexError with the array
	# indexing passed to search the first two columns
	if currentScorefile.shape[0] > 0:
		currentScorefile = currentScorefile.reshape(int(currentScorefile.size/10),10)
		if currentScorefile.size > 10:
			for i, line in enumerate(currentScorefile[:,:2]):
				if pdb in line:
					if pock in line:
						hitCounter += int(currentScorefile[i][6])
						currentScorefile = np.delete(currentScorefile,i,0)
		else:
			if pdb in currentScorefile:
				if pock in currentScorefile:
					hitCounter += int(currentScorefile[:,6])
					# if this is the first entry in the scorefile and it is also a duplicate,
					# the original line must be erased so there aren't duplicate entries.
					# this will trigger the downstream behavior to treat it as if it is the
					# first entry in the scorefile again.
					currentScorefile = np.array([])
	
	# add current structure to currentScorefile prior to sorting
	# concatenate requires same shape so reshape structure to be a 2d array
	# first, acquire the info on this structure
	info = [line for line in open(f'{pdbdir}infofiles/{pdb}.info').readlines()]
	# identify an possible cofactors
	for cof in info[-1].split(';'):
		if cof.split(':')[-1].strip() == pock:
			cofactor = ': '.join(cof.split(':')[0:2])
	if not cofactor:
		cofactor = 'None'

	currentStructure = np.array([pdb,pock,vol,v,bestScore,f'{float(bestScore)/float(vol):.3f}',hitCounter,info[1][:-2],cofactor,info[0][:-2]]).reshape(1,10)
	if currentScorefile.shape[0] == 0:
		currentScorefile = currentStructure
	else:
		if currentScorefile.ndim == 1:
			shape = 2
		else:
			shape = currentScorefile.shape[0]+1
		currentScorefile = np.append(currentScorefile,currentStructure).reshape(shape,10)

	# if there is more than one entry in the scorefile, sort it by int%
	# take the in% col [:,-2]; ensure it is a float; 
	# argsort is ascending [::-1] makes it descending
	if currentScorefile.shape[0] > 1:
		currentScorefile = currentScorefile[currentScorefile[:,5].astype(float).argsort()[::-1]]
	
	# being certain that the relevant info is correct rewrite scorefile
	with open(f'{outdir}score.txt','w') as sfile:
		sfile.write(header)
		for l in currentScorefile:
			line = f'{l[0]:<4};{l[1]:>8};{l[2]:>10.8};{l[3]:>10.8};{l[4]:>9.7};{float(l[5]):>{6}.{3}};{l[6]:>5}; {l[7]}; {l[8]}; {l[9]}'
			sfile.write(f'{line}\n')


def deleteSurfs(structure, outdir):
	"""
	In an effort to keep the data footprint small, delete surf files. this
	should only be called after scorefile has been written out.
	Inputs:
		structure - name of structure to delete files for
		outdir - directory in filepath of surf files
	Outputs:
		returns None, deletes appropriate files
	"""
	
	print(f'-----Cleaning up {structure} SURF/VASP files-----')
	n, p = structure.split('.')[0], structure.split('.')[1]
	shutil.rmtree(f'{outdir}VASP/{n}_{p}')
	os.remove(f'{outdir}VASP/pockets/{n}_{p}.SURF')
	os.remove(f'{outdir}VASP/pockets/{n}_{p}.vol.txt')


def moveScoredStructures(outdir,pdbdir):
	"""
	Function to move cleaned PDBs of structures that make it to the
	scorefile to an output directory.
	Inputs:
		outdir - output directory where scorefile is, also in the
				filepath of the PDB output directory
		pdbdir - directory where PDBs can be found initially
	Outputs:
		returns None, moves PDBs to output directory
	"""
	
	scores = np.genfromtxt(f'{outdir}score.txt',skip_header=1,delimiter=';',
							autostrip=True,dtype='unicode')
	
	if scores.size > 10:
		scored = scores[:,0]
	else:
		scored = scores[0]
	
	path = f'{outdir}scored_pdbs/'

	if not os.path.exists(path):
		os.mkdir(path)

	# if pdb still in pdbdir and not in scored pdbs, move it there
	for pdb in scored:
		if os.path.exists(f'{pdbdir}{pdb}.pdb'):
			if not os.path.exists(f'{path}{pdb}.pdb'):
				shutil.move(f'{pdbdir}{pdb}.pdb',path)


def rosetta_prep(outdir,indir,filt,hilt,pocket):
	"""
	This function generates rosetta pos files for each structure that
	passes both the int% and hit filters. The scorefile is read and for each
	success the fpocket output for that pocket is read and the resID
	is acquired for each residue that lines the pocket.
	Inputs:
		outdir - output directory where scorefile various outputs are
		indir - the pdb directory, also containing fpocket outputs
		filt - int% filter
		hilt - hit filter
		pocket - pdb ID and pocket number of each structure
	"""

	# check if the rosetta output directoyr exists, else make it
	if not os.path.exists(f'{outdir}rosetta'):
		os.mkdir(f'{outdir}rosetta')
	
	# get our list of pose files to make from the scorefile
	lst = np.genfromtxt(f'{outdir}score.txt', dtype='unicode', skip_header=1,
						delimiter=';',usecols=(0,1,5,6),autostrip=True)
	
	# this if statement ensures that no indexing errors occur if there is only
	# one good pocket in the group
	if lst.ndim == 1:
		array=[]
		pose_array=[]
		if float(lst[2]) > float(filt):
			if float(lst[3]) > float(hilt):
				a = lst[0]
				b = [ int(x) for x in lst[1] if x.isdigit() ]
				c=''
				for j in range(len(b)):
					c+=f'{b[j]}'
				fil = f'{indir}{a}_out/pockets/pocket{c}_atm.pdb'

				# need to check if fpocket messed up any pocket pdbs
				# we don't use any fancy text reading methods as some
				# of the fpocket errors will be difficult to parse without
				# going line by line as below
				lines = []
				with open(fil,'r') as prefile:
					for line in prefile:
						if line[:4] == 'ATOM':
							lines.append(line.split()[:6])
				
				lines = np.asarray(lines)
				pose_array = [resid for resid in lines[:,-1]]
				
				# a period indicates that something went awry with the pdb
				# generation during fpocket, we can just ignore these
				# residues
				for i in range(len(pose_array)):
					if pose_array[i] not in array:
						if "." in pose_array[i]:
							continue
						else:
							array.append(pose_array[i])

				array = [ int(x) if x[-1].isdigit() else int(x[:-1]) for x in array ]
				with open(f'{outdir}rosetta/{a}_pock{c}.pos','w') as outfile:
					for line in sorted(array):
						if line == sorted(array)[-1]:
							outfile.write(f'{line}')
						else:
							outfile.write(f'{line} ')

	else:
		for i in range(len(lst)):
			array=[]
			pose_array=[]

			if float(lst[i][2]) > float(filt):
				if float(lst[i][3]) > float(hilt):
					a = lst[i][0]
					b = [ int(x) for x in lst[i][1] if x.isdigit() ]
					c = ''
					for j in range(len(b)):
						c+=f'{b[j]}'
					fil = f'{indir}{a}_out/pockets/pocket{c}_atm.pdb'
					
					lines = []
					with open(fil, 'r') as prefile:
						for i,line in enumerate(prefile):
							if line[:4] == 'ATOM':
								lines.append(line.split()[:6])

					lines = np.asarray(lines)
					pose_array = [resid for resid in lines[:,-1]]

					for i in range(len(pose_array)):
						if pose_array[i] not in array:
							if "." in pose_array[i]:
								continue
							else:
								array.append(pose_array[i])

					array = [ int(x) if x[-1].isdigit() else int(x[:-1]) for x in array ]
					with open(f'{outdir}rosetta/{a}_pock{c}.pos','w') as outfile:
						for line in sorted(array):
							if line == sorted(array)[-1]:
								outfile.write(f'{line}')
							else:
								outfile.write(f'{line} ')


def updateCheckpoint(checkpoint_file, checkpoint_value):
    """
    Updates the first line of the checkpoint file. The checkpoint structure is
    (a) Index of which structure we are currently on (still indexes from 0)
    (b) One structure per line -> when loaded into an array
    Inputs:
        checkpoint_file - the full path and filename of the checkpoint
        checkpoint_value - the index corresponding to the most recently
                    COMPLETED pocketSearch run
    Returns:
        None
    """
    
    backup = f'{checkpoint_file}.BAK'
    shutil.copy(checkpoint_file, backup)
    
    f = open(backup)
    first_line, remainder = f.readline(), f.read()
    f.close()

    t = open(checkpoint_file, 'w')
    t.write(f'{checkpoint_value}\n')
    t.write(remainder)
    t.close()
    
    os.remove(backup)


def restartRun(chkpt):
    """
    Read in checkpoint file to restart a run.
    Inputs:
        chkpt - checkpoint filepath
    Returns:
        list of structures to run on
    """

    f = open(chkpt)
    idx, structures = int(f.readline().strip()), [line for line in f.read()]
    f.close()

    to_be_run = structures[i+1:]
    os.remove(chkpt)
    t = open(chkpt, 'w')
    t.write(to_be_run)
    t.close()

    return to_be_run
