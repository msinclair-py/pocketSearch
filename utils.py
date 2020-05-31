#!/usr/bin/env python
import os, glob, subprocess, random, Bio
import numpy as np
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
	highest = [f'{line[:16]} {line[17:]}' for line in filtered if line[16] == ' ' or line[16] == 'A']

	# renumber atoms beginning from 1
	# get all the resids for each line
	resid = np.array([line[22:26].strip() for line in highest]).astype(int)

	# need to map to new values
	uniqueValues = np.unique(resid)
	# perform mapping
	renumbered = [np.where(uniqueValues == x)[0][0] + 1 for x in resid]
	
	# put the new numbering back into each line
	final = [f'{line[:22]}{renumbered[i]:>4}{line[26:]}' for i, line in enumerate(highest)]

	with open(structure,'w') as outfile:
		for line in final:
			outfile.write(line)


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


def intersect(outdir,initial,pocket,count,total,catalogue,full=True):
	"""
	Run intersect VASP on given pocket.

	Inputs:
		outdir - output directory for VASP runs
		initial - directory containing target structures
				that will be intersected with given pocket
		pocket - pocket structure to be tested against target
		count - running count metric for progress display
		total - total count metric for progress display
		catalogue - list of target structures to intersect
		full - kwarg that guides behavior of the function,
				when False additional analysis is performed
				to inform whether translations will occur in
				full sampling
	
	Outputs:
		None, runs VASP for user-specified catalogue
	"""

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
		if (count%modulo==0):
			print(f'{pocket}: {count}/{total}')

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
	
	for inter in glob.iglob(f"{outdir}VASP/{pocket.split('.')[0]}_{pocket.split('.')[1]}/*"):
		p = os.path.basename(inter).split('.')
		na = p[1]
		co = p[2]
		ro = p[3]
		ti = p[4]
		tr = p[5]
		fl = p[6]
		arg = (surf,'-surveyVolume',inter)
		out = open(f'{outdir}VASP/scores/{na}_{co}_{ro}_{ti}_{tr}_{fl}_iscore.txt','w')
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
	sText = f'{outdir}VASP/scores/{n}_{p}_conf0*_0_0_iscore.txt'
	values = np.array([float([lines.split() for lines in open(struc,'r')][-1][-1])
						for struc in glob.glob(sText)])
	if np.where(values > cut)[0].size > 0:
		result = True

	for i in range(len(_tSamp)):
		tInt = [line.split() for line in \
				open(f'{outdir}VASP/scores/{n}_{p}_conf0_0_0_{i+1}_0_iscore.txt').readlines()][-1][-1]
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


def genScorefile(outdir,struc,filt,vol):
	"""
	Generates scorefile. Checks if structure has been written to file before
	and outputs updated values if so. 
	Inputs:
		outdir - output directory containing filepath for scores and scorefile
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
	scores = [s for s in glob.iglob(f'{outdir}VASP/scores/*{pdb}_{pock}*')]

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
	header = 'PDB   Pock      Target Vol   Pock Vol    Int Vol   Int %  #hits\n'
	# read scorefile lines and check if structure has been reported before
	# if so we will inherit the current number of hits and update our counter
	if not os.path.exists(f'{outdir}score.txt'):
		with open(f'{outdir}score.txt','w') as sfile:
			sfile.write(header)
		sfile.close()
	
	currentScorefile = np.genfromtxt(f'{outdir}score.txt',skip_header=1,dtype='unicode')

	# if shape > 0 then array not empty; if size > 7 there is more than one entry.
	# this is relevant because the 1D array will throw IndexError with the array
	# indexing passed to search the first two columns
	if currentScorefile.shape[0] > 0:
		if currentScorefile.size > 7:
			for i, line in enumerate(currentScorefile[:,:2]):
				if pdb in line:
					if pock in line:
						hitCounter += int(currentScorefile[i][-1])
						currentScorefile = np.delete(currentScorefile,i,0)
		else:
			if pdb in currentScorefile:
				if pock in currentScorefile:
					hitCounter += int(currentScorefile[-1])
	
	# add current structure to currentScorefile prior to sorting
	# concatenate requires same shape so reshape structure to be a 2d array
	currentStructure = np.array([pdb,pock,vol,v,bestScore,f'{float(bestScore)/float(vol):.3f}',hitCounter]).reshape(1,7)
	if currentScorefile.shape[0] == 0:
		currentScorefile = currentStructure
	else:
		if currentScorefile.ndim == 1:
			shape = 2
		else:
			shape = currentScorefile.shape[0]+1
		currentScorefile = np.append(currentScorefile,currentStructure).reshape(shape,7)

	# if there is more than one entry in the scorefile, sort it by int%
	# take the in% col [:,-2]; ensure it is a float; 
	# argsort is ascending [::-1] makes it descending
	if currentScorefile.shape[0] > 1:
		currentScorefile = currentScorefile[currentScorefile[:,-2].astype(float).argsort()[::-1]]
	
	# being certain that the relevant info is correct rewrite scorefile
	with open(f'{outdir}score.txt','w') as sfile:
		sfile.write(header)
		for l in currentScorefile:
			line = f'{l[0]:<6}{l[1]:<8}{l[2]:{12}.{8}}{l[3]:{12}.{8}}{l[4]:{10}.{7}}{float(l[5]):.3f}{l[6]:>5}'
			sfile.write(f'{line}\n')



def generate_scorefile(outdir,struc,filt): #####OBSOLETE
	"""
	Generates formatted scorefile for all existing intersect scores.
	Inputs:
		outdir - output directory for scorefile, also in the filepath
				of the intersect scores
		struc - structure to be scored
		filt - int% filter
	Outputs:
		returns None, writes scorefile in output directory
	"""
	
	files,name_array,inter,pockets=[],[],[],[]
	for name in glob.iglob(f'{outdir}VASP/scores/*iscore.txt'):
		score = os.path.basename(name)
		files.append(score)
	
	for name in files:
		name_array.append(name.split('_'))
	
	print('-----Getting Scores-----')
	for i, line in enumerate(name_array):
		name = f'{outdir}VASP/scores/{files[i]}'
		a=name_array[i]
		with open(name,'r') as infile:
			for line in infile:
				pass
			iline = float(line.split()[3])
			inter.append([a[0],a[1],a[2],a[3],a[4],a[5],a[6],iline])

	print('-----Sorting Scores-----')
	inter_sort,inter_final=[],[]
	inter_sort = sorted(inter,key=lambda x: x[-1], reverse=True)

	with open(f'{initial}vol.txt','r') as f:
		for line in f:
			pass
		vline = line.split()[-1]
	
	vols={}
	for p in inter_sort:
		with open(f'{outdir}VASP/pockets/{p[0]}_{p[1]}.vol.txt','r') as f:
			for line in f:
				pass
			key = f'{p[0]} {p[1]}'
			vol = line.split()[-1]
			vols.update({key: vol})
	
	tracker=[]
	hits={}

	for j in range(len(inter_sort)):
		key = f'{inter_sort[j][0]} {inter_sort[j][1]}'
		if [inter_sort[j][0],inter_sort[j][1]] not in tracker:
			tracker.append([inter_sort[j][0],inter_sort[j][1]])
			inter_final.append(inter_sort[j])
			hits.update({key: 0})

		if float(inter_sort[j][-1])/float(vline) > float(filt):
			count = hits.get(key)+1
			hits.update({key : count})

	print('-----Outputting scores-----')
	print(inter_final)
	with open(f'{outdir}score.txt','w') as outfile:
		outfile.write('PDB   Pock      Target Vol   Pock Vol      Int Vol   Int %  # hits\n')
		for i in range(len(inter_final)):
			a=inter_final[i][0]
			p=inter_final[i][1]
			b=hits.get(f'{a} {p}')
			v=vols.get(f'{a} {p}')
			c=float(inter_final[i][-1])/float(vline)
			d=inter_final[i][-1]
			out=f"{a:<6}{p:<11}{vline:{12}.{8}}{v:{12}.{8}} {d:{9}.{7}} {c:{6}.{3}}    {b}\n"
			outfile.write(out)



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
	
	scores = np.genfromtxt(f'{outdir}score.txt',skip_header=1,dtype='unicode')

	# list comprehension to get the pdb ID for all xxxx.pdb files
	pdbs = [pdb.split('/')[-1][:4] for pdb in glob.glob(f'{pdbdir}*.pdb') if len(pdb.split('/')[-1]) == 8]
	
	if scores.size > 7:
		notScored = np.setdiff1d(scores[:,0],np.array(pdbs))
	else:
		notScored = np.setdiff1d(scores[0],np.array(pdbs))

	# if there any unscored pdbs remove them from the pdb list using
	# a boolean mask with numpy.isin function
	if notScored.size > 0:
		scored = np.setdiff1d(pdbs,notScored)
	
	path = f'{outdir}scored_pdbs/'

	if not os.path.exists(path):
		os.mkdir(path)

	for pdb in scored:
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
		usecols=(0,1,5,6))
	
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
