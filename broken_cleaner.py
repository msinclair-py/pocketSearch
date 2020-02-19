#!/usr/bin/env python
import sys, os
from sys import stdout
from os.path import exists, basename
from optparse import OptionParser
from aminoacids import longer_names
from aminoacids import modres

shit_stat_insres = False
shit_stat_altpos = False
shit_stat_modres = False
shit_stat_misdns = False

fastaseq = {}

def check_and_print_pdb(count, residue_buffer, residue_letter):
	global pdbfile
	hasCA = False
	hasN = False
	hasC = False
	for line in residue_buffer:
		atomname = line[12:16]
		occupancy = float(line[55:60])
		if atomname == " CA " and occupancy > 0.0:
			hasCA = True
		if atomname == " N  " and occupancy > 0.0:
			hasN = True
		if atomname == " C  " and occupancy > 0.0:
			hasC = True

	if hasCA and hasN and hasC: ##this isnt evaluating to true ever
		for line in residue_buffer:
			newnum = '%4d ' % count
			line_edit = f'{line[0:22]}{newnum}{line[27:]}'
			pdbfile = pdbfile + line_edit

		chain = line[21]
		try:
			init = fastaseq[chain]
			fastaseq[chain] = "".join(init,residue_letter)
		except KeyError:
			fastaseq[chain] = residue_letter
		count+=1
		return True
	return False


def get_pdb_filename(name):
	if (os.path.exists(name)):
		return name
	if (os.path.exists(f'{name}.pdb')):
		return f'{name}.pdb'
	name=name.upper()
	if (os.path.exists(name)):
		return name
	if (os.path.exists(f'{name}.pdb')):
		return f'{name}.pdb'
	return None


def open_pdb(name):
	filename = get_pdb_filename(name)
	stem = os.path.basename(filename)
	if stem[-5:] == '.pdb1':
		stem = stem[:-5]
	if stem[-4:] == '.pdb':
		stem = stem[:-4]
	lines = open(filename, 'r').readlines()

	return lines, stem


parser = OptionParser(usage='%prog [options] <pdb> <chain id>')
options, args = parser.parse_args()

if len(args) != 2:
	parse.error('Must specify both the pdb and the chain id')

files_to_unlink = []

chainid = args[1]

lines, filename_stem = open_pdb(args[0])

oldresnum = '   '
count = 1

residue_buffer = []
residue_letter = ''

if chainid == '_':
	chainid = ' '

for line in lines:
	if line.startswith('ENDMDL'): break
	if len(line) > 21 and (line[21] in chainid):
		if line[0:4] != "ATOM" and line[0:6] != "HETATM":
			continue

		line_edit = line
		resn = line[17:20]

		if resn in modres.keys():
			orig_resn = resn
			resn = modres[resn]
			line_edit = f'ATOM {line[6:17]}{resn}{line[20:]}'
			
			if orig_resn == 'MSE':
				if (line_edit[12:14] == 'SE'):
					line_edit = f'{line_edit[0:12]} S{line_edit[14:]}'
				if len(line_edit) > 75:
					if (line_edi[76:78] == 'SE'):
						line_edit = f'{line_edit[0:76]} S{line_edit[78:]}'
			else:
				shit_stat_modres = True

		if not resn in longer_names.keys():
			continue

		resnum = line_edit[22:27]
		
		if not resnum == oldresnum:
			if residue_buffer != []:
				if not check_and_print_pdb(count, residue_buffer, residue_letter):
					shit_stat_misdns = True
				else:
					count+=1

			residue_buffer = []
			residue_letter = longer_names[resn]

		oldresnum = resnum

		insres = line[26]
		if insres != ' ':
			shit_stat_insres = True

		altpos = line[16]
		if altpos != ' ':
			shit_stat_altpos = True
			if altpos == 'A':
				line_edit = f'{line_edit[:16]} {line_edit[17:]}'
			else:
				continue

		residue_buffer.append(line_edit)

if residue_buffer != []:
	if not check_and_print_pdb(count, residue_buffer, residue_letter):
		shit_stat_misdns = True
	else:
		count+=1

flag_altpos = "---"
if shit_stat_altpos:
	flag_altpos = "ALT"
flag_insres = "---"
if shit_stat_insres:
	flag_insres = "INS"
flag_modres = "---"
if shit_stat_modres:
	flag_modres = "MOD"
flag_misdns = "---"
if shit_stat_misdns:
	flag_misdns = "DNS"

nres = len("".join(fastaseq.values()))

flag_successfule = "OK"
if nres <= 0:
	flag_successful = "BAD"

if chainid == ' ':
	chainid = '_'

print(filename_stem, "".join(chainid), nres, flag_altpos, flag_insres,
	flag_modres, flag_misdns, flag_successful)
if nres > 0:
	outfile = f'{filename_stem}_{chainid}.pdb'
	outid = open(outfile,'w')
	outid.write(pdbfile)
	outid.write('TER\n')
	outid.close()

	fastaid = stdout
	for chain in fastaseq:
		fastaid.write(f'>{filename_stem}_{chain}\n')
		fastaid.write(fastaseq[chain])
		fastaid.write('\n')
		handle = open(f'{filename_stem}_{"".join(chain)}.fasta','w')
		handle.write(f'>{filename_stem}_{"".join(chain)}\n')
		handle.write(fastaseq[chain])

if len(files_to_unlink) > 0:
	for file in files_to_unlink:
		os.unlink(file)
