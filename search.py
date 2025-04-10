from coordinate_manipulation import Coordinates
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
import shutil
import subprocess
from typing import List, Tuple, Union
import yaml

PathLike = Union[str, Path]

class pocketSearcher:
    """
    Main pocketSearch class. Reads in software paths from a yaml which is passed
    to the config kwarg. Main function is called `search` and all pre- and post- 
    processing is automatically handled by both the init and search methods.
    Arguments:
        pdb_directory (PathLike): Path to where the scaffold PDBs are located.
        out_directory (PathLike): Path to where the various file outputs should
            be placed, including SURF/VASP outputs and scorefiles.
        tgt_directory (PathLike): Path to where the Target SURF files are located
            for the desired catalytic pocket.
        min_alpha_spheres (float): Minimum proportion of the target pocket number
            of alpha spheres that a scaffold pocket must possess to be considered.
            Defaults to 0.8.
        max_alpha_spheres (float): Maximum proportion of the target pocket number
            of alphas spheres that a scaffold pocket can possess to be considered.
            Defaults to 1.5.
        min_intersect (float): Minimum proportion of intersection % to be considered
            a "hit". Defaults to 0.7.
        min_hits (int): Minimum number of "hits" for a scaffold to be advanced. 
            Defaults to 1.
        screen (float): Short screen filter intersect % filter. We are more permissive
            in the short screens to reduce false negatives for pockets that may
            otherwise be good. Defaults to 0.5.
        checkpoint (PathLike): Path to a previous checkpoint file. Defaults to
            `checkpoint.ckpt`.
        config (PathLike): A YAML configuration file which specifies the path
            to the fpocket, surf and vasp binaries. Defaults to `aliases.yaml`.
    """
    def __init__(self, 
                 pdb_directory: PathLike, 
                 out_directory: PathLike,
                 tgt_directory: PathLike,
                 min_alpha_spheres: float=0.8,
                 max_alpha_spheres: float=1.5,
                 min_intersect: float=0.7,
                 min_hits: int=1,
                 screen: float=0.5,
                 checkpoint: PathLike=Path('checkpoint.ckpt'),
                 config: PathLike='aliases.yaml'):
        self.pdb_dir = pdb_directory
        self.out_dir = out_directory
        self.tgt_dir = tgt_directory
        self.min_cut = min_alpha_spheres
        self.max_cut = max_alpha_spheres
        self.min_int = min_intersect
        self.min_hit = min_hits
        self.screen = screen

        self.restart_run() if checkpoint.exists() else self.preprocess()

        try:
            cfg = yaml.safe_load(open(config))
            self.fpocket = cfg['fpocket']
            self.surf = cfg['surf']
            self.vasp = cfg['vasp']
        except KeyError:
            raise KeyError('Configuration YAML missing one of the following: \
                           `fpocket`, `surf`, `vasp`!!!')

        self.preprocess()
        self.gen_scorefile()

    def preprocess(self) -> None:
        """

        """
        self.format_pdbs()
        (self.pdb_dir / 'original_pdbs').mkdir(exist_ok=True)

        for unclean in self.pdb_dir.glob('*.pdb'):
            self.get_info(unclean)
            self.clean(unclean)

        aspheres = max(
            [len(open(target).readlines()) + 1 for target in self.tgt_dir.glob('*pocket*')]
        )

        self.find_pockets(aspheres)

        name_array = [pdb.name for pdb in self.pdb_dir.glob('*_out/*_out.pdb')]

        print('Writing out pockets...')
        for entry in name_array:
            print(f'Extracting pockets from {entry}')
            self.write_pockets(entry, aspheres)

        self.identify_cofactors()

        print('Processing pockets ...')
        for pdb in self.pdb_dir.glob('*.pocket*.pdb'):
            stem = pdb.name.split('.')
            if (self.pdb_dir / f'{stem[0]}.pdb').exists():
                coordsystem = Coordinates(stem[0], pnum=stem[1])
                coordsystem.get_coords()
                coordsystem.center()
                coordsystem.principal()
                coordsystem.make_pdb(coordsystem.align())

        self.vasp_dir = self.out_dir / 'VASP'
        self.score_dir = self.vasp_dir / 'scores'
        self.pocket_dir = self.vasp_dir / 'pockets'
        self.rosetta_dir = self.out_dir / 'rosetta'

        for d in (self.vasp_dir, self.score_dir, self.pocket_dir, self.rosetta_dir):
            d.mkdir(exist_ok=True)

        print('Preprocessing complete ...')

    def gen_scorefile(self) -> None:
        self.score_file = self.out_dir / 'scores.csv'

        if self.score_file.exists():
            self.scores = pd.read_csv(self.score_file)
        else:
            self.scores = pd.DataFrame(columns=['PDB', 
                                                'Pocket', 
                                                'Target Vol.', 
                                                'Pocket Vol.',
                                                'Int. Vol.',
                                                'Int%',
                                                '#hits',
                                                'Exp. Method', 
                                                'Cofactor(s)',
                                                'Protein Name'])

    def search(self,
               structure: PathLike):
        """
        Main function.
        """
        translation_sampling = np.full(6, True)
        print(f'-----Running on: {structure}-----')

        self.gen_surfs()
        self.intersect()
        self.extract_score()
        self.original_volume()

        result, translation_sampling = self.screen_check()

        if np.any(np.append(result, translation_sampling)):
            print(f'-----Full screen on: {structure}-----')
            long_sample = self.generate_long_sample(result, translation_sampling)
            self.intersect()

            self.extract_score()

        self.append_scorefile()
        self.rosetta_prep()
        self.update_checkpoint()
        self.delete_surfs()

    def format_pdbs(self) -> None:
        """
        Moves all .ent files into .pdb files.
        """
        for f in self.pdb_dir:
            if f.suffix == '.ent':
                new_f = f.with_suffix('.pdb')
                os.rename(f, new_f)

    def get_info(self,
                 pdb: PathLike) -> None:
        """
        Obtain relevant information for each structure from the HEADER
        section (protein name, exp. method, resolution, cofactors, etc).
        Arguments:
            pdb (PathLike): Path to PDB for which to obtain info.
        """
        directory = pdb.parent
        structure = pdb.name
        print(structure)

        reader = [line for line in open(pdb).readlines()]

        exp_info = [None, None]
        cofactors = []
        for line in reader:
            if 'COMPND   2' in line:
                title = line[20].strip()
            elif 'EXPDATA' in line:
                method = line.replace(';', ',')
                method = ' '.join(method.split()[1:])
                exp_info[0] = method.strip()
            elif 'RESOLUTION' in line:
                resolution = ' '.join(line.split()[-2:])
                exp_info[1] = resolution.strip()
            elif 'HETNAM' == line[:6]:
                cofactors.append(line.split()[1:])

        exp_info = ': '.join(exp_info)

        if cofactors:
            cofs = []
            idx = 0
            cofs.append([cofactors[0][0], ' '.join(cofactors[0][1:])])

            for cofactor in cofactors[1:]:
                if cofactor[1] == cofs[idx][0]:
                    cofs[idx][1] = cofs[idx][1] + ''.join(cofactor[2:])
                else:
                    cofs.append([cofline[0], ' '.join(cofactor[1:])])

            cofactors = ';'.join([': '.join(cofactor) for cofactor in cofactors])

        else:
            cofactors = 'NONE'

        info_path = directory / 'infofiles'
        info_path.mkdir(exist_ok=True)

        outfile = info_path / structure.with_suffix('.info')
        with open(outfile, 'w') as out:
            out.write(title + '\n')
            out.write(exp_info.strip() + '\n')
            out.write(cofactors.strip())

    def clean(self,
              pdb: PathLike) -> None:
        """
        Strips header and other unimportant data from the PDB (like CONECT records).
        Arguments:
            pdb (PathLike): Path to PDB to be cleaned
        """
        # strip out any lines we care about
        filtered = []
        for line in open(pdb).readlines():
            if all(['ATOM' in line, # normal biomolecule
                    line[21] == 'A', # we only want chain A
                    line[22:26].strip().isdigit(), # has resid
                    line[16] in [' ', 'A']]): # in case there is an NMR ensemble
                filtered.append(f'{line[:16]} {line[17:]}')
        
        # renumber residues
        resids = np.array([line[22:26].strip() for line in filtered], dtype=int)
        renumbered = [np.where(np.unique(resids) == x)[0][0] + 1 for x in resids]
        new_lines = [f'{line[:22]}{renum:>4}{line[26:]}' 
                     for line, renum in zip(filtered, renumbered)]
        new_lines += ['TER']

        path = pdb.parent

        old_pdb_path = path / 'original_pdbs'
        old_pdb_path.mkdir(exist_ok=True)

        shutil.move(pdb, old_pdb_path / pdb.name)

        with open(pdb, 'w') as outfile:
            outfile.write('\n'.join(new_lines))

    def find_pockets(self,
                     n_spheres: int) -> None:
        """
        Runs fpocket on all pdbs in `self.pdb_dir`.
        Arguments:
            n_spheres (int): Number of alpha spheres detected in parent pocket
        """
        min_spheres = n_spheres * self.min_cut

        for pdb in self.pdb_dir.glob('*.pdb'):
            root = pdb.name.split('.')[0]
            if not (self.pdb_dir / f'{root}_out').exists():
                print(f'-----Running fpocket on {root}-----')
                args = (fpocket, '-f', pdb, '-i', str(min_spheres))
                subprocess.run(args)
    
    def write_pockets(self,
                      pdb: PathLike,
                      n_spheres: int) -> None:
        """
        Extracts all pockets from an input pdb which meet our maximum number of
        alpha spheres cutoff.
        Arguments:
            pdb (PathLike): Path to pdb to grab pockets from
            n_spheres (int): Number of alpha spheres in our parent pocket
        """
        max_spheres = n_spheres * self.max_cut

        pocket_IDs = [int(line[22:26].strip()) 
                      for line in open(self.pdb_dir / pdb.stem / pdb.name).readlines()
                      if 'STP' in line]
        
        unique = np.unique(np.array(pocket_IDs))
        for i in range(unique.shape[0]):
            j = i + 1
            a = pocket_IDs.count(j)

            if a <= max_spheres:
                output_file = self.pdb_dir / pdb.name.split('_')[0] + f'.pocket{j}.pdb'
                with open(output_file, 'w') as outfile:
                    for line in pocket_IDs:
                        if int(line[22:26].strip()) == unique[i]:
                            outfile.write(line)
    
    def gen_surfs(self,
                  pocket: str) -> None:
        """
        Runs SURF to generate a surface file of given pocket.
        Arguments:
            pocket (str): PDB ID and Pocket Number delimited by `.`
        """
        pocket = self.pdb_dir / f'aligned.{pocket}'
        n, p = pocket.split('.')[:2]

        arg = (surf, 
               '-surfProbeGen', 
               'pocket', 
               self.pocket_dir / f'{n}_{p}.SURF', 
               '3', 
               '.5')
        out = open(self.pocket_dir / 'surf.log', 'w')
        subprocess.run(arg, stdout=out)
        out.close()

    def intersect(self):
        """
        Runs intersect VASP on given pocket.
        """
        count = 0
        name = '_'.join(pocket.split('.')[:2])
        structure = self.pocket_dir / f'{name}.SURF'

        (self.vasp_dir / name).mkdir(exist_ok=True)

        for init in catalogue:
            count += 1
            if full:
                modulo = 25
            else:
                modulo = 5

            if count % 500 == 0:
                print(f'-----{snum + 1}/{total} pockets to be sampled-----')

            if count % modulo == 0:
                print(f'{pocket}: {count}/{samples} conformations')

            filename = init.name.split('.')[:5]
            surf_file = self.vasp_dir / name / f'intersect.{name}.{".".join(filename)}.SURF'
            arg = (vasp, '-csg', structure, init, 'I', surf_file, '.5')
            out = open(self.out_dir  / 'intersect.log', 'w') # NOTE: CHECK THIS AGAINST OG
            subprocess.run(arg, stdout=out)
            out.close()

        arg = (surf, '-surveyVolume', structure)
        out = open(self.pocket_dir / f'{name}.vol.txt', 'w')
        subprocess.run(arg, stdout=out)
        out.close()

    def extract_score(self,
                      pocket: str) -> None:
        """
        Runs SURF to get the intersect score for all intersections performed
        on supplied pocket.
        Arguments:
            pocket (str): The pocket to score
        """
        name = '_'.join(pocket.split('.')[:2])
        for intersect in (self.vasp_dir / name).glob('*.SURF'):
            p = '_'.join(interesect.name.split('.')[1:7])
            arg = (surf, '-surveyVolume', intersect)
            out = open(self.vasp_dir / name / f'{p}_iscore.txt', 'w')
            subprocess.run(arg, stdout=out)
            out.close()

    def original_volume(self,
                        pocket: PathLike) -> None:
        """
        Runs SURF -surveyVolume on supplied pocket.
        Arguments:
            pocket (PathLike): Path to pocket SURF file.
        """
        v = self.pocket_dir / f'{"_".join(pocket.split(".")[:2])}.SURF'
        arg = (self.surf, '-surveyVolume', v)
        out = open(self.pocket_dir / f'{v.stem}.vol.txt', 'w')
        subprocess.run(arg, stdout=out)
        out.close()

    def screen_check(self,
                     pocket: str,
                     cut: float,
                     _tSamp: List[bool]) -> Tuple[bool, List[bool]]:
        """
        Check on success of short screen. Map out which translations
        we want to consider during full sampling to reduce our computational
        burden.
        Arguments:
            pocket (str): The pocket to check short screen on
            cut (float): Volume cutoff for screening
            _tSamp (list[bool]): (6,) array of booleans to guide translation
                sampling
        """
        result = False
        name = '_'.join(pocket.split('.')[:2])

        small = self.vasp_dir / name
        values = np.array(
            [
                float([lines.split() for lines in open(struc, 'r')][-1][-1])
                for struc in small.glob(f'{name}_conf0*_0_0_iscore.txt')
            ]
        )

        if np.where(values > cut)[0].size > 0:
            result = True

        for i in range(len(_tSamp)):
            scorefile = small / f'{name}_conf0_0_0_{i+1}_0_iscore.txt'
            t = [line.split() for line in open(scorefile).readlines()][-1][-1]
            _tSamp[i] = float(t) > cut

        return result, _tSamp

    def generate_long_sample(self,
                             short: np.ndarray,
                             result: bool, 
                             translation_sampling: List[bool]) -> np.ndarray:
        """
        Generate long sampling array. This is done by subtracting out the
        short sampling array that has already been completed, as well as
        any translational screens that we have deemed unneccessary to do.
        Arguments:
            short (np.ndarray): Array of paths to target SURF files that
                we ran the short screen against.
            result (bool): Signal to continue non-translated screening.
            translation_sampling (List[bool]): List of which translation
                samples we need to run.
        """
        if result:
            # remove short sample from the full list
            full = np.setdiff1d(self.tgt_dir.glob('*.SURF'), short)
            
            # remove translations we don't want to do
            for i, trans in enumerate(translation_sampling):
                if not trans:
                    translation = self.tgt_dir.glob('*conf*.*.*.{i+1}.*.SURF')
                    full = np.setdiff1d(full, translation)

        else: # this means non-translated screen failed
            full = np.array([])
            for i, trans in enumerate(translation_sampling):
                if trans:
                    translation = self.tgt_dir.glob('*conf*.*.*.{i+1}.*.SURF')
                    full = np.append(full, translation)

        return full

    def append_scorefile(self,
                         structure: str,
                         _filter: float,
                         volume: float) -> None:
        """
        Appends run to scorefile. Checks if structure has been written to file
        before adding it and outputs updated values if so.
        Arguments:
            structure (str): Structure to examine scores for
            _filter (float): Proportion of total volume to be
                considered a "hit".
            volume (float): Volume of the target pocket.
        """
        pdb, pock = structure.split('.')[:2]
        pocket_volume = float(
            open(
                self.pocket_dir / f'{pdb}_{pock}.vol.txt'
            ).readlines()[-1].split()[-1]
        )

        scores = [s for s in (self.vasp_dir / f'{pdb}_{pock}').glob('*_iscore.txt')]

        hits = 0
        best = 0
        curr = 0

        print(f'-----Getting {pdb} {pock} Scores-----')
        for score in scores:
            curr = float(open(score).readlines()[-1].split()[-1])
            if not best or curr > best:
                best = curr
            if curr / volume > _filter:
                hits += 1
            
        print(f'------Updating Scorefile-----')
        info = [line 
                for line in open(self.pdb_dir / 'infofiles' / f'{pdb}.info').readlines()]
        cofactor = None
        for cof in info[-1].split(';'):
            if cof.split(':')[-1].strip() == pock:
                cofactor = ':'.join(cof.split(':')[:2])

        temp = pd.DataFrame({
            'PDB': pdb,
            'Pocket': pock,
            'Target Vol.': volume,
            'Pocket Vol.': pocket_volume,
            'Int. Vol.': best,
            'Int%': float(best / volume),
            '#hits': hits,
            'Exp. Method': info[1][:-2],
            'Cofactor(s)': cofactor,
            'Protein Name': info[0][:-2],
        }, index=[0])

        self.scores = pd.concat([self.scores, temp]).reset_index(drop=True)
        self.pretty_print()

    def rosetta_prep(self,
                     _filter: float,
                     hits: float,
                     pocket: str) -> None:
        """
        Generates ROSETTA .pos files for each structure that passes both the
        int% and hit filters. The fpocket output for each passing model is read
        and the resID is acquired for each pocket-lining residues to be written
        to the corresponding .pos file.
        Arguments:
            _filter (float): Percent intersection filter.
            hits (float): Hit filter
            pocket (str): PDB ID and Pocket number of each structure
        """
        good_pockets = self.scores[
            (self.scores['Int%'] > _filter) & (self.scores['#hits'] > hits)
        ]

        for pdb, pnum in zip(*good_pockets[['PDB', 'Pocket']].tolist()):
            pocket_file = self.pdb_dir / f'{pdb}_out' / 'pockets' / f'pocket{pnum}_atm.pdb'

            resids = np.array(
                [
                    line[22:26].strip() 
                    for line in open(pocket_file).readlines() 
                    if line[:4] == 'ATOM'
                ], 
                dtype=int
            )

            with open(self.rosetta_dir / f'{pdb}_pock{pnum}.pos', 'w') as outfile:
                outfile.write(' '.join([str(x) for x in sorted(resids)]))

    def update_checkpoint(self,
                          structure: str) -> None:
        """
        Updates running checkpoint file with the path to a structure that
        we are done computing intersections on.
        Arguments:
            structure (str): The model we have completed
        """
        f = open(self.out_dir / 'checkpoint.chk', 'a')
        f.write(structure + '\n')
        f.close()

    def delete_surfs(self,
                     structure: str) -> None:
        """
        Deletes used SURF files to keep data footprint low. Since these are
        binary files there is no way to further compress them.
        Arguments:
            structure (str): The particular structure whose SURF files are
                to be removed.
        """
        print(f'-----Cleaning up {structure} SURF/VASP files-----')
        name = '_'.join(structure.split('.')[:2])
        shutil.rmtree(self.vasp_dir / name)
        os.remove(self.pocket_dir / f'{name}.SURF')
        os.remove(self.pocket_dir / f'{name}.vol.txt')

    def pretty_print(self):
        """
        Writes out score dataframe with nice formatting.
        """
        self.scores.to_csv(self.score_file, sep='\t')
