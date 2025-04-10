import numpy as np
import pandas as pd
from pathlib import Path
from persim import wasserstein
from ripser import ripser
from search import pocketSearcher
from typing import List

Diagram = List[np.ndarray]
PathLike = Union[str, Path]

class pocketHomology(pocketSearcher):
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
        n_alpha_spheres (int): Number of alpha spheres in the target pocket.
    """
    def __init__(self,
                 target: PathLike,
                 pdb_directory: PathLike, 
                 out_directory: PathLike,
                 n_alpha_spheres: int,
                 aliases: PathLike):
        super().__init__(pdb_directory, out_directory, '', config=aliases)
        pocket_coords = self.scrape_coordinates(self.run_fpocket(target))
        self.target = self.generate_persistence_diagram(pocket_coords)
        self.gen_scorefile()
        self.max_alphas = n_alpha_spheres * self.max_cut

    def search(self, 
               structure: PathLike) -> None:
        """
        Main serial function. Takes in a single scaffold PDB, performs
        persistent homology protocol and writes wasserstein distances to
        internal score dataframe.
        """
        good_pockets = self.run_fpocket(structure)
        for good_pocket in good_pockets:
            coords = self.scrape_coordinates(good_pocket)
            diag = self.generate_persistence_diagram(coords)
            distances = self.compute_wasserstein(diag)
        
            self.append_scorefile(good_pocket, distances)

    def batch(self, 
              structures: List[PathLike]) -> None:
        """
        Batch out search calls. Intended to allow for parallelization
        via ray/parsl/mpi/etc so that we don't overwrite or miss any
        entries in the scores dataframe object.
        """
        for structure in structures:
            self.search(structure)

    def run_fpocket(self,
                    structure: PathLike) -> List[PathLike]:
        """
        Runs fpocket and then screens resulting pockets to ensure they are
        within our number of alpha sphere criteria. Returns a list of filepaths
        to viable pockets.
        """
        out = structure.with_suffix('_out')
        if not out.exists():
            args = (self.fpocket, '-f', structure, '-i', self.min_spheres)
            subprocess.run(args)
        
        pocket_IDs = [int(line[22:26].strip())
                      for line in open(out / structure.name).readlines()
                      if 'STP' in line]

        unique = np.unique(np.array(pocket_IDs))
        good = []
        for i in range(unique.shape[0]):
            j = i + 1
            a = pocket_IDs.count(j)

            if a <= self.max_alphas:
                output_file = self.pdb_dir / structure.name.split('_')[0] + f'.pocket{j}.pdb'
                good.append(output_file)
                with open(output_file, 'w') as outfile:
                    for line in pocket_IDs:
                        if int(line[22:26].strip()) == unique[i]:
                            outfile.write(line)

        return good

    def scrape_coordinates(self,
                           pdb: PathLike) -> np.ndarray:
        """
        Helper function to scrape out coordinates from a PDB file.
        """
        coords = []
        for line in open(pdb).readlines():
            if 'STP' in line:
                coords.append([line[30:38].strip(),  # x
                               line[38:46].strip(),  # y
                               line[46:54].strip()]) # z

        return np.array(coords, dtype=np.float32)

    def generate_persistence_diagram(self,
                                     coords: np.ndarray) -> Diagram:
        """
        Using alpha filtration protocol, generate a persistence diagram
        for a given set of coordinates. Returns only the diagrams.
        """
        return ripser(coords)['dgms']

    def compute_wasserstein(self,
                            diagram: Diagram) -> Tuple[float]:
        """
        Computes the Wasserstein distance between two persistence diagrams
        at 3 levels of resolution: H0, H1, and H2. Each corresponds to changes
        in connectivity (H0), holes (H1) or voids (H2) in the alpha filtration.
        Generally it is known that the H1/H2 features are more relevant in
        comparing different diagrams but we save all 3 out here anyways.
        """
        wass = np.zeros((3))
        for i in wass.shape[0]:
            wass[i] = wasserstein(self.target[i], diagram[i])

        return wass

    def gen_scorefile(self) -> None:
        """
        Generates a new score dataframe, or reads in an existing csv file if
        it exists.
        """
        self.score_file = self.out_dir / 'scores.csv'

        if self.score_file.exists():
            self.scores = pd.read_csv(self.score_file)
        else:
            self.scores = pd.DataFrame(columns=['PDB',
                                                'Pocket',
                                                'N Spheres (Target)',
                                                'N Spheres (Pocket)',
                                                'H0',
                                                'H1',
                                                'H2',
                                                'Exp. Method',
                                                'Cofactor(s)',
                                                'Protein Name'])

    def append_scorefile(self,
                         structure: PathLike,
                         dists: List[float]) -> None:
        """
        Adds a new set of wasserstein distances for a given pocket to
        score dataframe.
        """
        pdb, pock = structure.split('.')[:2]
        temp = pd.DataFrame({
            'PDB': pdb,
            'Pocket': pock,
            'N Spheres (Target)': ,
            'N Spheres (Pocket)': ,
            'H0': dists[0],
            'H1': dists[1],
            'H2': dists[2],
            'Exp. Method': ,
            'Cofactor(s)': ,
            'Protein Name': ,
        }, index=[0])
        
        self.scores = pd.concat([self.scores, temp]).reset_index(drop=True)
        self.pretty_print()
