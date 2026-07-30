import os
import pickle
import numpy as np
from Bio.PDB import PDBParser

def get_centroid(residue):
    # Retrieve all atoms and exclude backbone atoms N, CA, C, O
    sidechain_coords = []
    for atom in residue:
        atom_name = atom.get_name().strip()
        if atom_name not in ['N', 'CA', 'C', 'O']:
            sidechain_coords.append(atom.get_coord())
            
    resname = residue.get_resname().strip()
    # Glycine or residues with no remaining side-chain atoms fall back to the CA coordinate
    if resname == 'GLY' or len(sidechain_coords) == 0:
        if 'CA' in residue:
            return residue['CA'].get_coord()
        else:
            # Ultimate fallback: use the first available atom coordinate if CA is missing
            atoms = list(residue.get_atoms())
            if len(atoms) > 0:
                return atoms[0].get_coord()
            else:
                raise ValueError(f"Residue {residue} has no atoms!")
    else:
        return np.mean(sidechain_coords, axis=0)

def main():
    # Dataset information mapping input PKL files to their expected output PKL files
    datasets_info = [
        {"name": "Train335", "pkl_path": "./Dataset/Train_335.pkl", "out_pkl": "./Feature/psepos/Train335_psepos_SC.pkl"},
        {"name": "Test60", "pkl_path": "./Dataset/Test_60.pkl", "out_pkl": "./Feature/psepos/Test60_psepos_SC.pkl"},
        {"name": "Test315-28", "pkl_path": "./Dataset/Test_315-28.pkl", "out_pkl": "./Feature/psepos/Test315-28_psepos_SC.pkl"},
        {"name": "UBtest31-6", "pkl_path": "./Dataset/UBtest_31-6.pkl", "out_pkl": "./Feature/psepos/UBtest31-6_psepos_SC.pkl"},
    ]
    
    # Create the destination folders if they do not exist
    os.makedirs("./Feature/psepos", exist_ok=True)
    os.makedirs("./Feature/distance_map_SC", exist_ok=True)
    
    # Cache to avoid duplicate parsing of PDBs for IDs shared across datasets
    psepos_cache = {}
    
    parser = PDBParser(QUIET=True)
    
    # Collect all unique IDs and sequence info from all dataset files
    unique_id_seqs = {}
    dataset_ids = {}
    for ds in datasets_info:
        pkl_path = ds["pkl_path"]
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Dataset pkl file not found at {pkl_path}")
            
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        dataset_ids[ds["name"]] = list(data.keys())
        for ID, item in data.items():
            sequence = item[0]
            if ID in unique_id_seqs:
                if unique_id_seqs[ID] != sequence:
                    raise ValueError(f"Mismatch in sequence for ID {ID} across different datasets!")
            else:
                unique_id_seqs[ID] = sequence

    print(f"Total unique IDs to process across all datasets: {len(unique_id_seqs)}")
    
    # Process PDB files and compute side-chain centroids
    processed_count = 0
    for ID, sequence in unique_id_seqs.items():
        pdb_path = f"./PDB/{ID}.pdb"
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"PDB file for ID {ID} not found at {pdb_path}")
            
        pdb_id = ID[:-1]
        chain_id = ID[-1]
        
        structure = parser.get_structure(pdb_id, pdb_path)
        # Use model 0
        model = structure[0]
        
        if chain_id not in model:
            raise KeyError(f"Chain {chain_id} not found in model 0 of PDB structure for ID {ID}")
            
        chain = model[chain_id]
        
        # Get standard residues (excluding heteroatoms/water) in order
        residues = [r for r in chain if r.id[0] == ' ']
        
        # Handle the known sequence/structure mismatch in 2j3rA by dropping the extra Cysteine at index 55
        if ID == '2j3rA' and len(residues) == 159:
            del residues[55]
            
        # Assertion to guarantee 1:1 positional alignment
        if len(residues) != len(sequence):
            raise AssertionError(
                f"Length mismatch for ID {ID}: PDB standard residue count is {len(residues)}, "
                f"but dataset sequence length is {len(sequence)}."
            )
            
        # Compute per-residue side-chain centroid (or CA fallback)
        coords = []
        for r in residues:
            coord = get_centroid(r)
            coords.append(coord)
            
        psepos = np.vstack(coords)  # Shape (L, 3)
        psepos_cache[ID] = psepos
        
        # Save distance map matrix if missing from the Feature folder (e.g. 2j3rA)
        dist_map_path = f"./Feature/distance_map_SC/{ID}.npy"
        if not os.path.exists(dist_map_path):
            print(f"Generating missing distance map for ID: {ID}")
            # Pairwise Euclidean distance matrix (L x L)
            diff = psepos[:, np.newaxis, :] - psepos[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.save(dist_map_path, dist_matrix)
            print(f"Saved missing distance map to {dist_map_path}")
            
        processed_count += 1
        if processed_count % 50 == 0 or processed_count == len(unique_id_seqs):
            print(f"Processed {processed_count}/{len(unique_id_seqs)} proteins...")
            
    # Assemble and write the four separate pickle dicts mapping {ID: psepos_array}
    for ds in datasets_info:
        ds_name = ds["name"]
        ids = dataset_ids[ds_name]
        ds_dict = {ID: psepos_cache[ID] for ID in ids}
        
        with open(ds["out_pkl"], "wb") as f:
            pickle.dump(ds_dict, f)
        print(f"Successfully wrote {len(ds_dict)} IDs to {ds['out_pkl']}")
        
    print("\nSummary of Processed Datasets:")
    for ds in datasets_info:
        print(f"  - {ds['name']}: {len(dataset_ids[ds['name']])} IDs written to {ds['out_pkl']}")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()
