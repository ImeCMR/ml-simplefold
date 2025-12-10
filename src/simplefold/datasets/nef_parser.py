#
# Copyright 2025 by Imesh Ranaweera, Alberto Perez
# All rights reserved
#

import starfile
import numpy as np

class NEFParser:
    def __init__(self, nef_file):
        self.nef_file = nef_file
        self.data = starfile.read(nef_file)

    def get_distance_restraints(self):
        """
        Parses the NEF file to extract NOESY distance restraints.
        """
        restraints = []
        # starfile.read returns a dictionary where keys are the saveframe names (e.g., 'CYANA_distance_restraints_1')
        for saveframe_name, saveframe_content in self.data.items():
            # Check if the saveframe contains distance restraint metadata and is of origin 'noe'
            meta_key = '_nef_distance_restraint_list'
            if meta_key in saveframe_content:
                metadata = saveframe_content[meta_key]
                # Ensure metadata is a Series/dict-like object and check the origin
                if hasattr(metadata, 'get') and metadata.get('restraint_origin') == 'noe':
                    
                    # The actual restraint data is in a DataFrame with this key
                    restraint_data_key = '_nef_distance_restraint'
                    if restraint_data_key in saveframe_content:
                        df = saveframe_content[restraint_data_key]
                        
                        # Convert DataFrame to a list of dictionaries
                        for _, row in df.iterrows():
                            try:
                                # Skip restraints with '.' for values that should be numbers
                                if row['sequence_code_1'] == '.' or row['sequence_code_2'] == '.' or row['upper_limit'] == '.':
                                    continue
                                restraint = {
                                    'id': row['restraint_id'],
                                    'atom1_residue_number': int(row['sequence_code_1']),
                                    'atom1_atom_name': row['atom_name_1'],
                                    'atom2_residue_number': int(row['sequence_code_2']),
                                    'atom2_atom_name': row['atom_name_2'],
                                    'upper_bound': float(row['upper_limit']),
                                }
                                restraints.append(restraint)
                            except (ValueError, TypeError, KeyError):
                                # Handle cases where conversion fails or a column is missing
                                continue
        return restraints

if __name__ == '__main__':
    # This is an example of how to use the NEFParser
    # You would need a sample NEF file to run this.
    # parser = NEFParser('sample.nef')
    # restraints = parser.get_distance_restraints()
    # print(restraints)
    pass
