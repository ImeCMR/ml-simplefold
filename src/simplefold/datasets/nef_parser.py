#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
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
        This version is updated to match the format of the user-provided file (e.g., 1e9t.nef).
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

                        # Filter out rows with invalid data before iterating
                        df_filtered = df[
                            (df['sequence_code_1'] != '.') &
                            (df['sequence_code_2'] != '.') &
                            (df['upper_limit'] != '.')
                        ]
                        # Convert DataFrame to a list of dictionaries
                        for _, row in df_filtered.iterrows():
                            try:
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
