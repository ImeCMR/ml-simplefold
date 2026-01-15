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
        Properly handles multi-assignment restraints (same restraint_id in multiple rows).
        """
        restraints = []
        # starfile.read returns a dictionary where keys are the saveframe names (e.g., 'CYANA_distance_restraints_1')
        for saveframe_name, saveframe_content in self.data.items():
            # Check if the saveframe contains distance restraint metadata and is of origin 'noe'
            meta_key = '_nef_distance_restraint_list'

            # Skip saveframes without distance restraint metadata
            if meta_key not in saveframe_content:
                continue

            metadata = saveframe_content[meta_key]

            # Handle both formats:
            # - Real NMR files (1e9t.nef): have 'restraint_origin' == 'noe'
            # - Simple test files (test.nef): may not have 'restraint_origin' field
            restraint_origin = None  # Initialize to None to handle missing field
            if hasattr(metadata, 'get'):
                restraint_origin = metadata.get('restraint_origin')
            
            # Accept if: has origin='noe' OR doesn't specify origin (assume NOE)
            if restraint_origin and restraint_origin != 'noe':
                continue  # Skip non-NOE restraint types

            restraint_data_key = '_nef_distance_restraint'
            if restraint_data_key not in saveframe_content:
                continue

            df = saveframe_content[restraint_data_key]

            # Filter out rows with invalid data
            df_filtered = df[
                (df['sequence_code_1'] != '.') &
                (df['sequence_code_2'] != '.') &
                (df['upper_limit'] != '.')
            ]

            # GROUP BY restraint_id to handle multi-assignment restraints
            grouped = df_filtered.groupby('restraint_id')

            for restraint_id, group in grouped:
                # Create assignments list for all atom pairs with this restraint_id
                assignments = []
                upper_bound = None

                for _, row in group.iterrows():
                    try:
                        assignment = {
                            'atom1_residue_number': int(row['sequence_code_1']),
                            'atom1_atom_name': row['atom_name_1'],
                            'atom2_residue_number': int(row['sequence_code_2']),
                            'atom2_atom_name': row['atom_name_2'],
                        }
                        assignments.append(assignment)
                        
                        # Upper bound should be same for all assignments in a group
                        # (they represent the same restraint)
                        if upper_bound is None:
                            upper_bound = float(row['upper_limit'])
                            
                    except (ValueError, TypeError, KeyError):
                        # Skip malformed rows
                        continue

                if assignments and upper_bound is not None:
                    restraint = {
                        'id': restraint_id,
                        'assignments': assignments,
                        'upper_bound': upper_bound,
                    }
                    restraints.append(restraint)

        return restraints

if __name__ == '__main__':
    # This is an example of how to use the NEFParser
    # You would need a sample NEF file to run this.
    # parser = NEFParser('sample.nef')
    # restraints = parser.get_distance_restraints()
    # print(restraints)
    pass
