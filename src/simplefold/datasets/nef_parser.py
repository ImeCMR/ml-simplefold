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
        """
        restraints = []
        for saveframe in self.data.values():
            if '_Distance_restraint_list.Sf_category' in saveframe:
                restraint_list = saveframe['_Distance_restraint']
                for i in range(len(restraint_list['ID'])):
                    restraint = {
                        'id': restraint_list['ID'][i],
                        'atom1_residue_number': restraint_list['Comp_index_ID_1'][i],
                        'atom1_atom_name': restraint_list['Atom_ID_1'][i],
                        'atom2_residue_number': restraint_list['Comp_index_ID_2'][i],
                        'atom2_atom_name': restraint_list['Atom_ID_2'][i],
                        'upper_bound': restraint_list['Distance_upper_bound_val'][i],
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
