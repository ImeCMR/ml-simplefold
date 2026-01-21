#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import pandas as pd
import re
import io

class NEFParser:
    """
    A simple parser for NMR Exchange Format (NEF) files.
    Focuses on extracting distance restraints.
    """
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.restraints = []

    def parse(self, filepath=None):
        if filepath:
            self.filepath = filepath

        if not self.filepath:
            raise ValueError("No filepath provided for NEF parsing.")

        with open(self.filepath, 'r') as f:
            content = f.read()

        # Extract save_distance_restraint_list blocks
        # This is a simplified STAR parser
        save_blocks = re.findall(r'save_distance_restraint_list_(\w+)\s+(.*?)\s+save_', content, re.DOTALL)

        all_restraints = []
        for name, block in save_blocks:
            restraints = self._parse_block(block)
            all_restraints.extend(restraints)

        self.restraints = all_restraints
        return self.restraints

    def _parse_block(self, block):
        # Look for the distance_restraint loop
        loop_match = re.search(r'loop_\s+(_distance_restraint\.\w+\s+)+(.*?)(\n\s*stop_|\Z)', block, re.DOTALL)
        if not loop_match:
            return []

        # Extract headers
        headers = re.findall(r'(_distance_restraint\.\w+)', loop_match.group(0))
        data_part = loop_match.group(2).strip()

        # Parse data lines using pandas for convenience
        # We need to handle quoted strings if they exist, but NEF usually doesn't have them in loops
        try:
            df = pd.read_csv(io.StringIO(data_part), sep=r'\s+', names=[h.split('.')[-1] for h in headers], comment='#')
        except Exception as e:
            print(f"Error parsing NEF block: {e}")
            return []

        # Filter out invalid rows (some NEF files use '.' for missing values)
        # Specifically check for atom and residue identifiers and bounds
        required_cols = ['resid_1', 'atom_id_1', 'resid_2', 'atom_id_2', 'distance_upper_bound']
        for col in required_cols:
            if col in df.columns:
                df = df[df[col] != '.']

        return df.to_dict('records')

def parse_nef_restraints(nef_path):
    parser = NEFParser(nef_path)
    return parser.parse()
