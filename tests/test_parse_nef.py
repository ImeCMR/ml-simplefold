#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

"""
Unit tests for NEF (NMR Exchange Format) Parser
"""

import unittest
import tempfile
from pathlib import Path
from textwrap import dedent

from simplefold.nef.parser import NEFParser, ResidueAtom, DistanceRestraint


class TestResidueAtom(unittest.TestCase):
    """Test ResidueAtom dataclass."""
    
    def test_residue_atom_creation(self):
        """Test creating a ResidueAtom."""
        atom = ResidueAtom(chain_code='A', sequence_code=42, atom_name='CA')
        self.assertEqual(atom.chain_code, 'A')
        self.assertEqual(atom.sequence_code, 42)
        self.assertEqual(atom.atom_name, 'CA')
    
    def test_residue_atom_repr(self):
        """Test ResidueAtom string representation."""
        atom = ResidueAtom(chain_code='B', sequence_code=10, atom_name='HG')
        self.assertEqual(repr(atom), 'B10HG')


class TestDistanceRestraint(unittest.TestCase):
    """Test DistanceRestraint dataclass."""
    
    def test_distance_restraint_creation(self):
        """Test creating a DistanceRestraint."""
        atom1 = ResidueAtom('A', 1, 'HA')
        atom2 = ResidueAtom('A', 2, 'H')
        restraint = DistanceRestraint(
            atom1=atom1,
            atom2=atom2,
            lower_bound=1.8,
            upper_bound=5.0,
            weight=1.0
        )
        self.assertEqual(restraint.lower_bound, 1.8)
        self.assertEqual(restraint.upper_bound, 5.0)
        self.assertEqual(restraint.weight, 1.0)
    
    def test_distance_restraint_repr(self):
        """Test DistanceRestraint string representation."""
        atom1 = ResidueAtom('A', 1, 'HA')
        atom2 = ResidueAtom('A', 2, 'H')
        restraint = DistanceRestraint(atom1, atom2, 1.8, 5.0)
        expected = '[A1HA - A2H: 1.80-5.00 Å]'
        self.assertEqual(repr(restraint), expected)


class TestNEFParser(unittest.TestCase):
    """Test NEFParser functionality."""
    
    def setUp(self):
        """Set up temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_nef_file(self, content: str, filename: str = "test.nef") -> Path:
        """Helper to create a temporary NEF file."""
        filepath = self.temp_path / filename
        with open(filepath, 'w') as f:
            f.write(dedent(content))
        return filepath
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        with self.assertRaises(FileNotFoundError):
            NEFParser(Path("nonexistent_file.nef"))
    
    def test_simple_distance_restraint(self):
        """Test parsing a simple distance restraint."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           _nef_distance_restraint_list.sf_framecode    nef_distance_restraint_list_test
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.restraint_id
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.residue_name_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.residue_name_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.weight
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  55  A  1  MET  HA   A  1  MET  HG2  1  1.8  5.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 1)
        
        r = restraints[0]
        self.assertEqual(r.atom1.chain_code, 'A')
        self.assertEqual(r.atom1.sequence_code, 1)
        self.assertEqual(r.atom1.atom_name, 'HA')
        self.assertEqual(r.atom2.chain_code, 'A')
        self.assertEqual(r.atom2.sequence_code, 1)
        self.assertEqual(r.atom2.atom_name, 'HG2')
        self.assertEqual(r.lower_bound, 1.8)
        self.assertEqual(r.upper_bound, 5.0)
        self.assertEqual(r.weight, 1.0)
        self.assertEqual(r.restraint_id, 55)
    
    def test_multiple_restraints(self):
        """Test parsing multiple distance restraints."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.restraint_id
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  55  A  1  HA   A  2  H    1.8  3.5
              2  56  A  2  HA   A  3  H    1.8  2.9
              3  57  A  3  HA   A  4  H    1.8  5.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 3)
        
        # Check first restraint
        self.assertEqual(restraints[0].upper_bound, 3.5)
        # Check second restraint
        self.assertEqual(restraints[1].upper_bound, 2.9)
        # Check third restraint
        self.assertEqual(restraints[2].upper_bound, 5.0)
    
    def test_missing_values_use_defaults(self):
        """Test that missing values (dots) use default values."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              _nef_distance_restraint.weight
              
              1  A  1  HA  A  2  H  .  .  .
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 1)
        
        r = restraints[0]
        self.assertEqual(r.lower_bound, 1.8)  # Default
        self.assertEqual(r.upper_bound, 6.0)  # Default
        self.assertEqual(r.weight, 1.0)       # Default
    
    def test_ambiguous_restraints(self):
        """Test parsing ambiguous restraints with same restraint_id."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.restraint_id
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  69  A  2  HBx  A  2  HDx  1.8  7.0
              2  69  A  2  HBy  A  2  HDx  1.8  7.0
              3  70  A  3  HA   A  4  H    1.8  5.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 3)
        
        ambiguous = parser.get_ambiguous_restraints()
        self.assertEqual(len(ambiguous), 1)  # One ambiguous group
        self.assertIn(69, ambiguous)
        self.assertEqual(len(ambiguous[69]), 2)  # Two alternatives
    
    def test_cyana_format(self):
        """Test parsing CYANA-style NEF format with only upper_limit."""
        nef_content = """
        data_test_project
        
        save_CYANA_distance_restraints_1
           _nef_distance_restraint_list.sf_category       nef_distance_restraint_list
           _nef_distance_restraint_list.sf_framecode      CYANA_distance_restraints_1
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.restraint_id
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.residue_name_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.residue_name_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.weight
              _nef_distance_restraint.upper_limit
              
              1  1  A  3  TYR  H   A  3  TYR  HA   1.0  5.01
              2  2  A  3  TYR  H   A  3  TYR  HBy  1.0  4.86
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 2)
        
        # Check that lower bounds default to 1.8
        self.assertEqual(restraints[0].lower_bound, 1.8)
        self.assertEqual(restraints[0].upper_bound, 5.01)
        self.assertEqual(restraints[1].upper_bound, 4.86)
    
    def test_quoted_values(self):
        """Test parsing quoted values in data lines."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  'A'  1  "HA"  'A'  2  'H'  1.8  5.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 1)
        
        r = restraints[0]
        self.assertEqual(r.atom1.chain_code, 'A')
        self.assertEqual(r.atom1.atom_name, 'HA')
    
    def test_multiple_saveframes(self):
        """Test parsing multiple distance restraint saveframes."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_list1
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  A  1  HA  A  2  H  1.8  5.0
           
           stop_
        
        save_
        
        save_nef_distance_restraint_list_list2
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  B  10  CA  B  11  CA  1.8  6.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 2)
    
    def test_empty_file(self):
        """Test parsing an empty NEF file."""
        nef_content = """
        data_test_project
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 0)
    
    def test_no_distance_restraints(self):
        """Test parsing NEF file without distance restraints."""
        nef_content = """
        data_test_project
        
        save_nef_molecular_system
           _nef_molecular_system.sf_category   nef_molecular_system
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        restraints = parser.get_restraints()
        self.assertEqual(len(restraints), 0)
    
    def test_summary(self):
        """Test summary generation."""
        nef_content = """
        data_test_project
        
        save_nef_distance_restraint_list_test
           _nef_distance_restraint_list.sf_category     nef_distance_restraint_list
           
           loop_
              _nef_distance_restraint.index
              _nef_distance_restraint.chain_code_1
              _nef_distance_restraint.sequence_code_1
              _nef_distance_restraint.atom_name_1
              _nef_distance_restraint.chain_code_2
              _nef_distance_restraint.sequence_code_2
              _nef_distance_restraint.atom_name_2
              _nef_distance_restraint.lower_limit
              _nef_distance_restraint.upper_limit
              
              1  A  1  HA  A  2  H  1.8  3.5
              2  A  2  HA  A  3  H  2.0  6.0
           
           stop_
        
        save_
        """
        
        filepath = self._create_nef_file(nef_content)
        parser = NEFParser(filepath)
        
        summary = parser.summary()
        self.assertIn("Number of restraints: 2", summary)
        self.assertIn("Upper bound range: 3.50 - 6.00", summary)
        self.assertIn("Lower bound range: 1.80 - 2.00", summary)
    
    def test_real_nef_file(self):
        """Test parsing a real NEF file (1e9t.nef)."""
        real_nef_path = Path("tests/data/1e9t.nef")
        
        if not real_nef_path.exists():
            self.skipTest(f"Real NEF file not found at: {real_nef_path}")
        
        print(f"\n{'='*70}")
        print(f"Testing real NEF file: {real_nef_path.name}")
        print('='*70)
        
        parser = NEFParser(real_nef_path)
        restraints = parser.get_restraints()
        
        # Basic sanity checks
        self.assertIsInstance(restraints, list)
        self.assertGreater(len(restraints), 0, "Expected to find restraints in 1e9t.nef")
        
        if restraints:
            self.assertIsInstance(restraints[0], DistanceRestraint)
            
            print(f"\n{parser.summary()}")
            
            print(f"\nFirst 10 restraints:")
            for i, r in enumerate(restraints[:10], 1):
                print(f"  {i:3d}. {r}")
            
            # Check for ambiguous restraints
            ambiguous = parser.get_ambiguous_restraints()
            if ambiguous:
                print(f"\nFound {len(ambiguous)} ambiguous restraint groups")
                first_group_id = list(ambiguous.keys())[0]
                first_group = ambiguous[first_group_id]
                print(f"\nExample ambiguous group (ID {first_group_id}, {len(first_group)} alternatives):")
                for r in first_group[:3]:  # Show first 3
                    print(f"  - {r}")
            
            # Statistics
            chains = set(r.atom1.chain_code for r in restraints) | set(r.atom2.chain_code for r in restraints)
            residues = set(r.atom1.sequence_code for r in restraints) | set(r.atom2.sequence_code for r in restraints)
            
            print(f"\nChains: {', '.join(sorted(chains))}")
            print(f"Residue range: {min(residues)} - {max(residues)}")
            print(f"Total restraints: {len(restraints)}")
            print(f"Ambiguous groups: {len(ambiguous)}")


class TestSplitDataLine(unittest.TestCase):
    """Test the _split_data_line helper method."""
    
    def setUp(self):
        """Create a temporary NEF file for parser initialization."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create a minimal valid NEF file
        filepath = self.temp_path / "test.nef"
        with open(filepath, 'w') as f:
            f.write("data_test\n")
        
        self.parser = NEFParser(filepath)
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_split_simple_values(self):
        """Test splitting simple space-separated values."""
        line = "1 A 42 CA B 43 CB 1.8 5.0"
        result = self.parser._split_data_line(line)
        expected = ['1', 'A', '42', 'CA', 'B', '43', 'CB', '1.8', '5.0']
        self.assertEqual(result, expected)
    
    def test_split_quoted_values(self):
        """Test splitting quoted values."""
        line = "'A' 42 \"CA\" B 43 'CB'"
        result = self.parser._split_data_line(line)
        expected = ['A', '42', 'CA', 'B', '43', 'CB']
        self.assertEqual(result, expected)
    
    def test_split_mixed_values(self):
        """Test splitting mixed quoted and unquoted values."""
        line = "A 42 'C A' B 43 \"C B\""
        result = self.parser._split_data_line(line)
        expected = ['A', '42', 'C A', 'B', '43', 'C B']
        self.assertEqual(result, expected)
    
    def test_split_dots(self):
        """Test handling dot (null) values."""
        line = "A . CA . . CB"
        result = self.parser._split_data_line(line)
        expected = ['A', '.', 'CA', '.', '.', 'CB']
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)