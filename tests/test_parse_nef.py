import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from src.simplefold.datasets.nef_parser import NEFParser


class TestNEFParserWith1e9t(unittest.TestCase):
    
    # Hardcoded path to real NEF file
    NEF_FILE = '/orange/alberto.perezant/imesh.ranaweera/softwares/ml-simplefold/tests/data/1e9t.nef'
    
    def test_file_exists(self):
        """Test that the 1e9t.nef file exists"""
        self.assertTrue(os.path.exists(self.NEF_FILE), 
                       f"NEF file not found at {self.NEF_FILE}")
        print(f"✓ Found NEF file: {self.NEF_FILE}")
        print(f"  File size: {os.path.getsize(self.NEF_FILE)} bytes")
    
    def test_parse_1e9t_nef(self):
        """Test parsing the real 1e9t.nef file"""
        
        print(f"\n{'='*60}")
        print(f"Parsing Real 1e9t.nef File")
        print(f"{'='*60}")
        print(f"File: {self.NEF_FILE}")
        
        try:
            parser = NEFParser(self.NEF_FILE)
            print("✓ NEF file loaded successfully")
            
            restraints = parser.get_distance_restraints()
            print(f"✓ Distance restraints extracted")
            
            # Basic checks
            self.assertIsInstance(restraints, list)
            self.assertGreater(len(restraints), 0, "Should have at least one restraint")
            
            print(f"\n{'='*60}")
            print(f"Statistics")
            print(f"{'='*60}")
            print(f"Total restraints: {len(restraints)}")
            
            # Calculate statistics
            total_assignments = sum(len(r['assignments']) for r in restraints)
            multi_count = sum(1 for r in restraints if len(r['assignments']) > 1)
            single_count = len(restraints) - multi_count
            
            print(f"Single-assignment restraints: {single_count}")
            print(f"Multi-assignment restraints: {multi_count}")
            print(f"Total assignments: {total_assignments}")
            print(f"Average assignments per restraint: {total_assignments / len(restraints):.2f}")
            
            # Test structure of first restraint
            restraint = restraints[0]
            self.assertIn('id', restraint)
            self.assertIn('assignments', restraint)
            self.assertIn('upper_bound', restraint)
            
            print(f"\n{'='*60}")
            print(f"First Restraint Example")
            print(f"{'='*60}")
            print(f"ID: {restraint['id']}")
            print(f"Upper bound: {restraint['upper_bound']} Å")
            print(f"Number of assignments: {len(restraint['assignments'])}")
            
            for i, assign in enumerate(restraint['assignments'][:3], 1):
                print(f"  Assignment {i}:")
                print(f"    Res {assign['atom1_residue_number']}:{assign['atom1_atom_name']} - "
                      f"Res {assign['atom2_residue_number']}:{assign['atom2_atom_name']}")
            
            if len(restraint['assignments']) > 3:
                print(f"    ... and {len(restraint['assignments']) - 3} more assignments")
            
            # Test data types
            self.assertIsInstance(restraint['upper_bound'], float)
            self.assertGreater(restraint['upper_bound'], 0)
            
            # Test a few more restraints
            print(f"\n{'='*60}")
            print(f"Sample Restraints (first 5)")
            print(f"{'='*60}")
            
            for i, r in enumerate(restraints[:5], 1):
                print(f"{i}. ID={r['id']}, upper_bound={r['upper_bound']:.2f} Å, "
                      f"assignments={len(r['assignments'])}")
            
            # Validate all restraints have correct structure
            for restraint in restraints:
                self.assertIsInstance(restraint['upper_bound'], float)
                self.assertGreater(restraint['upper_bound'], 0)
                self.assertGreater(len(restraint['assignments']), 0)
                
                for assignment in restraint['assignments']:
                    self.assertIsInstance(assignment['atom1_residue_number'], int)
                    self.assertIsInstance(assignment['atom2_residue_number'], int)
                    self.assertIsInstance(assignment['atom1_atom_name'], str)
                    self.assertIsInstance(assignment['atom2_atom_name'], str)
            
            print(f"\n{'='*60}")
            print(f"✓ All tests passed!")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR: Failed to parse 1e9t.nef")
            print(f"{'='*60}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            print(f"\nThis is likely due to starfile library version 0.5.13")
            print(f"being incompatible with the NEF file format.")
            print(f"\nSuggested solutions:")
            print(f"1. Try: pip install --upgrade starfile")
            print(f"2. Try: pip install 'starfile>=0.4.0,<0.5.0'")
            print(f"3. Or use gemmi library instead: pip install gemmi")
            raise


if __name__ == '__main__':
    unittest.main(verbosity=2)