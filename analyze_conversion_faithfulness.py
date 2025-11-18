#!/usr/bin/env python3
"""
Analyze the faithfulness of conversion from monolithic to modular structure.

This script compares the original convert_hf_to_gguf.py with the refactored
modular version split across conversion/ subdirectory.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import difflib


class ClassInfo:
    """Store information about a class definition."""

    def __init__(self, name: str, decorators: List[str], source: str, file_path: str):
        self.name = name
        self.decorators = decorators
        self.source = source
        self.file_path = file_path
        self.registrations = self._extract_registrations(decorators)

    def _extract_registrations(self, decorators: List[str]) -> Set[str]:
        """Extract @ModelBase.register(...) registrations."""
        registrations = set()
        for dec in decorators:
            if 'ModelBase.register' in dec or '@register' in dec:
                # Extract the registration arguments
                registrations.add(dec.strip())
        return registrations

    def __repr__(self):
        return f"ClassInfo({self.name}, decorators={len(self.decorators)}, file={self.file_path})"


class ConversionAnalyzer:
    """Analyze the faithfulness of the conversion."""

    def __init__(self, reference_file: Path, conversion_dir: Path, current_file: Path):
        self.reference_file = reference_file
        self.conversion_dir = conversion_dir
        self.current_file = current_file

        self.reference_classes: Dict[str, ClassInfo] = {}
        self.converted_classes: Dict[str, ClassInfo] = {}

    def extract_class_info(self, file_path: Path) -> Dict[str, ClassInfo]:
        """Extract all class definitions from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return {}

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return {}

        classes = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract decorators
                decorators = []
                for dec in node.decorator_list:
                    decorators.append(ast.unparse(dec))

                # Extract class source code
                class_source = ast.get_source_segment(source, node)
                if class_source is None:
                    # Fallback: try to extract by line numbers
                    lines = source.split('\n')
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        class_source = '\n'.join(lines[node.lineno - 1:node.end_lineno])
                    else:
                        class_source = ""

                classes[node.name] = ClassInfo(
                    name=node.name,
                    decorators=decorators,
                    source=class_source,
                    file_path=str(file_path)
                )

        return classes

    def analyze_reference(self):
        """Extract all classes from the reference file."""
        print(f"Analyzing reference file: {self.reference_file}")
        self.reference_classes = self.extract_class_info(self.reference_file)
        print(f"Found {len(self.reference_classes)} classes in reference file")

    def analyze_conversion(self):
        """Extract all classes from conversion directory files."""
        print(f"\nAnalyzing conversion directory: {self.conversion_dir}")

        # Get all Python files in conversion directory
        py_files = list(self.conversion_dir.glob("*.py"))
        print(f"Found {len(py_files)} Python files in conversion directory")

        for py_file in py_files:
            if py_file.name.startswith('_'):
                continue

            classes = self.extract_class_info(py_file)
            for name, info in classes.items():
                if name in self.converted_classes:
                    print(f"Warning: Duplicate class {name} found in {py_file} and {self.converted_classes[name].file_path}")
                self.converted_classes[name] = info

        print(f"Found {len(self.converted_classes)} classes in conversion directory")

    def unified_diff_ignore_whitespace(self, lines1, lines2, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n'):
        """
        Generate unified diff while ignoring whitespace-only changes.

        Returns a generator that produces lines of unified diff format,
        but skips hunks that only contain whitespace differences.
        """
        def normalize_line(line):
            return line.strip()

        # Create normalized versions for comparison
        norm_lines1 = [normalize_line(line) for line in lines1]
        norm_lines2 = [normalize_line(line) for line in lines2]

        # Generate unified diff with original lines
        all_diff_lines = list(difflib.unified_diff(
            lines1, lines2,
            fromfile=fromfile, tofile=tofile,
            fromfiledate=fromfiledate, tofiledate=tofiledate,
            n=n, lineterm=lineterm
        ))

        if not all_diff_lines:
            return  # No differences at all

        # Filter out hunks that only contain whitespace changes
        filtered_diff = []
        hunk = []
        in_hunk = False

        for line in all_diff_lines:
            if line.startswith('@@'):
                # Start of new hunk
                if hunk:
                    # Check if previous hunk had meaningful changes
                    if self.hunk_has_meaningful_changes(hunk, norm_lines1, norm_lines2):
                        filtered_diff.extend(hunk)
                hunk = [line]
                in_hunk = True
            elif line.startswith('---') or line.startswith('+++'):
                filtered_diff.append(line)
            elif line.startswith(' '):
                # Context line
                if in_hunk:
                    hunk.append(line)
            elif line.startswith('-') or line.startswith('+'):
                # Changed line
                if in_hunk:
                    hunk.append(line)
            else:
                # End of diff or other line
                if hunk:
                    if self.hunk_has_meaningful_changes(hunk, norm_lines1, norm_lines2):
                        filtered_diff.extend(hunk)
                    hunk = []
                in_hunk = False
                filtered_diff.append(line)

        # Handle last hunk
        if hunk and self.hunk_has_meaningful_changes(hunk, norm_lines1, norm_lines2):
            filtered_diff.extend(hunk)

        if len(filtered_diff) == 2 and filtered_diff[0].strip().startswith("---") and filtered_diff[1].strip().startswith("+++"):
            return iter([])

        return iter(filtered_diff)

    def hunk_has_meaningful_changes(self, hunk_lines, norm_lines1, norm_lines2):
        """
        Check if a hunk contains changes beyond just whitespace.
        """
        # Extract line numbers from @@ line
        if not hunk_lines or not hunk_lines[0].startswith('@@'):
            return True

        minus_lines = []
        plus_lines = []

        for line in hunk_lines[1:]:  # Skip @@ line
            if line.startswith('-'):
                minus_lines.append(line[1:])
            elif line.startswith('+'):
                plus_lines.append(line[1:])
            elif line.startswith(' '):
                # Context lines - not relevant for change detection
                pass

        # Normalize the changed lines
        norm_minus = [line.strip() for line in minus_lines]
        norm_plus = [line.strip() for line in plus_lines]

        # Remove empty strings (lines that were only whitespace)
        norm_minus = [line for line in norm_minus if line]
        norm_plus = [line for line in norm_plus if line]

        # If both are empty, it was only whitespace changes
        if not norm_minus and not norm_plus:
            return False

        # Check if the meaningful content is the same
        return norm_minus != norm_plus

    def normalize_source(self, source: str, remove_empty_lines: bool = False) -> str:
        """Normalize source code for comparison (remove extra whitespace)."""
        # Remove leading/trailing whitespace from each line
        lines = [line.rstrip() for line in source.split('\n')]
        # Remove empty lines at start and end
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        if remove_empty_lines:
            lines = [x for x in lines if len(x.strip()) > 0]
        return '\n'.join(lines)

    def compare_classes(self) -> Dict[str, any]:
        """Compare classes between reference and converted versions."""
        results = {
            'missing': [],
            'extra': [],
            'registration_mismatch': [],
            'code_mismatch': [],
            'perfect_match': [],
        }

        # Find missing classes
        for class_name in self.reference_classes:
            if class_name not in self.converted_classes:
                # Check if it's a base class that should be in base.py
                if class_name in ['ModelBase', 'TextModel', 'MmprojModel']:
                    # These are expected to be in base.py
                    continue
                results['missing'].append(class_name)

        # Find extra classes (shouldn't happen, but check anyway)
        for class_name in self.converted_classes:
            if class_name not in self.reference_classes:
                results['extra'].append(class_name)

        # Compare matching classes
        for class_name in self.reference_classes:
            if class_name not in self.converted_classes:
                continue

            ref_class = self.reference_classes[class_name]
            conv_class = self.converted_classes[class_name]

            # Compare registrations
            if ref_class.registrations != conv_class.registrations:
                results['registration_mismatch'].append({
                    'class': class_name,
                    'reference': ref_class.registrations,
                    'converted': conv_class.registrations,
                    'ref_file': ref_class.file_path,
                    'conv_file': conv_class.file_path,
                })

            # Compare source code (normalized)
            ref_source_norm = self.normalize_source(ref_class.source, True)
            conv_source_norm = self.normalize_source(conv_class.source, True)

            if ref_source_norm != conv_source_norm:
                results['code_mismatch'].append({
                    'class': class_name,
                    'ref_file': ref_class.file_path,
                    'conv_file': conv_class.file_path,
                })
            elif ref_class.registrations == conv_class.registrations:
                # Perfect match
                results['perfect_match'].append(class_name)

        return results

    def print_report(self, results: Dict, show_perfect_matches: bool = False):
        """Print a detailed report of the comparison."""
        print("\n" + "=" * 80)
        print("CONVERSION FAITHFULNESS REPORT")
        print("=" * 80)

        total_ref_classes = len(self.reference_classes)
        total_conv_classes = len(self.converted_classes)

        print("\nSummary:")
        print(f"  Reference classes: {total_ref_classes}")
        print(f"  Converted classes: {total_conv_classes}")
        print(f"  Perfect matches: {len(results['perfect_match'])}")
        print(f"  Missing classes: {len(results['missing'])}")
        print(f"  Extra classes: {len(results['extra'])}")
        print(f"  Registration mismatches: {len(results['registration_mismatch'])}")
        print(f"  Code mismatches: {len(results['code_mismatch'])}")

        if results['missing']:
            print(f"\n{'=' * 80}")
            print("MISSING CLASSES (not found in conversion/):")
            print('=' * 80)
            for class_name in sorted(results['missing']):
                ref_class = self.reference_classes[class_name]
                print(f"\n  {class_name}")
                print(f"    Decorators: {ref_class.decorators}")
                print(f"    Registrations: {ref_class.registrations}")

        if results['extra']:
            print(f"\n{'=' * 80}")
            print("EXTRA CLASSES (in conversion/ but not in reference):")
            print('=' * 80)
            for class_name in sorted(results['extra']):
                conv_class = self.converted_classes[class_name]
                print(f"\n  {class_name}")
                print(f"    File: {conv_class.file_path}")
                print(f"    Decorators: {conv_class.decorators}")

        if results['registration_mismatch']:
            print(f"\n{'=' * 80}")
            print("REGISTRATION MISMATCHES:")
            print('=' * 80)
            for mismatch in results['registration_mismatch']:
                print(f"\n  {mismatch['class']}")
                print(f"    Reference: {mismatch['reference']}")
                print(f"    Converted: {mismatch['converted']}")
                print(f"    Ref file: {mismatch['ref_file']}")
                print(f"    Conv file: {mismatch['conv_file']}")

        if results['code_mismatch']:
            print(f"\n{'=' * 80}")
            print("CODE MISMATCHES:")
            print('=' * 80)
            for mismatch in results['code_mismatch']:
                class_name = mismatch['class']
                print(f"\n  {class_name}")
                print(f"    Ref file: {mismatch['ref_file']}")
                print(f"    Conv file: {mismatch['conv_file']}")

                # Show a diff
                ref_class = self.reference_classes[class_name]
                conv_class = self.converted_classes[class_name]

                ref_lines = self.normalize_source(ref_class.source).split('\n')
                conv_lines = self.normalize_source(conv_class.source).split('\n')

                diff = list(self.unified_diff_ignore_whitespace(
                    ref_lines,
                    conv_lines,
                    fromfile=f'reference/{class_name}',
                    tofile=f'conversion/{class_name}',
                    lineterm=''
                ))

                if len(diff) > 0:
                    print("\n    Diff (first 50 lines):")
                    for line in diff[:50]:
                        print(f"      {line}")
                    if len(diff) > 50:
                        print(f"      ... ({len(diff) - 50} more lines)")

        if results['perfect_match'] and show_perfect_matches:
            print(f"\n{'=' * 80}")
            print(f"PERFECT MATCHES ({len(results['perfect_match'])} classes):")
            print('=' * 80)
            # Group by file
            by_file = defaultdict(list)
            for class_name in sorted(results['perfect_match']):
                conv_class = self.converted_classes[class_name]
                file_name = Path(conv_class.file_path).name
                by_file[file_name].append(class_name)

            for file_name in sorted(by_file.keys()):
                print(f"\n  {file_name}:")
                for class_name in by_file[file_name]:
                    print(f"    - {class_name}")

        print(f"\n{'=' * 80}")
        print("CONVERSION STATUS:")
        print('=' * 80)

        if not results['missing'] and not results['registration_mismatch'] and not results['code_mismatch']:
            print("\n  ✓ CONVERSION IS FAITHFUL")
            print("  All classes have been successfully migrated with matching registrations and code.")
        else:
            print("\n  ✗ CONVERSION HAS ISSUES")
            if results['missing']:
                print(f"    - {len(results['missing'])} classes are missing")
            if results['registration_mismatch']:
                print(f"    - {len(results['registration_mismatch'])} classes have registration mismatches")
            if results['code_mismatch']:
                print(f"    - {len(results['code_mismatch'])} classes have code differences")

        print()

    def run(self):
        """Run the full analysis."""
        self.analyze_reference()
        self.analyze_conversion()
        results = self.compare_classes()
        self.print_report(results)
        return results


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    reference_file = script_dir / "reference" / "convert_hf_to_gguf.py"
    conversion_dir = script_dir / "conversion"
    current_file = script_dir / "convert_hf_to_gguf.py"

    if not reference_file.exists():
        print(f"Error: Reference file not found: {reference_file}")
        sys.exit(1)

    if not conversion_dir.exists():
        print(f"Error: Conversion directory not found: {conversion_dir}")
        sys.exit(1)

    analyzer = ConversionAnalyzer(reference_file, conversion_dir, current_file)
    results = analyzer.run()

    # Return exit code based on results
    if results['missing'] or results['registration_mismatch'] or results['code_mismatch']:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
