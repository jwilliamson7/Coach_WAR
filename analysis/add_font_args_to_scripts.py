"""
Automated script to add --font argument support to all matplotlib-based analysis scripts.
Adds import of figure_utils and font configuration to each script.
"""

import os
import re

def add_font_support_to_script(filepath):
    """Add font argument support to a matplotlib script."""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already has figure_utils import
    if 'from figure_utils import' in content or 'import figure_utils' in content:
        print(f"  ✓ {os.path.basename(filepath)} - already has font support")
        return False

    # Skip if doesn't use matplotlib
    if 'matplotlib' not in content and 'plt.' not in content:
        print(f"  - {os.path.basename(filepath)} - doesn't use matplotlib")
        return False

    modified = False
    lines = content.split('\n')
    new_lines = []

    # Track state
    found_imports = False
    found_main = False
    added_import = False
    added_argparse = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Add import after other imports
        if not added_import and ('import matplotlib' in line or 'import pandas' in line):
            found_imports = True

        if found_imports and not added_import and (line.strip() == '' or (not line.startswith('import') and not line.startswith('from'))):
            if not added_import:
                new_lines.append('from figure_utils import configure_matplotlib_fonts')
                added_import = True
                modified = True

        # Add argparse import if not present
        if not added_argparse and 'import argparse' not in content:
            if 'import pandas' in line and 'argparse' not in content:
                new_lines.append(line)
                new_lines.append('import argparse')
                added_argparse = True
                modified = True
                i += 1
                continue

        # Add font config at start of main function
        if 'def main(' in line and 'configure_matplotlib_fonts' not in content:
            new_lines.append(line)
            # Add argparse in main
            indent = '    '
            new_lines.append(f"{indent}parser = argparse.ArgumentParser()")
            new_lines.append(f"{indent}parser.add_argument('--font', default='serif', help='Font family')")
            new_lines.append(f"{indent}args = parser.parse_args()")
            new_lines.append(f"{indent}configure_matplotlib_fonts(args.font)")
            new_lines.append('')
            modified = True
            i += 1
            continue

        new_lines.append(line)
        i += 1

    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        print(f"  ✓ {os.path.basename(filepath)} - updated with font support")
        return True
    else:
        print(f"  - {os.path.basename(filepath)} - no changes needed")
        return False

def main():
    """Add font support to all relevant scripts."""
    print("="*80)
    print("Adding --font argument support to analysis scripts")
    print("="*80)
    print()

    analysis_dir = 'analysis'
    scripts = [f for f in os.listdir(analysis_dir) if f.endswith('.py') and f not in ['__init__.py', 'figure_utils.py', 'add_font_args_to_scripts.py']]

    modified_count = 0
    for script in sorted(scripts):
        filepath = os.path.join(analysis_dir, script)
        if add_font_support_to_script(filepath):
            modified_count += 1

    print()
    print("="*80)
    print(f"Modified {modified_count} scripts")
    print("="*80)

if __name__ == "__main__":
    main()
