"""
Helper script to prepare files for Overleaf upload
Creates a ZIP file with all necessary files for compiling the LaTeX report
"""

import os
import zipfile
from pathlib import Path

def prepare_overleaf_zip():
    """Create ZIP file for Overleaf upload"""
    
    print("="*80)
    print("PREPARING FILES FOR OVERLEAF")
    print("="*80)
    
    # Files to include
    files_to_include = [
        'assignment3_report.tex',
        'architecture_diagram.pdf',
    ]
    
    # Check which files exist
    existing_files = []
    missing_files = []
    
    for file in files_to_include:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✓ Found: {file}")
        else:
            missing_files.append(file)
            print(f"✗ Missing: {file}")
    
    # Check for plots
    plots_dir = Path('plots')
    plot_files = []
    
    if plots_dir.exists():
        plot_files = list(plots_dir.glob('*.pdf'))
        if plot_files:
            print(f"\n✓ Found {len(plot_files)} plot files:")
            for plot in plot_files:
                print(f"  - {plot}")
                existing_files.append(str(plot))
        else:
            print("\n⚠ Warning: plots/ folder exists but no PDF files found")
            print("  Run: python evaluate_results.py --plot")
    else:
        print("\n⚠ Warning: plots/ folder doesn't exist yet")
        print("  Run: python evaluate_results.py --plot")
    
    # Create ZIP file
    if existing_files:
        zip_filename = 'assignment3_overleaf.zip'
        
        print(f"\nCreating {zip_filename}...")
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in existing_files:
                zipf.write(file, arcname=os.path.basename(file))
                print(f"  Added: {file}")
        
        print(f"\n✅ SUCCESS! Created: {zip_filename}")
        print(f"   Size: {os.path.getsize(zip_filename) / 1024:.1f} KB")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Go to: https://www.overleaf.com/")
        print("2. Sign up / Log in")
        print("3. Click 'New Project' → 'Upload Project'")
        print(f"4. Upload: {zip_filename}")
        print("5. Click 'Recompile' button")
        print("6. Edit your names (lines 13-18)")
        print("7. Download PDF")
        print("\nDONE! 🎉")
        print("="*80)
        
    else:
        print("\n❌ ERROR: No files found to include!")
        print("\nPlease make sure:")
        print("  1. You're in the Assignment 3 folder")
        print("  2. assignment3_report.tex exists")
        print("  3. architecture_diagram.pdf exists")
        print("  4. Run 'python evaluate_results.py --plot' first")
    
    if missing_files:
        print(f"\n⚠ Warning: {len(missing_files)} files are missing:")
        for file in missing_files:
            print(f"  - {file}")

if __name__ == "__main__":
    prepare_overleaf_zip()

