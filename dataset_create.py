import pandas as pd
import re
import os
import stat

# Function to determine patch type based on annotations
def determine_patch_type(annotations):
    # Split annotations into parts (e.g., "(b:mem->re):(s:leak->fo)")
    annot_parts = [part.strip("()") for part in annotations.split("):(")]
    
    # Check primary annotation
    for part in annot_parts:
        if part.startswith("b:"):
            return "Bug"
        elif part.startswith("p:"):
            return "Performance"
        elif part.startswith("c:"):
            return "Reliability"
        elif part.startswith("misc->"):
            return "Maintenance"
        elif part.startswith("f"):
            return "Feature"
    
    # Default to Feature if no specific bug/performance/reliability/maintenance annotation
    return "Feature"

# Function to parse changelog and create dataset
def create_dataset(changelog_file):
    # Check if file is executable
    file_mode = os.stat(changelog_file).st_mode
    if stat.S_ISREG(file_mode) and (file_mode & stat.S_IXUSR):
        print(f"{changelog_file} is an executable file. Reading as text...")

    # Initialize dataset structure
    data = {
        "Version": [],
        "Description": [],
        "Component": [],
        "Files_Modified": [],
        "Lines_Added": [],
        "Lines_Removed": [],
        "Patch_Type": []
    }
    
    # Read the changelog file
    try:
        with open(changelog_file, 'r', encoding='utf-8') as f:
            changelog = f.read()
    except PermissionError:
        print(f"Error: Permission denied when reading {changelog_file}. Ensure the script has read permissions.")
        exit(1)
    except UnicodeDecodeError:
        print(f"Error: {changelog_file} contains non-UTF-8 content. Attempting binary read with fallback encoding...")
        with open(changelog_file, 'rb') as f:
            changelog = f.read().decode('utf-8', errors='replace')
    
    # Remove shebang line if present (though not in your sample, keeping for robustness)
    if changelog.startswith('#!'):
        changelog = '\n'.join(changelog.splitlines()[1:])
    
    # Split into sections by version headers
    sections = re.split(r'ChangeLog-2\.6\.\d+\s*[-]*\s*\n', changelog)
    # Remove empty or irrelevant sections (e.g., leading document header)
    sections = [s.strip() for s in sections if s.strip() and not s.strip().startswith('<DOCUMENT>')]
    
    # Regular expression to match annotation line
    annot_pattern = r"\.\s*\((\d+)\):\((.*?)\):\((.*?)\)\s*(\d+)\s*(\d+)\s*(\d+)"
    # Regular expression to match non-annotated entries (e.g., maintenance without numbers)
    simple_entry_pattern = r"^\s*([^\n]+?)\s*\n\s*\.\s*\((\d+)\):\((.*?)\)(?:\s*(\d+)\s*(\d+)\s*(\d+))?"
    
    current_version = None
    for section in sections:
        # Extract version from the section header (e.g., "ChangeLog-2.6.1" -> "2.6.1")
        version_match = re.search(r'ChangeLog-2\.6\.(\d+)', changelog.split(section)[0])
        if version_match:
            current_version = f"2.6.{version_match.group(1)}"
        
        # Split section into lines
        lines = section.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('----'):
                i += 1
                continue
            
            # Check if this is a patch entry
            desc = line
            if i + 1 < len(lines):
                annot_line = lines[i + 1].strip()
                
                # Try annotated entry
                match = re.match(annot_pattern, annot_line)
                if match:
                    version_num, component, annotations, files, added, removed = match.groups()
                    version = f"2.6.{version_num}"
                    patch_type = determine_patch_type(annotations)
                    
                    # Add to dataset
                    data["Version"].append(version)
                    data["Description"].append(desc)
                    data["Component"].append(component)
                    data["Files_Modified"].append(int(files))
                    data["Lines_Added"].append(int(added))
                    data["Lines_Removed"].append(int(removed))
                    data["Patch_Type"].append(patch_type)
                    i += 2  # Skip annotation line
                    continue
                
                # Try simple entry (e.g., maintenance or doc entries without numbers)
                simple_match = re.match(simple_entry_pattern, line + '\n' + annot_line)
                if simple_match:
                    desc, version_num, component, files, added, removed = simple_match.groups()
                    version = f"2.6.{version_num}"
                    patch_type = "Maintenance"  # Default for misc-> entries without annotations
                    files = int(files) if files else 0
                    added = int(added) if added else 0
                    removed = int(removed) if removed else 0
                    
                    # Add to dataset
                    data["Version"].append(version)
                    data["Description"].append(desc)
                    data["Component"].append(component if component else "unknown")
                    data["Files_Modified"].append(files)
                    data["Lines_Added"].append(added)
                    data["Lines_Removed"].append(removed)
                    data["Patch_Type"].append(patch_type)
                    i += 2  # Skip annotation line
                    continue
            
            # Handle entries without annotations (e.g., standalone maintenance/doc)
            if not re.match(r'^\s*\.\s*\(', line):  # Not an annotation line
                data["Version"].append(current_version)
                data["Description"].append(desc)
                data["Component"].append("unknown")
                data["Files_Modified"].append(0)
                data["Lines_Added"].append(0)
                data["Lines_Removed"].append(0)
                data["Patch_Type"].append("Maintenance")  # Default for unannotated
            i += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Main execution
if __name__ == "__main__":
    # Specify the changelog file path
    changelog_file = "fs-patch/xfs-patch"
    output_name = changelog_file.split('/')[1]
    
    # Check if file exists
    if not os.path.exists(changelog_file):
        print(f"Error: {changelog_file} not found. Please save the ChangeLog document as 'changelog'.")
        exit(1)
    
    # Create dataset
    dataset = create_dataset(changelog_file)
    
    # Save to CSV
    output_file = f"{output_name}_type_dataset.csv"
    dataset.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    # Display basic info
    print("\nDataset Info:")
    print(dataset.info())
    print("\nPatch Type Distribution:")
    print(dataset["Patch_Type"].value_counts())