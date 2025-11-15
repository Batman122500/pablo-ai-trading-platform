# collect_code.py
import os

def collect_all_code():
    print("🚀 Collecting all Python code from project...")

    # Directories to skip
    skip_dirs = {'__pycache__', '.git', 'venv', 'env', 'node_modules', '.idea', '.vscode'}

    all_files = []

    # Walk through all directories
    for root, dirs, files in os.walk('.'):
        # Remove skipped directories from the traversal
        dirs[:] = [d for d in dirs if d not in skip_dirs]

        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                all_files.append(full_path)

    print(f"📁 Found {len(all_files)} Python files")

    # Write all code to an output file
    with open('5.txt', 'w', encoding='utf-8') as output:
        output.write("🚀 PABLO AI TRADING PLATFORM - COMPLETE CODE DUMP\n")
        output.write("=" * 80 + "\n\n")

        for file_path in sorted(all_files):
            output.write("\n" + "=" * 80 + "\n")
            output.write(f"📄 FILE: {file_path}\n")
            output.write("=" * 80 + "\n\n")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    output.write(f.read())
                output.write("\n\n")  # Spacing between files
            except Exception as e:
                output.write(f"❌ ERROR reading file: {e}\n")

    print("✅ Project code saved to: 5.txt")

if __name__ == "__main__":
    collect_all_code()
