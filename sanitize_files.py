
import glob
import os

def sanitize_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = []
    for char in content:
        if ord(char) > 127:
            new_content.append(f"\\u{ord(char):04x}")
        else:
            new_content.append(char)
    
    sanitized = "".join(new_content)
    
    if content != sanitized:
        print(f"Sanitizing {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(sanitized)
    else:
        print(f"No changes needed for {filepath}")

def main():
    files = glob.glob("step*.py")
    for file in files:
        sanitize_file(file)

if __name__ == "__main__":
    main()
