with open('requirements_spec.txt', 'r') as f:
    lines = f.readlines()

with open('requirements_fixed.txt', 'w') as f:
    for line in lines:
        if '=' in line and not line.startswith('#'):
            # Extract only the package name and version
            parts = line.split('=')
            if len(parts) > 2:
                f.write(f"{parts[0]}=={parts[1]}\n")
            else:
                f.write(f"{parts[0]}\n")
        else:
            f.write(line)