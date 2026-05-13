import random
import sys

# Usage: python add_weights.py input.txt output.txt [seed]
# Reads edges in "from to" format, writes "from to weight"
# Comment lines (starting with #) are preserved as-is

input_file  = sys.argv[1] if len(sys.argv) > 1 else "soc-LiveJournal1.txt"
output_file = sys.argv[2] if len(sys.argv) > 2 else "soc-LiveJournal1-weighted.txt"
seed        = int(sys.argv[3]) if len(sys.argv) > 3 else 42

rng = random.Random(seed)

print(f"Reading {input_file}...")
with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        if line.startswith("#") or line.strip() == "":
            fout.write(line)
            continue
        parts = line.split()
        if len(parts) < 2:
            fout.write(line)
            continue
        from_node = parts[0]
        to_node   = parts[1]
        weight    = rng.randint(1, 100)
        fout.write(f"{from_node}\t{to_node}\t{weight}\n")

print(f"Written to {output_file}")
