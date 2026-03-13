import os
import glob
import re

files = glob.glob('src/*.cu')
for f in files:
    with open(f, 'r') as file:
        content = file.read()

    if 'auto idx' not in content and 'idx(' not in content:
        continue

    # Add include if not present
    if '#include "minitorch/utils.cuh"' not in content:
        content = '#include "minitorch/utils.cuh"\n' + content

    # Pattern 1: n_cols logic
    content = re.sub(
        r'auto\s+idx\s*=\s*\[&n_cols\]\(.*?\)\s*\{\s*return.*?\};\n?',
        '',
        content
    )
    # Pattern 2: b_cols logic inside ops.cu
    content = re.sub(
        r'auto\s+idx\s*=\s*\[&b_cols\]\(.*?\)\s*\{\s*return.*?\};\n?',
        '',
        content
    )

    # Remove commented versions
    content = re.sub(r'\s*//\s*auto\s+idx.*\n?', '\n', content)

    # Replace usages where n_cols is the context (most files)
    if f != 'src/ops.cu':
        content = re.sub(r'idx\((.*?),\s*(.*?)\)', r'get_idx_2d(\1, \2, n_cols)', content)
    else:
        # In ops.cu, there are multiple functions with different closures:
        # We need to target them specifically. Easiest is to revert ops.cu manual or
        # replace `idx(y, x)` with `get_idx_2d(y, x, n_cols)` mostly, and for ker_b_add, `get_idx_2d(y, x, b_cols)`
        # Let's do a smart replace for ops.cu
        pass

    with open(f, 'w') as file:
        file.write(content)

print("done")
