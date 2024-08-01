import os
from glob import glob
import numpy as np
from pdb import set_trace

mode = 993

path = "/gemini/data-2/car_%d" % mode
print("process ", path)

dir_paths = []

lines = []
for dir_path in glob(os.path.join(path, '*')):
    if dir_path.endswith('DS_Store'):
        continue
    dir_path = dir_path.replace(path, 'cars')
    dir_paths.append(dir_path)
    
    for root, _, files in os.walk(dir_path):
        if 'render' not in root:
            continue
        for fname in files:
            if not fname.endswith('png'):
                continue
            render_path = os.path.join(root, fname)
            gt_path = render_path.replace('cars', path).replace('render/', '')[:-3] + 'jpg'
            if not os.path.exists(gt_path):
                continue
            lines.append('%s %s\n' % (render_path, gt_path))

print(len(lines))

train_path = "txt/train%d.txt" % mode
test_path = "txt/test%d.txt" % mode
np.random.seed(20240531)
np.random.shuffle(lines)
num = int(len(lines)/10)
with open(test_path, 'w') as f:
    f.writelines(lines[:num])
with open(train_path, 'w') as f:
    f.writelines(lines[num:])
