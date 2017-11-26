import re
from collections import defaultdict
import csv

path = '../results/rotation_test.txt'

data = defaultdict(lambda: {})
nextData = [["Epoch"]]
with open(path) as f:
    content = f.readlines()
    rotation = None
    for line in content:
        if line.startswith('data argument prarams'):
            match = re.search("'rotation_range': (\d+)", line)
            rotation = match.group(1)
        elif line.startswith('epoch'):
            match = re.search(r'epoch\s+(\d+)\s+test.*accuracy:\s+(\d+\.\d+)', line)
            if match:
                epoch = int(match.group(1))
                accuracy = float(match.group(2))
                data[int(rotation)][epoch] = accuracy
                if not rotation in nextData[0]:
                    nextData[0].append(rotation)
                while len(nextData) <= epoch:
                    nextData.append([str(len(nextData))])
                nextData[epoch].append(accuracy)

with open('rotation.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    for row in nextData:
        spamwriter.writerow(row)
