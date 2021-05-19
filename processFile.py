import numpy as np

def get3DFaceModelAsArray(path):
    coordinates = np.empty((468, 3), dtype=np.float64)
    idx = 0
    with open(path) as f:
        while idx < 468:
            line = f.readline().replace('v', '').replace('\n', '')
            parsed = line.split(' ')[1:]
            row = np.asarray(parsed).astype(np.float64)
            coordinates[idx] = row
            idx += 1
    return coordinates