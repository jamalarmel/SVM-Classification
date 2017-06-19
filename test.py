import glob
import numpy as np
file_list = glob.glob('corpus/corpus/*.txt')
arrays = [np.genfromtxt(f, delimiter=',', dtype=None) for f in file_list]
final_array = np.concatenate([arrays])
print final_array