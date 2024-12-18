from numba import jit
import numpy as np

@jit(nopython=True)
def segment_schedule(schedule):
    change_indices = np.where(schedule[:-1] != schedule[1:])[0]+1
    segment_lengths = np.diff(np.concatenate((np.array([0]), change_indices, np.array([len(schedule)]))))
    return segment_lengths
