import random as rnd
import time
from datetime import timedelta

import numpy as np
from numba import jit

from additional_functions import segment_schedule
from config import min_time


@jit(nopython=True)
def check_type1(segments):
    segments = segment_schedule(segments)
    segment_type = 0
    current_time = 0
    work_time = 0
    current_day = 0
    for segment in segments:
        if current_time + segment * min_time >= 24 * 60:
            current_day += 1
        current_time = (current_time + segment * min_time) % (24 * 60)
        if current_day >= 5:
            if segment_type != 0:
                return False
        else:
            if segment_type % 4 == 0:
                if not (6 * 60 <= current_time < 10 * 60):
                    return False
                work_time = current_time
            elif segment_type % 4 == 1:
                if not (13 * 60 <= current_time < 15 * 60):
                    if 13 * 60 > current_time:
                        return False
                    else:
                        return False
                elif (segment * min_time) % 90 != 0:
                    return False
            elif segment_type % 4 == 2:
                if segment != 60 // min_time:
                    return False
            elif segment_type % 4 == 3:
                if current_time - work_time > 9 * 60:
                    return False
                elif (segment * min_time) % 90 != 0:
                    return False
        segment_type = (segment_type + 1) % 4
    return True


@jit(nopython=True)
def check_type2(segments):
    if sum(segments) == 0:
        return False
    segments = segment_schedule(segments)
    segment_type = 0
    current_time = 0
    work_time = 0
    big_rest = False
    for segment in segments:
        current_time += segment * min_time
        if current_time == 7 * 24 * 60:
            segment_type = (segment_type + 1) % 2
            continue
        if segment_type % 2 == 0:
            if work_time == 0 or big_rest:
                if big_rest and current_time - work_time < 3 * 24 * 60:
                    return False
                work_time = current_time
                big_rest = False
            else:
                if segment * min_time != 10:
                    return False
        elif segment_type % 2 == 1:
            if current_time - work_time < 12 * 60:
                if segment % 90 != 0:
                    return False
            else:
                if current_time - work_time > 12 * 60:
                    return False
                big_rest = True
        segment_type = (segment_type + 1) % 2
    return True


def goal(schedule, type1, type2):
    num_minutes_per_week = 7 * 24 * 60 // min_time
    week = np.zeros(num_minutes_per_week, dtype=np.int64)
    for i in range(type1):
        if not check_type1(schedule[i]):
            return -1
        week |= schedule[i]
    for i in range(type1, type2):
        if not check_type2(schedule[i]):
            return -1
        week |= schedule[i]
    return np.sum(week)


def brute_force(type1, type2):
    max_score = 0
    best_schedule = 0
    size = (type1 + type2) * 7 * 24 * 60 // min_time
    for i in range(size):
        array = np.array([int(bit) for bit in f"{i:0{size}b}"])
        parts = np.array_split(array, type1 + type2)
        driver_goal = goal(parts, type1, type2)
        if driver_goal > max_score:
            max_score = driver_goal
            best_schedule = parts
    return best_schedule