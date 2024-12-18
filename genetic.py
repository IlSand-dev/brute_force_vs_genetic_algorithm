import itertools
from copy import deepcopy
import random as rnd
from datetime import timedelta

import numpy as np
from numba import jit

from additional_functions import segment_schedule
from config import min_time


@jit(nopython=True)
def get_penalty_type1(segments):
    if sum(segments) == 0:
        return -10000000000000
    segments = segment_schedule(segments)
    segment_type = 0
    current_time = 0
    work_time = 0
    current_day = 0
    penalty = 0
    for segment in segments:
        if current_time + segment * min_time >= 24 * 60:
            current_day += 1
        current_time = (current_time + segment * min_time) % (24 * 60)
        if current_day >= 5:
            if segment_type != 0:
                penalty += segment * min_time
        else:
            if segment_type % 4 == 0:
                if not (6 * 60 <= current_time < 10 * 60):
                    if 6 * 60 > current_time:
                        penalty += 6 * 60 - current_time
                    else:
                        penalty += current_time - 10 * 60
                work_time = current_time
            elif segment_type % 4 == 1:
                if not (13 * 60 <= current_time < 15 * 60):
                    if 13 * 60 > current_time:
                        penalty += 13 * 60 - current_time
                    else:
                        penalty += current_time - 15 * 60
                elif (segment * min_time) % 90 != 0:
                    penalty += (segment * min_time) % 90
            elif segment_type % 4 == 2:
                if segment != 60 // min_time:
                    penalty += abs(segment - 60 // min_time)
            elif segment_type % 4 == 3:
                if current_time - work_time > 9 * 60:
                    penalty += abs(current_time - work_time - 9 * 60)
                elif (segment * min_time) % 90 != 0:
                    penalty += (segment * min_time) % 90
        segment_type = (segment_type + 1) % 4
    return -penalty * 1000000000


@jit(nopython=True)
def get_penalty_type2(segments):
    if sum(segments) == 0:
        return -10000000000000
    segments = segment_schedule(segments)
    segment_type = 0
    current_time = 0
    work_time = 0
    big_rest = False
    penalty = 0
    end_work = 0
    for segment in segments:
        current_time += segment * min_time
        if current_time == 7 * 24 * 60:
            segment_type = (segment_type + 1) % 2
            continue
        if segment_type % 2 == 0:
            if work_time == 0 or end_work > 0:
                if end_work > 0 and current_time - end_work < 2 * 24 * 60:
                    penalty += abs(2 * 24 * 60 - (current_time - work_time))
                end_work = 0
                work_time = current_time
            else:
                if segment * min_time != 10:
                    penalty += abs(segment * min_time - 10)
        elif segment_type % 2 == 1:
            if current_time - work_time >= 12 * 60 - 90:
                end_work = current_time
            if current_time - work_time <= 12 * 60:
                if not(2 * 60 <= segment * min_time <= 4 * 60):
                    if 2 * 60 > segment * min_time and not big_rest:
                        penalty += abs(2 * 60 - segment * min_time)
                    else:
                        penalty += abs(segment * min_time - 4 * 60)
                if segment * min_time % 90 != 0:
                    penalty += (segment * min_time) % 90
            else:
                penalty += abs(current_time - work_time - 12 * 60)
        segment_type = (segment_type + 1) % 2
    return -penalty * 1000000000


def generate_type1_drivers(amount):
    drivers = []
    for _ in range(amount):
        schedule = []

        for day in range(5):
            start_time = rnd.randint(6 * 60 // min_time, 10 * 60 // min_time)
            shifts = rnd.randint(2, 5)
            end_time = (5 - shifts) * 90 // min_time
            worked_time = start_time + 60 // min_time + 90 * 5 // min_time

            schedule += ([0] * start_time +
                         [1] * (90 * shifts // min_time) +
                         [0] * (60 // min_time) +
                         [1] * end_time +
                         [0] * (24 * 60 // min_time - worked_time))
        drivers.append(np.array(schedule + [0] * (2 * 24 * 60 // min_time)))
    return np.array(drivers)


def generate_type2_drivers(amount):
    drivers = []
    for _ in range(amount):
        schedule = np.zeros(7 * 24 * 60 // min_time, dtype=np.int64)
        start_time = rnd.randint(0, (3 * 24 * 60 - 12 * 60) // min_time)
        end_time = start_time + (4 * 160) // min_time
        schedule[start_time:end_time] = 1
        while start_time < (7 * 24 * 60) // min_time:
            start_time = min(end_time + (2 * 24 * 60 // min_time),  (7 * 24 * 60) // min_time)
            end_time = min(start_time + (4*160) // min_time,  (7 * 24 * 60) // min_time)
            schedule[start_time:end_time] = 1
        count_ones = 0
        for i in range(len(schedule)):
            if schedule[i] == 1:
                if count_ones == 2 * 90 // min_time:
                    schedule[i:min(i + 10//min_time, len(schedule))] = 0
                    count_ones = 0
                else:
                    count_ones += 1
            else:
                count_ones = 0
        drivers.append(schedule)
    return np.array(drivers)


def goal(schedule):
    num_minutes_per_week = 7 * 24 * 60 // min_time
    week = np.zeros(num_minutes_per_week, dtype=np.int64)
    penalty = 0
    for driver in schedule[0]:
        penalty += get_penalty_type1(driver)
        week |= driver
    for driver in schedule[1]:
        penalty += get_penalty_type2(driver)
        week |= driver
    return np.sum(week) + penalty


def selection(arrays):
    strongest = []
    for i in range(0, len(arrays)):
        i1, i2, i3 = rnd.sample(range(0, len(arrays)), 3)
        strongest.append(max(arrays[i1], arrays[i2], arrays[i3], key=lambda k: goal(k)))
    return strongest


def crossover(parents, birth_rate):
    children = []
    for p1, p2 in zip(parents[::2], parents[1::2]):
        p1, p2 = copy(p1), copy(p2)
        for i in range(len(p1[0])):
            bus_left = list(range(len(p1[0])))
            j = bus_left.pop(rnd.randint(0, len(p1[0]) - 1))
            if rnd.random() < birth_rate:
                d = rnd.randint(1, len(p1[0][i]) - 1)
                p1[0][i], p2[0][j] = np.concatenate((p1[0][i][:d], p2[0][j][d:])), np.concatenate(
                    (p2[0][j][:d], p1[0][i][d:]))
        for i in range(len(p1[1])):
            bus_left = list(range(len(p1[1])))
            j = bus_left.pop(rnd.randint(0, len(p1[1]) - 1))
            if rnd.random() < birth_rate:
                d = rnd.randint(1, len(p1[1][i]) - 1)
                p1[1][i], p2[1][j] = np.concatenate((p1[1][i][:d], p2[1][j][d:])), np.concatenate(
                    (p2[1][j][:d], p1[1][i][d:]))
        children.append(p1)
        children.append(p2)
    return children


def mutate(arrays, e):
    for schedule in range(len(arrays)):
        for driver_type in range(len(arrays[schedule])):
            for driver in range(len(arrays[schedule][driver_type])):
                for segment in range(len(arrays[schedule][driver_type][driver])):
                    if rnd.random() < e:
                        arrays[schedule][driver_type][driver][segment] = int(not arrays[schedule][driver_type][driver][segment])


def copy(a):
    return deepcopy(a)


def best(arrays):
    return max([goal(i) for i in arrays])


def genetic(type1, type2, attempts_amount=10000, schedules_amount=1000, birth_rate=0.9, mutation_rate=1/(10**8)):
    type1_drivers = [generate_type1_drivers(type1) for _ in range(schedules_amount)]
    type2_drivers = [generate_type2_drivers(type2) for _ in range(schedules_amount)]
    arrays = list(zip(type1_drivers, type2_drivers))
    history = []
    for _ in range(attempts_amount):
        parents = selection(arrays)
        parents = list(map(copy, parents))
        arrays = crossover(parents, birth_rate)
        mutate(arrays, mutation_rate)
        history.append(best(arrays))
        print(history[-1])
    return list(itertools.chain(*max(arrays, key=lambda x: goal(x))))
