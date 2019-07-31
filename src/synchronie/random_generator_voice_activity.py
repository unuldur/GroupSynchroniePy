import random as rnd
from read_features import read_times
from constante import *
import numpy as np


def generate_random_speak(range_speak, min_size_speak=10, max_size_speak=10000, min_size_pause=10,
                          max_size_pause=30000):
    end = range_speak[0] * 100
    speak_r = []
    while end < range_speak[1] * 100:
        end += rnd.randint(min_size_pause, max_size_pause)
        size = rnd.randint(min_size_speak, max_size_speak)
        speak_r.append((end / 100, (end + size) / 100))
        end = end + size
    print('end')
    return speak_r


def generate_random_from_existante(is_turn, rag, number_of_cell=100):
    splits = [is_turn[i: i + number_of_cell] for i in range(0, len(is_turn) + number_of_cell, number_of_cell)]
    rnd.shuffle(splits)
    shuffle_turn = [i for sub in splits for i in sub]
    return read_times({"isTurn": shuffle_turn,
                       "frameTime": np.arange(0, (len(is_turn) + 1) * frame_time, frame_time)}, rag), shuffle_turn
