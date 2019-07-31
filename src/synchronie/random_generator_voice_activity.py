import random as rnd
from read_features import read_times
from constante import *
import numpy as np


def generate_random_speak(range_speak, min_size_speak=10, max_size_speak=10000, min_size_pause=10,
                          max_size_pause=30000):
    """
    Genere une piste totalment aléatoirer
    :param range_speak: debut et la fin de la génération en seconde
    :param min_size_speak: taille minimum d'une utterance
    :param max_size_speak: taille maximum d'une utterance
    :param min_size_pause:  taille minimum d'une pause
    :param max_size_pause:  taille maximum d'une pause
    :return: contient une liste de toutes les utterance sous le format (début, fin)
    """
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
    """
    Mélange une piste existante toute les x temps
    :param is_turn: la piste existante
    :param rag: range de la piste
    :param number_of_cell: nombre de cellule cote à cote à mélanger
    :return: une nouvelle piste sous format d'une liste de 0 et 1
    """
    splits = [is_turn[i: i + number_of_cell] for i in range(0, len(is_turn) + number_of_cell, number_of_cell)]
    rnd.shuffle(splits)
    shuffle_turn = [i for sub in splits for i in sub]
    return read_times({"isTurn": shuffle_turn,
                       "frameTime": np.arange(0, (len(is_turn) + 1) * frame_time, frame_time)}, rag), shuffle_turn
