#!/usr/bin/env python3

import collections
import math
import re


def entropy(word):
    # bits of information needed to encode the entropy
    counts = collections.Counter(word)
    p = [v / len(word) for v in counts.values()]
    return abs(sum(map(lambda x: x * math.log2(x), p)))


def metric_entropy(word):
    # randomness of the information
    if len(word):
        return entropy(word) / len(word)
    else:
        return 0


def CCR(word):
    # CharacterContinuityRate
    if len(word):
        letter_groups = re.findall(r"\D+", word)
        digit_groups = re.findall(r"\d+", word)
        symbol_groups = re.findall(r"[,./<>?\'\";:!@#$%^&*]+", word)
        letter = 0
        if len(letter_groups):
            letter = len(max(letter_groups, key=len))

        digit = 0
        if len(digit_groups):
            digit = len(max(digit_groups, key=len))
        symbol = 0
        if len(symbol_groups):
            symbol = len(max(symbol_groups, key=len))

        return (letter + digit + symbol) / len(word)
    else:
        return 0
