import re
import numpy as np


def fen_to_bit_vector(fen: str):
    parts = re.split(" ", fen)
    piece_placement = re.split("/", parts[0])
    active_color = parts[1]
    castling_rights = parts[2]
    en_passant = parts[3]
    halfmove_clock = int(parts[4])
    fullmove_clock = int(parts[5])

    bit_vector = np.zeros((13, 8, 8), dtype=np.uint8)

    # piece to layer structure taken from reference [1]
    piece_to_layer = {
        'R': 1,
        'N': 2,
        'B': 3,
        'Q': 4,
        'K': 5,
        'P': 6,
        'p': 7,
        'k': 8,
        'q': 9,
        'b': 10,
        'n': 11,
        'r': 12
    }

    castling = {
        'K': (7, 7),
        'Q': (7, 0),
        'k': (0, 7),
        'q': (0, 0),
    }

    for r, row in enumerate(piece_placement):
        c = 0
        for piece in row:
            if piece in piece_to_layer:
                bit_vector[piece_to_layer[piece], r, c] = 1
                c += 1
            else:
                c += int(piece)

    if en_passant != '-':
        bit_vector[0, ord(en_passant[0]) - ord('a'), int(en_passant[1]) - 1] = 1

    if castling_rights != '-':
        for char in castling_rights:
            bit_vector[0, castling[char][0], castling[char][1]] = 1

    if active_color == 'w':
        bit_vector[0, 7, 4] = 1
    else:
        bit_vector[0, 0, 4] = 1

    if halfmove_clock > 0:
        c = 7
        while halfmove_clock > 0:
            bit_vector[0, 3, c] = halfmove_clock % 2
            halfmove_clock = halfmove_clock // 2
            c -= 1
            if c < 0:
                break

    if fullmove_clock > 0:
        c = 7
        while fullmove_clock > 0:
            bit_vector[0, 4, c] = fullmove_clock % 2
            fullmove_clock = fullmove_clock // 2
            c -= 1
            if c < 0:
                break

    return bit_vector
