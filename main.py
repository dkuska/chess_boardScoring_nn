import net
import utils

import chess
import torch
import time

play_mode = "player_vs_ai" if False else "ai_vs_ai"

net_type = "NN" if True else "CNN"


def game_init():
    board = chess.Board()

    if net_type == "NN":
        state_dict = torch.load("chess_500k.pth", map_location=torch.device('cpu'))
        nn = net.Net()
    elif net_type == "CNN":
        state_dict = torch.load("chess_cnn_500k.pth", map_location=torch.device('cpu'))
        nn = net.Net()
    else:
        return

    nn.load_state_dict(state_dict, strict=False)

    return board, nn


# Color == Color of CPU
# Net returns positive values for favourable board for white and negative values for favourable board for black
def determine_next_move(color: str, board, nn):
    best_score = -100000 if color == "w" else 100000
    best_move = chess.Move.null()

    for move in board.legal_moves:
        if move == chess.Move.null():
            continue

        board.push(move)

        fen = board.fen()
        fen_vector = utils.fen_to_bit_vector(fen)
        fen_tensor = torch.from_numpy(fen_vector).byte()

        score = nn(fen_tensor)[0]

        if color == "w":
            if score > best_score:
                # print("{} is better than {}".format(score, best_score))
                best_score = score
                best_move = move
        elif color == "b":
            if score < best_score:
                best_score = score
                best_move = move

        # print("Move: {}, Score: {}".format(move, score))
        board.pop()
    return best_move


def input_move(board):
    while True:
        move_str = input("Your Move: ")
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
        else:
            print("Invalid move, try again:")

def ai_vs_ai():
    ai_black = net.Net()
    ai_white = net.Net()

    ai_white.load_state_dict(torch.load("chess_fc_fulldropout_500k.pth", map_location=torch.device('cpu')))
    ai_black.load_state_dict(torch.load("chess_fc_singledropout_500k.pth", map_location=torch.device('cpu')))

    board = chess.Board()

    while not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material():
        white_move = determine_next_move("w", board, ai_white)
        board.push(white_move)
        print("White_AI decided: {} is the best move".format(white_move))
        print(board)

        black_move = determine_next_move("b", board, ai_black)
        board.push(black_move)
        print("Black_AI decided: {} is the best move".format(black_move))
        print(board)

        print("-----------")
        # time.sleep(10)

    if board.is_checkmate():
        reason = "Checkmate"
    elif board.is_stalemate():
        reason = "Stalemate"
    elif board.is_insufficient_material():
        reason = "Insufficient Material"
    else:
        reason = "WTF"

    print("Game ended because: ", reason)


def game_loop():
    board, nn = game_init()

    player_color = input("Play black or white? [b]/[w]: ")
    if player_color == "w":
        ai_color = "b"
    elif player_color == "b":
        ai_color = "w"
    else:
        print("Invalid color entered. Aborting game")
        return

    while not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material():
        if ai_color == "w":
            ai_move = determine_next_move(ai_color, board, nn)
            print("AI decided: {} is the best move".format(ai_move))
            board.push(ai_move)

            print(board)

            player_move = input_move(board)
            board.push(player_move)
        else:
            player_move = input_move(board)
            board.push(player_move)

            ai_move = determine_next_move(ai_color, board, nn)
            print("AI decided: {} is the best move".format(ai_move))
            board.push(ai_move)

        print(board)
        print("---------")

    if board.is_checkmate():
        reason = "Checkmate"
    elif board.is_stalemate():
        reason = "Stalemate"
    elif board.is_insufficient_material():
        reason = "Insufficient Material"
    else:
        reason = "WTF"

    print("Game ended because: ", reason)


if __name__ == '__main__':
    if play_mode == "player_vs_ai":
        game_loop()
    else:
        ai_vs_ai()
