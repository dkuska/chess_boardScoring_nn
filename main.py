import net
import utils

import chess
import torch
import numpy as np

play_mode = "player_vs_ai" if False else "ai_vs_ai"

net_type = "NN" if True else "CNN"
model_path = "chess_fc_singledropout_500k.pth" if net_type == "NN" else ""

# This parameter is used to insert some randomness into the game
# TODO: Tune this!
epsilon = 0.5


def game_init():
    board = chess.Board()

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    nn = net.Net()
    nn.load_state_dict(state_dict, strict=False)

    return board, nn


# Color == Color of CPU
# Net returns positive values for favourable board for white and negative values for favourable board for black
def determine_next_move(color: str, board, nn):
    move_list = []
    score_list = []

    for move in board.legal_moves:
        if move == chess.Move.null():
            continue

        board.push(move)

        fen = board.fen()
        fen_vector = utils.fen_to_bit_vector(fen)
        fen_tensor = torch.from_numpy(fen_vector).byte()
        with torch.no_grad():
            score = nn(fen_tensor)[0].item()

        score_list.append(score)
        move_list.append(move)

        board.pop()

    if color == "w":
        best_score = max(score_list)
    else:
        best_score = min(score_list)

    # Good moves are all those, whose score is within range of the best score
    good_moves = [move_list[i] for i in range(len(move_list)) if abs(best_score - score_list[i]) <= epsilon]
    best_move = np.random.choice(good_moves)

    return best_move


def input_move(board):
    legal_move = False
    move = chess.Move.null()
    while not legal_move:
        move_str = input("Your Move: ")
        move = chess.Move.from_uci(move_str)
        if move not in board.legal_moves or move == chess.Move.null():
            print("Invalid move, try again:")
        else:
            return move
            legal_move = True
    return move


def ai_vs_ai():
    ai_black = net.Net()
    ai_white = net.Net()

    ai_white.load_state_dict(torch.load("chess_fc_fulldropout_500k.pth", map_location=torch.device('cpu')))
    ai_black.load_state_dict(torch.load("chess_fc_fulldropout_500k.pth", map_location=torch.device('cpu')))

    board = chess.Board()

    move_counter = 0

    while not board.is_checkmate() and not board.is_stalemate() and not board.is_insufficient_material():
        move_counter += 1
        white_move = determine_next_move("w", board, ai_white)
        board.push(white_move)
        print(f"Move {move_counter} - White_AI decided: {white_move}")
        print(board)

        move_counter += 1
        black_move = determine_next_move("b", board, ai_black)
        board.push(black_move)
        print(f"Move {move_counter} - Black_AI decided: {black_move} is the best move")
        print(board)

        outcome = board.outcome()
        if outcome is not None:
            print("Game over: ", outcome)
            print("Winner: ", outcome.winner)
            print("Reason: ", outcome.termination)
            print("Result: ", outcome.result())
            break


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
            board.push(ai_move)
            print(f"### AI decided: {ai_move} ###")
            print(board)

            player_move = input_move(board)
            board.push(player_move)
        else:
            player_move = input_move(board)
            board.push(player_move)
            print(board)

            ai_move = determine_next_move(ai_color, board, nn)
            board.push(ai_move)
            print(f"### AI decided: {ai_move} ###")

        print(board)
        print("---------")

        outcome = board.outcome()
        if outcome is not None:
            print("Game over: ", outcome)
            print("Winner: ", outcome.winner)
            print("Reason: ", outcome.termination)
            print("Result: ", outcome.result())
            break


if __name__ == '__main__':
    if play_mode == "player_vs_ai":
        game_loop()
    else:
        ai_vs_ai()
