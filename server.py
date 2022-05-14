from keras.models import model_from_json
from flask import Flask, request
from BoardStateLearner import MODEL_SAVE_FILE_WITHOUT_TYPE
from BoardStateLearner import BoardStateLearner as BSL

app = Flask(__name__)


# Not trying to really follow a scalable pattern, just doing easy stuff to make this play project work.
# This is the entry point for the REST service that uses this app.
def best_move_call(side, fen):
    json_file = open(MODEL_SAVE_FILE_WITHOUT_TYPE+".json", "r")
    model_json = json_file.read()
    json_file.close()

    trained_model = model_from_json(model_json)
    trained_model.load_weights(MODEL_SAVE_FILE_WITHOUT_TYPE+".h5")

    neural_net = BSL("EMPTY")
    neural_net.overwrite_model(trained_model)
    return str(neural_net.best_next_move(fen, side))


# Expected Query Params:
#   side=True/False -> white/black
#   fen=VALID_BOARD_FEN
# Return: Move String (like: 'e2e4')
@app.route('/best_move')
def find_best_move():
    if len(request.args) != 2:
        return "Error: Invalid query parameters.", 404

    side = eval(request.args.get("side"))
    fen = request.args.get("fen")

    best_next_move = best_move_call(side, fen)

    return {
        "best_move": best_next_move
    }
