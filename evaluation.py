import os
import shutil
from Connect4 import *
from tools.alphazero import SelfPlayWorker
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.5,
        allow_growth=None,
    )
)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


# define directories
_dir_model = "model_data"
_filename_config = "model_config.json"
_filename_weight = "model_weight.h5"
_dir_data = "play_data"
_threshold = 0.55

# Game settings
_env = GameEnv()
_model_next = AlphaZeroModel(_env.observation)
_model_best = AlphaZeroModel(_env.observation)

_p1 = GameAgent(1, _model_next, _env)
_p2 = GameAgent(2, _model_best, _env)
_env.add_agent(_p1)
_env.add_agent(_p2)

for k in range(100000000):
    print(f"Round {k + 1}:")

    # self-play, play with two best model.
    _model_best.load(
        os.path.join(_dir_model, _filename_config),
        os.path.join(_dir_model, _filename_weight)
    )
    _model_next.load(
        os.path.join(_dir_model, _filename_config),
        os.path.join(_dir_model, _filename_weight)
    )
    _worker = SelfPlayWorker(_env)
    _worker.nb_game_in_file = 100
    _worker.play_data_dir = _dir_data
    _worker.start()

    # optimization
    _optimizer = GameOptimizer(_env, _dir_data)
    _optimizer.training("model_data/model_config.json", "model_data/model_weight.h5")

    # evaluation
    while True:
        _dir_next_model = None
        for _f1 in os.listdir(_dir_model):
            _path = os.path.join(_dir_model, _f1)
            if os.path.isdir(_path):
                _dir_next_model = _path

        if _dir_next_model is None:
            break

        _model_best.load(
            os.path.join(_dir_model, _filename_config),
            os.path.join(_dir_model, _filename_weight)
        )
        _model_next.load(
            os.path.join(_dir_next_model, _filename_config),
            os.path.join(_dir_next_model, _filename_weight)
        )

        _total = 50
        _win = 0
        _draw = 0
        for i in range(_total):
            _env.reset()
            while not _env.done:
                agent: Agent = _env.current_agent()
                action = agent.action(_env)
                _env.step(action)
            if _env.winner == 2:
                _win += 1
            elif _env.winner == -1:
                _draw += 1
            print(f"Game {i + 1}: {_win}/{_total}({_draw})"
                  f"-- {_win / (_total - _draw) * 100}% -- {_env.observation.STATES[_env.winner]}")
            if _win / (_total - _draw) > _threshold:
                print("Best model changed.")
                _model_next.save(
                    os.path.join(_dir_model, _filename_config),
                    os.path.join(_dir_model, _filename_weight)
                )
                break
            elif (_win + _total - _draw - i) / (_total - _draw) <= _threshold:
                print("Keep the old one.")
                break
        shutil.rmtree(_dir_next_model)
    print(f"Finished {k + 1} rounds.")
print("END")
