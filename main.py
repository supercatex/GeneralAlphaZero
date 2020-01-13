from Game import *
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(
        per_process_gpu_memory_fraction=0.2,
        allow_growth=None,
    )
)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

_env = GameEnv()
_model = AlphaZeroModel(_env.observation)
_model.load("model_data/model_best_config.json", "model_data/model_best_weight.h5")

_p1 = GameAgent(1, _model, _env)
_p2 = HumanAgent(2)
_env.add_agent(_p1)
_env.add_agent(_p2)

_env.reset()
while not _env.done:
    agent: Agent = _env.current_agent()
    action = agent.action(_env)
    _env.step(action)
    print(_env.observation)
