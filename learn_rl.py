import numpy as np
import torch

from diffusion import Diffusion
from utils import *

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'resnet50', '학습에 사용할 모델명')
flags.DEFINE_float('learning_rate', 0.001, '학습률')
flags.DEFINE_integer('batch_size', 32, '배치 사이즈')
flags.DEFINE_boolean('debug', False, '디버그 모드 여부')

def main(argv):
    # 설정값 사용
    print(f"모델: {FLAGS.model}")
    print(f"학습률: {FLAGS.learning_rate}")
    
    reward_diffusion = Diffusion(state_dim, action_dim, hidden_dim, hidden_layers)
    


if __name__ == '__main__':
    app.run(main)