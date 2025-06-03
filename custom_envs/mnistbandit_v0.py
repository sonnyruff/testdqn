# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MNIST classification as a bandit.

In this environment, we test the agent's generalization ability, and abstract
away exploration/planning/memory etc -- i.e. a bandit, with no 'state'.
"""

from typing import Optional, Any, TypeVar, SupportsFloat
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# from bsuite.environments import base # https://github.com/google-deepmind/bsuite/blob/main/bsuite/environments/base.py
# from bsuite.experiments.mnist import sweep
# from bsuite.utils import datasets # https://github.com/google-deepmind/bsuite/blob/main/bsuite/utils/datasets.py

# import dm_env # https://github.com/google-deepmind/dm_env/blob/master/dm_env/_environment.py
# from dm_env import specs # https://github.com/google-deepmind/dm_env/blob/master/dm_env/specs.py
import numpy as np
import gymnasium as gym

import array
import gzip
import struct
import os
from os import path
from six.moves.urllib.request import urlretrieve


class MNISTBanditEnv(gym.Env):
    """MNIST classification as a bandit environment."""
    
    def __init__(self, dims: int = 1,
                 arms: int = 10,
                 dynamic_rate: int | None = None,
                 seed: int | None = None,
                 noisy: bool = False):
        """Loads the MNIST training set (60K images & labels) as numpy arrays.

        Args:
            seed: Optional integer. Seed for numpy's random number generator (RNG).
        """
        super().__init__()
        (images, labels) = load_mnist()

        num_data = len(labels)

        self._num_data = int(num_data)
        self._image_shape = images.shape[1:]

        self._images = images[:self._num_data]
        self._labels = labels[:self._num_data]
        self._rng = np.random.RandomState(seed)
        self._correct_label = None

        self._total_regret = 0.
        self._optimal_return = 1.
        
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.prod(self._image_shape),), dtype=np.float32)

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Agent gets an MNIST image to 'classify' using its next action."""
        idx = self._rng.randint(self._num_data)
        image = self._images[idx].astype(np.float32) / 255
        self._correct_label = self._labels[idx]
        # return image, {}
        return image.flatten(), {}

    def step(self, action: int) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """+1/-1 for correct/incorrect guesses. This also terminates the episode."""
        correct = action == self._correct_label
        reward = 1. if correct else -1.
        regret = self._optimal_return - reward
        self._total_regret += regret
        observation = np.zeros(shape=np.prod(self._image_shape), dtype=np.float32)
        return observation, reward, False, False, {'regret': regret, 'total_regret': self._total_regret}


def _download(url, filename, directory="/tmp/mnist"):
    """Download a url to a file in the given directory."""
    if not path.exists(directory):
        os.makedirs(directory)
    out_file = path.join(directory, filename)
    if not path.isfile(out_file):
        urlretrieve(url, out_file)

def load_mnist(directory="/tmp/mnist"): 
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()),
                            dtype=np.int8).reshape((num_data, rows, cols))

    for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
        _download(base_url + filename, filename, directory)

    train_images = parse_images(
        path.join(directory, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(
        path.join(directory, "train-labels-idx1-ubyte.gz"))
    
    return (train_images, train_labels)