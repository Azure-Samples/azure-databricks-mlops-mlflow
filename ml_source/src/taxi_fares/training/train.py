from typing import Dict

import lightgbm as lgb
import numpy as np


def train(
    train_data: Dict[str, np.ndarray], params: dict, num_rounds: int
) -> lgb.Booster:
    train_lgb_dataset = lgb.Dataset(train_data["X"], label=train_data["y"].values)

    # Train a lightGBM model
    model = lgb.train(params, train_lgb_dataset, num_rounds)
    return model
