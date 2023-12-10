import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import torch
from lightning import Trainer

from src.model.center_n_normals_net import LightingCenterNNormalsNet
from src.dataset.shrec2023 import SymmetryDataModule


def save_prediction(batch, y_pred, path):
    idxs, points, y_true, transforms = batch
    y_pred = y_pred[0, :, :].clone()
    idxs, points, y_true = transforms[0].inverse_transform(idxs.clone(), points.clone(), y_true[0].clone())
    _, _, y_pred[:, 0:6] = transforms[0].inverse_transform(idxs.clone(), points.clone(), y_pred[:, 0:6].clone())

    idx = idxs[0]
    n_heads = len(y_pred)

    with open(os.path.join(path, f"points{idx}_res.txt"), "w") as f:
        f.write(str(n_heads))
        f.write("\n")
        np.savetxt(f, y_pred.cpu().detach().numpy())


def save_predictions(prediction_list, path):
    if not os.path.exists(path):
        os.mkdir(path)
    for (batch, y_pred) in prediction_list:
        save_prediction(batch, y_pred, path)


parser = ArgumentParser()
parser.add_argument("--data_path", type=pathlib.Path, required=True)
parser.add_argument("--output_path", type=pathlib.Path, required=True)
parser.add_argument("--model_path", type=pathlib.Path, required=True)
parser.add_argument("--n_workers", type=int, required=False, default=4)

if __name__ == "__main__":
    args = vars(parser.parse_args())
    DATA_PATH = args["data_path"]
    OUTPUT_PATH = args["output_path"]
    MODEL_PATH =args["model_path"]
    N_WORKERS = args["n_workers"]

    model = LightingCenterNNormalsNet.load_from_checkpoint(MODEL_PATH)
    data_module = SymmetryDataModule(
        test_data_path=DATA_PATH,
        predict_data_path=DATA_PATH,
        does_predict_has_ground_truths=True,
        batch_size=1,
        n_workers=N_WORKERS
    )

    trainer = Trainer()
    predictions = trainer.predict(model, data_module)
    save_predictions(predictions, OUTPUT_PATH)
