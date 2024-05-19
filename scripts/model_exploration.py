import pathlib
from argparse import ArgumentParser
from importlib import import_module

import torch
import yaml
from tqdm import tqdm
import lightning as L

from src.dataset.SymDataModule import SymDataModule
from src.dataset.SymDatasetItem import SHAPE_TYPE, SHAPE_TYPE_AMOUNT, PERTURBATION_TYPE, PERTURBATION_TYPE_AMOUNT
from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry, \
    get_match_sequence_discrete_rotational_symmetry, get_match_sequence_continue_rotational_symmetry
from src.model.LightingCenterNNormalsNet import LightingCenterNNormalsNet



SYM_TYPES = {"plane": 0, "d_axis": 1, "c_axis": 2}
METRICS_TAGS = {"map": 0, "phc": 1, "loss": 2}
SYM_TYPES_AMOUNT = len(SYM_TYPES.keys())
METRICS_TAGS_AMOUNT = len(METRICS_TAGS.keys())


def get_class_factory(class_path):
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        class_factory = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(class_path)
    return class_factory


def parse(args):
    if type(args) is dict:
        for k, v in args.items():
            if k == "class_path":
                class_factory = get_class_factory(v)
                if "init_args" in args.keys():
                    init_args = parse(args["init_args"])
                    del args["init_args"]
                else:
                    init_args = {}
                return class_factory(**init_args)

            args[k] = parse(v)
        return args
    elif type(args) is list:
        return [parse(arg) for arg in args]
    else:
        return args


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=pathlib.Path, required=True)
    parser.add_argument("--config_path", type=pathlib.Path, required=True)
    parser.add_argument("--output_path", type=pathlib.Path, default="valid")
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--n_workers", type=int, required=False, default=0)
    parser.add_argument("--device", type=str, required=False, default="cpu")
    parser.add_argument("--examples_used", type=int, required=False, default=-1)
    return parser


def get_dataloader(a_datamodule: SymDataModule, split: str):
    if split == "train":
        datamodule.setup("fit")
        return a_datamodule.train_dataloader()
    elif split == "val":
        datamodule.setup("fit")
        return a_datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup("test")
        return a_datamodule.test_dataloader()
    else:
        raise ValueError(f"Split should be one of ['train', 'val', 'test']. Got: {split}!")


def run_prediction(curr_model,
                   curr_batch, sym_pred, sym_true,
                   loss_fun, metrics_match_sequence_fun
                   ):
    c_hat, match_pred, match_true, _, _ = curr_model.matcher.get_optimal_assignment(curr_batch.get_points(),
                                                                                    sym_pred, sym_true)
    bundled_predictions = (curr_batch, sym_pred, c_hat, match_pred, match_true)
    loss, others = loss_fun(bundled_predictions)

    eval_predictions = [(curr_batch.get_points(), sym_pred, sym_true)]
    map, phc, pr_curve = calculate_metrics_from_predictions(eval_predictions, metrics_match_sequence_fun,
                                                            curr_model.metric_param_dict)

    result_metrics = torch.concat((map.view(1), phc.view(1), loss.view(1), others))
    result_head_usage = c_hat

    return result_metrics, result_head_usage[0]


if __name__ == "__main__":
    parser = get_parser()
    args = vars(parser.parse_args())

    CKPT_PATH = args["ckpt_path"]
    CONFIG_PATH = args["config_path"]
    OUTPUT_PATH = args["output_path"]
    SPLIT_USED = args["split"]
    N_WORKERS = args["n_workers"]
    DEVICE = args["device"]
    EXAMPLES_USED = args["examples_used"]

    # parse config yaml file and get configs for trainer, datamodule and model
    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    config['data']['n_workers'] = N_WORKERS
    config['data']['batch_size'] = 1
    config['data']['transform'] = parse(config['data']['transform'])
    config['data']['collate_function'] = get_class_factory(config['data']['collate_function'])

    L.seed_everything(config['seed_everything'])

    datamodule = SymDataModule(**config['data'])
    dataloader = get_dataloader(datamodule, SPLIT_USED)

    model = LightingCenterNNormalsNet.load_from_checkpoint(CKPT_PATH)
    model.eval().to(DEVICE)

    PLANE_LOSS_COMPONENTS_AMOUNT = 4
    D_AXIS_LOSS_COMPONENTS_AMOUNT = 5
    C_AXIS_LOSS_COMPONENTS_AMOUNT = 4
    PLANE_PREDICTED_AMOUNT = model.net.amount_plane_normals
    D_AXIS_PREDICTED_AMOUNT = model.net.amount_axis_discrete_normals
    C_AXIS_PREDICTED_AMOUNT = model.net.amount_axis_continue_normals

    counts = torch.zeros((PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT), device=DEVICE)

    metrics = [
        torch.zeros((
            METRICS_TAGS_AMOUNT + PLANE_LOSS_COMPONENTS_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT
        ), device=DEVICE),
        torch.zeros((
            METRICS_TAGS_AMOUNT + D_AXIS_LOSS_COMPONENTS_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT
        ), device=DEVICE),
        torch.zeros((
            METRICS_TAGS_AMOUNT + C_AXIS_LOSS_COMPONENTS_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT
        ), device=DEVICE),
    ]

    heads_used = [
        torch.zeros(PLANE_PREDICTED_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT,
                    device=DEVICE),
        torch.zeros(D_AXIS_PREDICTED_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT,
                    device=DEVICE),
        torch.zeros(C_AXIS_PREDICTED_AMOUNT, PERTURBATION_TYPE_AMOUNT, SHAPE_TYPE_AMOUNT,
                    device=DEVICE)
    ]

    metric_param_dict = {
        "eps": 0.01,
        "theta": 0.00015230484,
        "confidence_threshold": 0.01,
        "rot_angle_threshold": 0.0174533,
    }
    i = 0
    # We are using batch_size = 1 to be able to split the metric accumulator
    with torch.no_grad():
        for batch in tqdm(iter(dataloader)):
            batch.device = DEVICE
            model.matcher.device = DEVICE

            item = batch.get_item(0)
            shape_type = SHAPE_TYPE[item.shape_type]
            perturbation_type = PERTURBATION_TYPE[item.perturbation_type]
            ###################
            # Forward pass
            ###################
            points = torch.stack(batch.get_points())
            points = torch.transpose(points, 1, 2).float()
            plane_predictions, axis_discrete_predictions, axis_continue_predictions = model.net.forward(points)

            ###################
            # Plane
            ###################
            if PLANE_PREDICTED_AMOUNT > 0:
                curr_metrics, curr_heads = run_prediction(
                    model, batch, plane_predictions, batch.get_plane_syms(),
                    model.plane_loss, get_match_sequence_plane_symmetry
                )

                heads_used[0][:, perturbation_type, shape_type] += curr_heads
                metrics[0][:, perturbation_type, shape_type] += curr_metrics

            ###################
            # d_axis
            ###################
            if D_AXIS_PREDICTED_AMOUNT > 0:
                curr_metrics, curr_heads = run_prediction(
                    model, batch, axis_discrete_predictions, batch.get_axis_discrete_syms(),
                    model.discrete_rotational_loss, get_match_sequence_discrete_rotational_symmetry
                )

                heads_used[1][:, perturbation_type, shape_type] += curr_heads
                metrics[1][:, perturbation_type, shape_type] += curr_metrics

            ###################
            # c_axis
            ###################
            if C_AXIS_PREDICTED_AMOUNT > 0:
                curr_metrics, curr_heads = run_prediction(
                    model, batch, axis_continue_predictions, batch.get_axis_continue_syms(),
                    model.continue_rotational_loss, get_match_sequence_continue_rotational_symmetry
                )

                heads_used[2][:, perturbation_type, shape_type] += curr_heads
                metrics[2][:, perturbation_type, shape_type] += curr_metrics

            counts[perturbation_type, shape_type] += 1

            if i > EXAMPLES_USED != -1:
                break
            else:
                i += 1


    # Aggregating metrics
    # Total
    plane_map = metrics[0][0, :, :].sum() / counts.sum()
    plane_phc = metrics[0][1, :, :].sum() / counts.sum()
    plane_loss = metrics[0][2, :, :].sum() / counts.sum()

    print("==============================")
    print("Metrics Summary")
    print("==============================")
    print(f"{'Plane Symmetry MAP:':<25}{plane_map.item():.3f}")
    print(f"{'Plane Symmetry PHC:':<25}{plane_phc.item():.3f}")
    print(f"{'Plane Symmetry Loss:':<25}{plane_loss.item():.3f}")

    print("==============================")
    print("Metrics Summary by Shape Type")
    print("==============================")
    for shape_type_text, shape_type_idx in SHAPE_TYPE.items():
        shape_map = metrics[0][0, :, shape_type_idx].sum() / counts[:, shape_type_idx].sum()
        shape_map = 0.0 if shape_map.isnan() else shape_map.item()
        sup = counts[:, shape_type_idx].sum()
        print(f"{f'Plane Symmetry MAP {shape_type_text}:':<45}{shape_map:.3f}{' ':<3}{f'({sup})'}")

    print("==============================")
    print("Metrics Summary by Perturbation")
    print("==============================")
    for perturbation_text, perturbation_idx in PERTURBATION_TYPE.items():
        perturbation_map = metrics[0][0, perturbation_idx, :].sum() / counts[perturbation_idx, :].sum()
        perturbation_map = 0.0 if perturbation_map.isnan() else perturbation_map.item()
        sup = counts[perturbation_idx, :].sum()
        print(f"{f'Plane Symmetry MAP {perturbation_text}:':<45}{perturbation_map:.3f}{' ':<3}{f'({sup})'}")

    print("==============================")
    print("Metrics Summary by Loss type")
    print("==============================")
    for loss_idx, loss_text in enumerate(["total"]+model.plane_loss_tag):
        loss = metrics[0][2 + loss_idx, :, :].sum() / counts.sum()
        loss = 0.0 if loss.isnan() else loss.item()
        sup = loss / plane_loss
        print(f"{f'Plane Symmetry {loss_text} loss:':<45}{loss:.3f}{' ':<3}{f'({sup:.2f})'}")

    print("==============================")
    print("Head Usage")
    print("==============================")

    distribution = heads_used[0].sum(dim=2).sum(dim=1)
    total = heads_used[0].sum()
    distribution = distribution / total
    torch.set_printoptions(linewidth=50)
    torch.set_printoptions(precision=3)
    torch.set_printoptions(sci_mode=False)
    print(distribution)