from typing import Callable, Union

import lightning
import torch
import wandb

from src.metrics.eval_script import calculate_metrics_from_predictions, get_match_sequence_plane_symmetry, \
    get_match_sequence_continue_rotational_symmetry, \
    get_match_sequence_discrete_rotational_symmetry
from src.model.CenterNNormalsNet import CenterNNormalsNet
from src.model.losses.ConfidenceLoss import ConfidenceLoss
from src.model.losses.DiscreteRotationalSymmetryLoss import DiscreteRotationalSymmetryLoss
from src.model.losses.DistanceLoss import DistanceLoss
from src.model.losses.NormalLoss import NormalLoss
from src.model.losses.ReflectionSymmetryDistance import ReflectionSymmetryDistance
from src.model.losses.ReflectionSymmetryLoss import ReflectionSymmetryLoss
from src.model.losses.RotationalSymmetryDistance import RotationalSymmetryDistance
from src.model.losses.RotationalSymmetryLoss import RotationalSymmetryLoss
from src.model.matchers.SimpleMatcher import SimpleMatcher
from src.model.matchers.cost_matrix_methods import calculate_cost_matrix_normals

from src.dataset.SymDatasetItem import SHAPE_TYPE
from src.model.wandb.render_shape import wandb_log_gpu
from src.model.wandb.worst_losses_tracker import WorstLossesTracker

class LightingCenterNNormalsNet(lightning.LightningModule):
    def __init__(self,
                 amount_of_plane_normals_predicted: int = 32,
                 amount_of_axis_discrete_normals_predicted: int = 16,
                 amount_of_axis_continue_normals_predicted: int = 16,
                 plane_loss: Union[ReflectionSymmetryLoss, str] = "default",
                 discrete_rotational_loss: Union[DiscreteRotationalSymmetryLoss, str] = "default",
                 continue_rotational_loss: Union[RotationalSymmetryLoss, str] = "default",
                 w1: float = 1.0,
                 w2: float = 1.0,
                 w3: float = 1.0,
                 eps: float = 0.01,
                 theta: float = 0.00015230484,  # 1° between axis/normals
                 confidence_threshold: float = 0.01,
                 rot_angle_threshold: float = 0.0174533,  # 1° of difference between rot angles
                 cost_matrix_method: Callable = calculate_cost_matrix_normals,
                 print_losses: bool = False,
                 use_bn: bool = False,
                 normalize_normals: bool = True,
                 encoder: str = "pointnet",
                 n_points: int = 8192,
                 use_wandb: bool = True,
                 worst_losses_max_entries: int = 20,
                 ):
        super().__init__()
        self.use_bn = use_bn
        self.n_points = n_points
        self.encoder_used = encoder
        self.normalize_normals = normalize_normals
        self.print_losses = print_losses
        self.cost_matrix_method = cost_matrix_method
        self.matcher = SimpleMatcher(self.cost_matrix_method, self.device)
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.use_wandb = use_wandb
        self.worst_losses_max_entries = worst_losses_max_entries

        self.worst_losses_tracker = WorstLossesTracker(max_entries=self.worst_losses_max_entries)	# dict with the worst losses (e.g. fn, loss, batch_idx, batch, tag)

        if plane_loss == "default":
            self.plane_loss = ReflectionSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                reflection_symmetry_distance_weight=0.1,
                reflection_symmetry_distance=ReflectionSymmetryDistance()
            )
        else:
            self.plane_loss = plane_loss
        self.plane_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "ref_sym_distance",
        ]

        if discrete_rotational_loss == "default":
            self.discrete_rotational_loss = DiscreteRotationalSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                angle_weight=1.0, angle_loss=DistanceLoss(),
                rotational_symmetry_distance_weight=0.1,
                rotational_symmetry_distance=RotationalSymmetryDistance()
            )
        else:
            self.discrete_rotational_loss = discrete_rotational_loss

        self.discrete_rotational_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "rot_sym_distance",
            "angle",
        ]
        if continue_rotational_loss == "default":
            self.continue_rotational_loss = RotationalSymmetryLoss(
                confidence_weight=1.0, confidence_loss=ConfidenceLoss(weighted=True),
                normal_weight=1.0, normal_loss=NormalLoss(),
                distance_weight=1.0, distance_loss=DistanceLoss(),
                rotational_symmetry_distance_weight=0.1,
                rotational_symmetry_distance=RotationalSymmetryDistance()
            )
        else:
            self.continue_rotational_loss = continue_rotational_loss
        self.continue_rotational_loss_tag = [
            "confidence",
            "normal",
            "distance",
            "rot_sym_distance"
        ]

        self.net = CenterNNormalsNet(
            amount_of_plane_normals_predicted,
            amount_of_axis_discrete_normals_predicted,
            amount_of_axis_continue_normals_predicted,
            use_bn=self.use_bn,
            normalize_normals=self.normalize_normals,
            encoder=encoder,
            n_points=self.n_points
        )
        self.eps = eps
        self.theta = theta
        self.confidence_threshold = confidence_threshold
        self.rot_angle_threshold = rot_angle_threshold
        self.metric_param_dict = {
            "eps": self.eps,
            "theta": self.theta,
            "confidence_threshold": self.confidence_threshold,
            "rot_angle_threshold": self.rot_angle_threshold,
        }

        # If warning concerns you read this https://github.com/Lightning-AI/pytorch-lightning/discussions/13615
        # Honestly idk will leave it like this for now
        self.save_hyperparameters(ignore=["net"]) # , "plane_loss", "discrete_rotational_loss", "continue_rotational_loss"

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def _log(
            self, metric_val, metric_name, sym_tag, step_tag, batch_size,
            on_step=True, on_epoch=True, prog_bar=False, sync_dist=True
    ):
        self.log(f"{sym_tag}_{step_tag}_{metric_name}", metric_val, on_step=on_step, on_epoch=on_epoch,
                 prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist)
        '''
        if self.use_wandb:
            wandb.log(f"{sym_tag}_{step_tag}_{metric_name}", metric_val, on_step=on_step, on_epoch=on_epoch,
                 prog_bar=prog_bar, batch_size=batch_size, sync_dist=sync_dist)
            wandb.log({f'plane_val_map_epoch': loss, 'epoch': self.current_epoch})
            wandb.log({f'plane_val_phc_epoch': loss, 'epoch': self.current_epoch})
        '''

    def _process_prediction(self,
                            batch, sym_pred, sym_true,
                            loss_fun, sym_tag, step_tag,
                            losses_tags, metrics_match_sequence_fun):
        c_hat, match_pred, match_true, pred2true, true2pred = self.matcher.get_optimal_assignment(batch.get_points(),
                                                                                                  sym_pred, sym_true)
        bundled_predictions = (batch, sym_pred, c_hat, match_pred, match_true)
        loss, others = loss_fun(bundled_predictions)

        eval_predictions = [(batch.get_points(), sym_pred, sym_true)]
        map, phc, pr_curve = calculate_metrics_from_predictions(eval_predictions, metrics_match_sequence_fun,
                                                                self.metric_param_dict)

        for idx in range(others.shape[0]):
            self._log(others[idx], f"loss_{losses_tags[idx]}", sym_tag, step_tag, batch.size)

        self._log(loss, "loss", sym_tag, step_tag, batch.size)
        self._log(map, "map", sym_tag, step_tag, batch.size)
        self._log(phc, "phc", sym_tag, step_tag, batch.size)

        return loss, map, phc

    def _step(self, batch, batch_idx, step_tag):
        if self.use_wandb and batch_idx == 0 and self.current_epoch == 0:
            print(f'Using wandb')
            #wandb.init(project='symmetria-ablation-test')		# Variable from config inside `trainer` are available only after having called fit()
            wandb.init(project='-'.join(self.trainer.logger.name.split('-')[:6]))	# e.g. symmetria-ablation-noise-undersampling-cylinder-1000
            #print(f'Wandb initialized - config: {wandb.config}')

            print(f'-= Renaming current run to: {self.trainer.logger.name} =-')
            wandb.run.name    = self.trainer.logger.name
            #wandb.run.project = 'symmetria-ablation-test-'+self.trainer.logger.name.split('-')[2] 		# (class name, e.g. 'astroid', 'citrus', etc.)
            wandb.run.save()
        batch.device = self.device
        self.matcher.device = self.device
        #print(f'Batch[0]: {batch.get_filenames()[0]} - {batch.get_shape_type_classification_labels()[0]}')
        #print(f'Batch[0]: {batch.get_points()[0]}  - {batch.get_plane_syms()[0]}')
        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)
        loss = torch.tensor(0.0, device=points.device)

        if plane_predictions is not None:
            plane_loss, plane_map, plane_phc = self._process_prediction(
                batch, plane_predictions, batch.get_plane_syms(), self.plane_loss,
                "plane", step_tag, self.plane_loss_tag, get_match_sequence_plane_symmetry,
            )
            loss += plane_loss * self.w1

        if axis_discrete_predictions is not None:
            discrete_axis_loss, map_discrete_axis, phc_discrete_axis = self._process_prediction(
                batch, axis_discrete_predictions, batch.get_axis_discrete_syms(), self.discrete_rotational_loss,
                "d_axis", step_tag, self.discrete_rotational_loss_tag, get_match_sequence_discrete_rotational_symmetry
            )
            loss += discrete_axis_loss * self.w2

        if axis_continue_predictions is not None:
            continue_axis_loss, map_continue_axis, phc_continue_axis = self._process_prediction(
                batch, axis_continue_predictions, batch.get_axis_continue_syms(), self.continue_rotational_loss,
                "c_axis", step_tag, self.continue_rotational_loss_tag, get_match_sequence_continue_rotational_symmetry

            )
            loss += continue_axis_loss * self.w3

        self._log(loss, "loss", "total", step_tag, batch.size, prog_bar=True)

        #print(f'plane_predictions: {plane_predictions}')

        if self.use_wandb:
            if False:						# use this for debugging purposes
                if batch_idx % 101 == 0 and batch_idx > 0:
                    self.send_worst_losses_to_wandb()

            wandb.log({f'{step_tag}_loss': loss, 'step': batch_idx})

            if plane_predictions is not None:
                wandb.log({f'{step_tag}_plane_loss': plane_loss, 'step': batch_idx})
                wandb.log({f'{step_tag}_plane_map' : plane_map,  'step': batch_idx})
                wandb.log({f'{step_tag}_plane_phc' : plane_phc,  'step': batch_idx})

            if axis_discrete_predictions is not None:
                wandb.log({f'{step_tag}_discrete_axis_loss': discrete_axis_loss, 'step': batch_idx})

            if axis_continue_predictions is not None:
                wandb.log({f'{step_tag}_continue_axis_loss': continue_axis_loss, 'step': batch_idx})

            fn, cl = self.get_fn_and_class(batch)
            entry = {'fn': fn, 'loss': loss, 'batch_idx': batch_idx, 'batch': batch, 'train_val_test_tag': step_tag, 'class_id': cl,
			'plane_predictions': plane_predictions,
			'axis_discrete_predictions': axis_discrete_predictions, 'axis_continue_predictions': axis_continue_predictions}
            self.worst_losses_tracker.add(entry=entry)

            if batch_idx % 1000 == 0:
                wandb_log_gpu(batch, preds=plane_predictions, filename=fn, shape_class=cl, loss=loss, train_valid_test_tag=step_tag,
				wandb_project="symmetry_visualization", init_and_finalize=False)

        return loss

    def get_fn_and_class(self, batch):
        fn = batch.get_filenames()[0]
        cl = batch.get_shape_type_classification_labels()[0]
        cl = int(torch.argmax(cl))
        cl = list(SHAPE_TYPE.keys())[list(SHAPE_TYPE.values()).index(cl)]
        return fn, cl

    def send_worst_losses_to_wandb(self):
        worst_losses = self.worst_losses_tracker.get_entries()
        step_tag     = worst_losses[0]["train_val_test_tag"]
        print(f'Epoch {self.current_epoch}: sending {len(worst_losses)} {step_tag} worst losses to WandB...\n', flush=True)

        # create a wandb.Table() with corresponding columns
        #columns = ["id", "image", "prediction", "truth"]
        columns = ["id", "loss", "fn", "class", "top_loss_str", "str"]

        worst_losses_list = []

        for idx, entry in enumerate(worst_losses):
            #wandb.log({f'{step_tag}_epoch_{self.current_epoch}_top_losses': f'{idx}: {entry["loss"]} - {entry["class_id"]} - {entry["fn"]}'})
            top_loss_str = f'{step_tag}_epoch_{self.current_epoch}_top_losses'
            worst_losses_list.append([idx, f'{entry["loss"]:.2f}', f'{entry["fn"]}', f'{entry["class_id"]}',
					top_loss_str, f'{idx}: {entry["loss"]:.2f} - {entry["class_id"]} - {entry["fn"]}'])	# TODO: this stuff here is almost useless now...

            wandb_log_gpu(batch=entry['batch'], preds=entry['plane_predictions'], filename=entry['fn'], shape_class=entry['class_id'], loss=entry['loss'],
				train_valid_test_tag=top_loss_str, wandb_project="symmetry_visualization", init_and_finalize=False)
        test_table = wandb.Table(data=worst_losses_list, columns=columns)
        wandb.log({f'{step_tag}_epoch_{self.current_epoch}_top_losses': test_table})

    def on_train_epoch_start(self):
        if self.use_wandb:
            self.worst_losses_tracker.empty()
            #print(f'self.val_los
    def on_validation_epoch_start(self):
        if self.use_wandb:
            self.worst_losses_tracker.empty()
    def on_train_epoch_end(self):
        if self.use_wandb:
            self.send_worst_losses_to_wandb()
            #print(f'{self.trainer.logs["val"]["map"]:.2f}', flush=True)	# TODO: .logged_metrics
            #print(f'{self.trainer.logs["val"]["map"]:.2f}', flush=True)	# TODO: .logged_metrics
            #wandb.log({f'plane_val_map_epoch': loss, 'epoch': self.current_epoch})
            #wandb.log({f'plane_val_phc_epoch': loss, 'epoch': self.current_epoch})
    def on_validation_epoch_end(self):
        if self.use_wandb:
            self.send_worst_losses_to_wandb()
            '''
            print(f'{self.current_epoch = }')
            print(f'{self.trainer.callback_metrics = }')
            if 'val_loss' in self.trainer.callback_metrics:
                val_loss_epoch = self.trainer.callback_metrics["val_loss"]
            if 'val_loss_epoch' in self.trainer.callback_metrics:
                val_loss_epoch = self.trainer.callback_metrics["val_loss_epoch"]
            print(f"{val_loss_epoch = }")
            wandb.log({f'val_loss_epoch ': val_loss_epoch , 'epoch': self.current_epoch})
            '''
            if False:						# use this for debugging purposes
                print(f'{self.trainer.callback_metrics = }')
            total_val_loss_epoch = self.trainer.callback_metrics["total_val_loss_epoch"]
            wandb.log({f'total_val_loss_epoch': total_val_loss_epoch, 'epoch': self.current_epoch})
            if 'plane_val_map_epoch' in self.trainer.callback_metrics:
                plane_val_map_epoch = self.trainer.callback_metrics["plane_val_map_epoch"]
                wandb.log({f'plane_val_map_epoch': plane_val_map_epoch, 'epoch': self.current_epoch})
            if 'plane_val_phc_epoch' in self.trainer.callback_metrics:
                plane_val_phc_epoch = self.trainer.callback_metrics["plane_val_phc_epoch"]
                wandb.log({f'plane_val_phc_epoch': plane_val_phc_epoch, 'epoch': self.current_epoch})
            '''
            if hasattr(self, 'validation_step_outputs'):
                print(f'{self.validation_step_outputs} = ')
                sys.exit(1)
            '''

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        batch.device = self.device
        self.matcher.device = self.device

        points = torch.stack(batch.get_points())
        points = torch.transpose(points, 1, 2).float()

        plane_predictions, axis_discrete_predictions, axis_continue_predictions = self.net.forward(points)

        return batch, plane_predictions, axis_discrete_predictions, axis_continue_predictions

    def on_after_backward(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                if param.grad.isnan().any():
                    print(f"{name} got nan!")
