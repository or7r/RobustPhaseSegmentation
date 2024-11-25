# Do not touch the following imports
from focal_loss_wrapper import FocalLoss
from functools import partial
from torch.cuda.amp import GradScaler
from loguru import logger
from tqdm import tqdm
import numpy as np
import copy
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
from stc import STC
from ctc import CTC
from asg import ASG
import gtn
import sys

import wandb
from utils import get_distance_func
sys.path.insert(1, 'gtn_applications/criterions')


# from torch_asg import ASGLoss


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


class Chomp1d(nn.Module):
    def __init__(self, chomp_size, direction='right'):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.direction = direction

    def forward(self, x):
        if self.direction == 'right':
            return x[:, :, :-self.chomp_size].contiguous()
        elif self.direction == 'left':
            return x[:, :, self.chomp_size:].contiguous()

    # better print
    def __repr__(self):
        return self.__class__.__name__ + '(chomp_size=' + str(self.chomp_size) + ', direction=' + self.direction + ')'


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, is_casual=False):
        super(MultiStageModel, self).__init__()
        arch = 'casual' if is_casual else 'non-casual'
        self.tower_stage = TowerModel(
            num_layers, num_f_maps, dim, num_classes, arch)
        self.single_stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, 3, arch))
                                            for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = x
        middle_out, out = self.tower_stage(out, mask)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return middle_out, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, arch):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(
            num_layers, num_f_maps, dim, num_classes, 3, arch)
        self.stage2 = SingleStageModel(
            num_layers, num_f_maps, dim, num_classes, 5, arch)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size, arch):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        if arch == 'casual':
            self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualCasualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                        for i in range(num_layers)])
        else:
            self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size))
                                        for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]

# adopted from https://github.com/xmed-lab/TimeStamp-Surgical/blob/main/uncertainty_model.py


class DilatedResidualCasualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualCasualLayer, self).__init__()
        self.padding = 2 * int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.pad(x, [self.padding, 0], 'constant', 0)
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks,
                 num_layers,
                 num_f_maps,
                 dim,
                 num_classes,
                 ctc_weight=1,
                 ctc_downsample_rate=1/15,
                 entropy_weight=0,
                 smoothness_loss_weight=0,
                 confidence_weight=0,
                 ctc_varient='ctc',
                 stc_insertion_penalty=1,
                 ctc_distance='euclidean',
                 use_initial_psuedo_labels=False,
                 partial_pseudo_labels=False,
                 partial_pseudo_labels_method="uncertainty",
                 update_every=1,
                 mc_forward_passes=1,
                 uncertainty_thresh=0.1,
                 uncertainty_method="entropy",
                 loss_weights=None,
                 focal_loss=False,
                 focal_gamma=2.0,
                 max_ipw_weight=np.inf,
                 disable_mixed_precision=False,
                 is_casual=False,
                 loss_normalization=False,
                 temp=None,
                 ce_loss_weight=1.0,
                 ):

        self.loss_weights = loss_weights

        self.disable_mixed_precision = disable_mixed_precision

        if loss_weights is not None:
            if "ipw" in loss_weights:
                loss_weights = None  # Handled in train function
            elif isinstance(loss_weights, list):
                loss_weights = torch.tensor(loss_weights)
            else:
                raise ValueError(
                    "unvalid loss_weights, got {}".format(loss_weights))

        self.max_ipw_weight = max_ipw_weight

        self.ce_ignore_index = -100
        frame_wise_params = {"ignore_index": self.ce_ignore_index,
                             "reduction": "mean" if loss_normalization else "none"}
        self.loss_normalization = loss_normalization

        self.focal_loss = focal_loss

        if focal_loss:
            self.frame_wise_loss = partial(FocalLoss,
                                           gamma=focal_gamma,
                                           **frame_wise_params)
        else:

            self.frame_wise_loss = partial(nn.CrossEntropyLoss,
                                           **frame_wise_params)

        self.ce = self.frame_wise_loss(weight=loss_weights)
        self.ce_loss_weight = ce_loss_weight
        self.mse = nn.MSELoss(reduction='none')

        ctc_weight = float(ctc_weight)
        if ctc_varient == 'ctc':
            self.ctc_loss = CTC(blank=num_classes,
                                use_pt=True)
            extended_class_num = num_classes + int(ctc_weight > 0)
        elif ctc_varient == 'asg':
            self.num_replabels = 1
            self.use_garbage = True
            self.ctc_loss = ASG(num_classes=num_classes,
                                num_replabels=self.num_replabels,
                                use_garbage=self.use_garbage)

            extended_class_num = num_classes + \
                int(ctc_weight > 0) * \
                (self.num_replabels + int(self.use_garbage))
        elif ctc_varient == 'stc':
            self.ctc_loss = STC(blank_idx=0,
                                plast=stc_insertion_penalty, p0=stc_insertion_penalty)
            extended_class_num = num_classes + int(ctc_weight > 0)
        else:
            raise ValueError('ctc_varient must be ctc or asg or stc')

        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim,
                                     extended_class_num,
                                     is_casual=is_casual,
                                     )

        self.confidence_weight = confidence_weight
        self.smoothness_loss_weight = smoothness_loss_weight

        self.ctc_distance = ctc_distance
        self.distance = get_distance_func(ctc_distance)
        self.ctc_weight = ctc_weight
        self.ctc_downsample_rate = ctc_downsample_rate
        self.ctc_varient = ctc_varient

        self.blank_idx = num_classes
        self.num_classes = num_classes

        self.use_initial_psuedo_labels = use_initial_psuedo_labels

        self.partial_pseudo_labels = partial_pseudo_labels
        if partial_pseudo_labels:
            self.partial_pseudo_labels_method = partial_pseudo_labels_method
        else:
            self.partial_pseudo_labels_method = None
        self.entropy_weight = entropy_weight
        self.update_every = update_every
        self.mc_forward_passes = mc_forward_passes
        self.uncertainty_thresh = uncertainty_thresh
        self.uncertainty_method = uncertainty_method
        self.temp = 1.0 if temp is None else temp

    def confidence_loss(self, pred, confidence_mask, device):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            m_mask = torch.from_numpy(confidence_mask[b]).type(
                torch.float).to(device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = (pred[b, :, :-1] - pred[b, :, 1:])
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss

    def train(self, save_dir, batch_gen, writer, initial_epochs, num_epochs, batch_size, learning_rate, device, upload=False,
              ctc_start_epoch=np.inf):

        if self.loss_weights == "ipw":
            ipw_weights = batch_gen.calc_ipw_weights()
            ipw_weights = torch.clamp(ipw_weights, max=self.max_ipw_weight)
            self.ce = self.frame_wise_loss(weight=ipw_weights)
        elif self.loss_weights == "ipw_min_normalization":
            ipw_weights = batch_gen.calc_ipw_weights_min_normalization()
            ipw_weights = torch.clamp(ipw_weights, max=self.max_ipw_weight)
            self.ce = self.frame_wise_loss(weight=ipw_weights)
        elif self.loss_weights == "ipw_root":
            ipw_weights = batch_gen.calc_ipw_weights(apply_root=True)
            ipw_weights = torch.clamp(ipw_weights, max=self.max_ipw_weight)
            self.ce = self.frame_wise_loss(weight=ipw_weights)
        elif self.loss_weights == "ipw_log":
            ipw_weights = batch_gen.calc_ipw_weights(apply_log=True)
            ipw_weights = torch.clamp(ipw_weights, max=self.max_ipw_weight)
            self.ce = self.frame_wise_loss(weight=ipw_weights)

        self.model.train()
        self.model.to(device)
        self.ctc_loss.to(device)
        self.ce.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_epochs = initial_epochs
        scaler = GradScaler()

        print('start epoch of single supervision is:', start_epochs)
        for epoch in tqdm(range(num_epochs), disable=False):
            epoch_loss = 0
            correct = 0
            total = 0
            pseudo_labels_parition = 0
            blank_parition = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask, batch_confidence = batch_gen.next_batch(
                    batch_size)
                batch_input, batch_target, mask = batch_input.to(
                    device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=not self.disable_mixed_precision):
                    middle_pred, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                if epoch < start_epochs and self.use_initial_psuedo_labels:
                    batch_boundary = batch_gen.get_initial_labels(
                        batch_size, batch_input.size(-1))
                elif epoch < start_epochs:
                    batch_boundary = batch_gen.get_single_random(
                        batch_size, batch_input.size(-1))
                elif epoch >= start_epochs and self.partial_pseudo_labels_method == "uncertainty":
                    batch_boundary = batch_gen.get_boundary_with_uncertainty(
                        batch_size, batch_input.size(-1))
                else:
                    batch_boundary = batch_gen.get_boundary(
                        batch_size, middle_pred.detach())
                batch_boundary = batch_boundary.to(device)

                if self.ctc_weight > 0 and epoch >= ctc_start_epoch:
                    ctc_input_lengths = mask[:, 0, :].sum(axis=1).int()
                    if self.ctc_downsample_rate > 0:
                        ctc_batch_target = [F.interpolate(batch_target[i, :ctc_input_lengths[i]].view(1, 1, -1).float(),
                                            scale_factor=self.ctc_downsample_rate).int().view(-1)
                                            for i in range(batch_target.shape[0])]
                    else:
                        ctc_batch_target = [torch.unique_consecutive(batch_target[i, :ctc_input_lengths[i]])
                                            for i in range(batch_target.shape[0])]

                    # remove ce_ignore_index
                    ctc_batch_target = [ctc_batch_target[i][ctc_batch_target[i] != self.ce_ignore_index]
                                        for i in range(batch_target.shape[0])]
                    ctc_output_lengths = [ctc_batch_target[i].shape[0]
                                          for i in range(batch_target.shape[0])]
                    ctc_batch_target = torch.nn.utils.rnn.pad_sequence(
                        ctc_batch_target, batch_first=True, padding_value=-100)

                    ctc_preds = predictions
                    predictions, blank_mask = self.fill_pred(
                        predictions, return_mask=True)
                elif self.ctc_weight > 0:
                    predictions, blank_mask = self.fill_pred(
                        predictions, return_mask=True)
                else:
                    blank_mask = torch.ones(
                        predictions.shape[0], predictions.shape[1], predictions.shape[3]).to(device)

                loss = 0
                ce_loss = 0
                smoothness_loss = 0
                confidence_loss = 0
                total_ctc_loss = 0
                entropy_loss = 0

                for i, p in enumerate(predictions):
                    p_ce = p.transpose(2, 1).contiguous(
                    ).view(-1, self.num_classes)
                    if self.focal_loss:
                        p_ce = F.softmax(p_ce, dim=-1)
                    if self.loss_normalization:
                        ce_loss += self.ce_loss_weight * \
                            self.ce(p_ce, batch_boundary.view(-1))
                    else:
                        ce_loss += self.ce_loss_weight * (self.ce(p_ce, batch_boundary.view(-1))).sum(
                        ) / torch.sum(batch_boundary != self.ce_ignore_index)

                    smoothness_loss += self.smoothness_loss_weight * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask[:, :, 1:] * blank_mask[i, :, None, 1:])

                    if self.confidence_weight > 0:
                        confidence_loss += self.confidence_weight * \
                            self.confidence_loss(p, batch_confidence, device)

                    if self.ctc_weight > 0 and epoch >= ctc_start_epoch:
                        ctc_loss = 0
                        ctc_p = ctc_preds[i]

                        for j, ctc_bp in enumerate(ctc_p):
                            input_bp = ctc_bp[:, :ctc_input_lengths[j]]
                            target_bp = ctc_batch_target[j,
                                                         :ctc_output_lengths[j]]

                            input_bp = input_bp.unsqueeze(0).permute(0, 2, 1)
                            target_bp = (target_bp, )
                            args = ()

                            if self.ctc_varient == "stc":
                                input_bp = torch.nn.functional.log_softmax(
                                    input_bp, dim=2)

                                idx = torch.arange(
                                    input_bp.shape[2], device=input_bp.device)
                                with torch.no_grad():
                                    idx[[0, -1]] = idx[[-1, 0]]

                                input_bp = input_bp[:, :, idx]
                                input_bp = input_bp.permute(1, 0, 2)
                                # replace taget_bp as well
                                target_bp = target_bp[0]
                                target_bp[target_bp == 0] = self.num_classes
                                target_bp = (target_bp, )

                                target_bp = tuple(list(t.detach().cpu().numpy()) if isinstance(
                                    t, torch.Tensor) else t for t in target_bp)

                            ctc_loss += self.ctc_loss(input_bp,
                                                      target_bp, *args)

                        ctc_loss = self.ctc_weight * ctc_loss / len(ctc_p)
                        total_ctc_loss += ctc_loss

                    if self.entropy_weight > 0:
                        labeled_mask = (batch_boundary.view(-1)
                                        != self.ce_ignore_index).float()
                        labeled_mask = blank_mask[i].view(-1) * labeled_mask
                        entropy_loss += self.entropy_weight * self.entropy_regularization(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                                                                                          labeled_mask)

                loss = ce_loss + smoothness_loss + confidence_loss + total_ctc_loss + entropy_loss
                epoch_loss += loss.item()

                if self.disable_mixed_precision:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float() *
                            mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                pseudo_labels_parition += (((batch_boundary != self.ce_ignore_index)
                                           * mask[:, 0, :]).sum(1) / (mask[:, 0, :]).sum(1)).sum().item()
                blank_parition += 1 - (blank_mask * 1.0).mean()

            batch_gen.reset()

            writer.add_scalar('trainLoss', epoch_loss /
                              len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('trainAcc', float(correct)/total, epoch + 1)
            writer.add_scalar('pseudo_labels_parition', pseudo_labels_parition /
                              len(batch_gen.list_of_examples), epoch + 1)
            writer.add_scalar('blank_parition', blank_parition /
                              len(batch_gen.list_of_examples), epoch + 1)
            # added losses
            writer.add_scalar('ce_loss', ce_loss.item(), epoch + 1)
            writer.add_scalar('smoothness_loss',
                              smoothness_loss.item(), epoch + 1)
            writer.add_scalar('confidence_loss', confidence_loss.item() if isinstance(
                confidence_loss, torch.Tensor) else confidence_loss, epoch + 1)
            writer.add_scalar('total_ctc_loss', total_ctc_loss, epoch + 1)
            writer.add_scalar('entropy_loss', entropy_loss.item() if isinstance(
                entropy_loss, torch.Tensor) else entropy_loss, epoch + 1)
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

            if upload:
                wandb.log({"epoch": epoch + 1,
                           "trainLoss": epoch_loss / len(batch_gen.list_of_examples),
                           "trainAcc": float(correct)/total,
                           "pseudo_labels_parition": pseudo_labels_parition / len(batch_gen.list_of_examples),
                           "blank_parition": blank_parition / len(batch_gen.list_of_examples),
                           "ce_loss": ce_loss.item(),
                           "smoothness_loss": smoothness_loss.item(),
                           "confidence_loss": confidence_loss.item() if isinstance(confidence_loss, torch.Tensor) else confidence_loss,
                           "total_ctc_loss": total_ctc_loss,
                           "entropy_loss": entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss
                           })

            if self.partial_pseudo_labels_method == "uncertainty" and epoch >= start_epochs and epoch % self.update_every == 0:
                self.model.eval()
                enable_dropout(self.model)

                with torch.no_grad():
                    while batch_gen.has_next():
                        batch_input, batch_target, mask, batch_confidence = batch_gen.next_batch(
                            batch_size)
                        batch_input, batch_target, mask = batch_input.to(
                            device), batch_target.to(device), mask.to(device)

                        mc_probs = []

                        for _ in range(self.mc_forward_passes):

                            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=not self.disable_mixed_precision):
                                middle_pred, predictions = self.model(
                                    batch_input, mask)

                            if self.ctc_weight > 0:
                                probs = F.softmax(
                                    predictions[-1].detach().transpose(2, 1) / self.temp, dim=2)
                                blank_probs = probs[:, :, -1]
                                predictions = predictions[:, :, :-1]
                            else:
                                blank_probs = torch.zeros(
                                    (predictions.shape[1], predictions.shape[3])).to(device)

                            probs = F.softmax(
                                predictions[-1].detach().transpose(2, 1), dim=2)
                            mc_probs.append(probs)

                        mc_probs = torch.stack(mc_probs, dim=0)
                        std = torch.std(mc_probs, dim=0)
                        mc_probs = torch.mean(mc_probs, dim=0)
                        preds = torch.argmax(mc_probs, dim=2)

                        if self.uncertainty_method == "entropy":
                            uncertainty_scores = torch.sum(
                                -mc_probs * torch.log(mc_probs), dim=2)
                        elif self.uncertainty_method == "std":
                            uncertainty_scores = std.gather(
                                2, preds.unsqueeze(-1))
                            uncertainty_scores = uncertainty_scores.squeeze(-1)
                        else:
                            raise NotImplementedError(
                                "Uncertainty method must be entropy or std")

                        # Assert that blank remain uncertain
                        uncertainty_scores = uncertainty_scores + blank_probs * self.num_classes

                        batch_gen.update_pseudo_labels(
                            batch_size, preds, uncertainty_scores, self.uncertainty_thresh)

                self.model.train()
                batch_gen.reset()

        torch.save(self.model.state_dict(), save_dir +
                   "/epoch-" + str(epoch + 1) + ".model")
        torch.save(optimizer.state_dict(), save_dir +
                   "/epoch-" + str(epoch + 1) + ".opt")

    def fill_pred(self, predictions, blank_idx=None, return_mask=False):
        if self.ctc_weight > 0 and self.ctc_varient == "asg":
            predictions, blank_mask = self.fill_asg_pred(
                predictions, blank_idx=blank_idx)
        elif self.ctc_weight > 0 and (self.ctc_varient == "ctc" or self.ctc_varient == "stc"):
            predictions, blank_mask = self.fill_blank(
                predictions, blank_idx=blank_idx)
        elif self.ctc_weight > 0:
            raise ValueError("ctc_varient must be ctc or asg or stc")

        if return_mask:
            return predictions, blank_mask
        return predictions

    @staticmethod
    def fill_blank(predictions, blank_idx=None):

        # https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
        # modified for pytorch

        blank_idx = predictions.shape[2] - \
            1 if blank_idx is None else blank_idx
        legal_idx = torch.arange(
            predictions.shape[2], device=predictions.device)

        if isinstance(blank_idx, list):
            blank_mask = ~torch.isin(torch.argmax(predictions, dim=2), torch.tensor(
                blank_idx, device=predictions.device))
            legal_idx = legal_idx[~torch.isin(
                legal_idx, torch.tensor(blank_idx, device=predictions.device))]
        else:
            blank_mask = torch.argmax(predictions, dim=2) != blank_idx
            legal_idx = legal_idx[legal_idx != blank_idx]

        idx = torch.where(blank_mask, torch.arange(
            blank_mask.shape[2], device=blank_mask.device), 0)
        idx, _ = torch.cummax(idx, dim=2)
        predictions = torch.gather(predictions, 3, idx.unsqueeze(
            2).expand(-1, -1, predictions.shape[2], -1))
        predictions = predictions[:, :, legal_idx]
        return predictions, blank_mask

    def fill_asg_pred(self, predictions, blank_idx=None):
        blank_idx = predictions.shape[2] - \
            1 if blank_idx is None else blank_idx
        return self.fill_blank(predictions, blank_idx=[blank_idx] + [i for i in range(self.num_replabels)])

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate,
                save_certain=False, certain_window=25, certain_temp=0.5):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(
                model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                _, predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device))

                predictions = self.fill_pred(predictions)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()

                if save_certain:
                    def calc_uncertainty(logits, T=1):
                        logits = logits / T
                        probs = F.softmax(logits, dim=0)
                        return (- probs * torch.log(probs)).sum(dim=0)

                    def draw_graph(vid, folder_path="temp/uncertanity"):
                        import os
                        import matplotlib.pyplot as plt
                        from itertools import groupby
                        os.makedirs(folder_path, exist_ok=True)

                        def get_sequence_indices(r):
                            indices = []
                            index = 0
                            for _, group in groupby(r):
                                count = len(list(group))
                                indices.append(index)
                                index += count
                            return indices[1:]

                        def read_file(path):
                            return open(path).read().split("\n")[:-1]

                        fig, ax = plt.subplots()
                        for t in [0.25, 0.5, 1][::-1]:
                            uncertainty = calc_uncertainty(
                                predictions[-1].squeeze(), T=t)
                            ax.plot(uncertainty.cpu(), label=rf'$M_{{T}}={t}$')
                        ax.axhline([0.05], c='red', linestyle="--",
                                   label="Threshold")

                        # set dataset var by number of classes
                        dataset = "cholec80" if self.num_classes == 7 else "MultiBypass140"

                        r = read_file(f'data/{dataset}/groundTruth/{vid}')
                        indices = get_sequence_indices(r)

                        for index in indices:
                            ax.axvline(x=index + certain_window, color='black', linestyle='--',
                                       linewidth=1.5, label="Ground Truth\nSurrounding Window")
                            ax.axvline(x=index - certain_window, color='black', linestyle='--',
                                       linewidth=1.5, label="Ground Truth\nSurrounding Window")

                        # x tile is time
                        ax.set_xlabel('Time')
                        # remove ticks
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_ylabel('Uncertainty Measure')

                        handles, labels = ax.get_legend_handles_labels()
                        by_label = dict(zip(labels, handles))
                        ax.legend(by_label.values(), by_label.keys(
                        ), loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
                        fig.savefig(f"{folder_path}/{vid}.png",
                                    bbox_inches='tight')

                    # draw_graph(vid)

                    uncertainty = calc_uncertainty(
                        predictions[-1].squeeze(), T=certain_temp)

                    uncertainty_mask = uncertainty > 0.05
                    uncertainty_mask = uncertainty_mask.float()
                    assert certain_window % 2 == 1, "certain_window must be odd"
                    uncertainty_mask = F.conv1d(uncertainty_mask.view(1, 1, -1),
                                                torch.ones(1, 1, certain_window).to(
                                                    device),
                                                padding=certain_window // 2).squeeze()
                    uncertainty_mask = uncertainty_mask > 0

                assert predicted.shape[0] == features.shape[1], "Predicted shape: " + str(
                    predicted.shape[0]) + " Features shape: " + str(features.shape[1])

                recognition = []
                missing_ratio = 0

                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(
                        predicted[i].item())
                    recognition = np.concatenate(
                        (recognition, [list(actions_dict.keys())[index]]*sample_rate))

                missing_ratio /= len(predicted)

                logger.debug("Missing ratio: " + str(missing_ratio))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

                if save_certain:
                    uncertainty_mask = F.interpolate(uncertainty_mask.view(1, 1, -1).float(),
                                                     scale_factor=sample_rate).squeeze().bool()
                    recognition[uncertainty_mask.cpu().numpy()] = "uncertain"

                    os.makedirs(results_dir + "_certain/", exist_ok=True)
                    f_ptr = open(results_dir + "_certain/" +
                                 f_name + ".txt", "w")
                    f_ptr.write('\n'.join(recognition))
                    f_ptr.write('\n')
                    f_ptr.close()

    def entropy_regularization(self, logits, mask):
        prob = F.softmax(logits, dim=1)
        entropy = torch.sum(-prob * torch.log(prob), dim=1)
        entropy = entropy * mask
        return torch.sum(entropy) / torch.sum(mask)
