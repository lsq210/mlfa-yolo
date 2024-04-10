
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr, emojis, clean_url
import ultralytics.utils.callbacks.tensorboard as tb_module
import numpy as np
import time
import torch
from torch.nn import functional as F
from torch import distributed as dist
import warnings
from torchvision.ops import roi_align, box_convert

from .layers import Discriminator

class MLFATrainer(DetectionTrainer):
    def __init__(self, target_domain_data_cfg, *args, **kwargs):
        super(MLFATrainer, self).__init__(*args, **kwargs)
        self.t_trainset, self.t_testset = self.get_dataset_t(target_domain_data_cfg)
        self.t_iter = None
        self.t_train_loader = None

        self.feature_handler = []
        self.feature_layer_idx: list[int] = [4, 6, 8]
        self.feature_roi_size = [int(i * (self.args.imgsz / 640)) for i in [80, 40, 20]]
        self.model_hooked_features: None | list[torch.tensor] = None

        self.instance_handler = []
        self.instance_layer_idx: list[int] = [15, 18, 21]
        self.instance_roi_size = [int(i * (self.args.imgsz / 640)) for i in [80, 40, 20]]
        self.model_hooked_instance: None | list[torch.tensor] = None

        self.feature_discriminator_model = None
        self.instance_discriminator_model = None
        self.additional_models = []
        self.add_callback('on_train_start', self.init_helper_model)

    def get_dataset_t(self, target_domain_data_cfg):
        # Load target domain's dataset
        try:
            if target_domain_data_cfg in ('yaml', 'yml') or self.args.task in ('detect', 'segment', 'pose'):
                t_data = check_det_dataset(target_domain_data_cfg)
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(target_domain_data_cfg)}' error ❌ {e}")) from e

        return t_data["train"], t_data.get("val") or t_data.get("test")

    def init_helper_model(self, *args, **kwargs):
        self.feature_discriminator_model = Discriminator(chs = [128, 256, 512], amp=self.amp).to(self.device)
        self.instance_discriminator_model = Discriminator(chs = [128, 256, 512], amp=self.amp).to(self.device)
        self.additional_models.append(self.feature_discriminator_model)
        self.additional_models.append(self.instance_discriminator_model)

    def get_t_batch(self):
        if self.t_iter is None:
            self.t_train_loader = self.get_dataloader(self.t_trainset, batch_size=self.batch_size, rank=RANK, mode='train')
            self.t_iter = iter(self.t_train_loader)
        try:
            batch = next(self.t_iter)
        except StopIteration:
            self.t_iter = iter(self.t_train_loader)
            batch = next(self.t_iter)
        return batch

    def activate_hook(self, feature_layer_indices: list[int] = None, targets_layer_indices: list[int] = None):
        if feature_layer_indices is not None:
            self.feature_layer_idx = feature_layer_indices
        if targets_layer_indices is not None:
            self.instance_layer_idx = targets_layer_indices
        self.model_hooked_features = [None for _ in self.feature_layer_idx]
        self.model_hooked_instance = [None for _ in self.instance_layer_idx]
        self.feature_handler = \
            [self.model.model[l].register_forward_hook(self.hook_fn('feature', i)) for i, l in enumerate(self.feature_layer_idx)]
        self.instance_handler = \
            [self.model.model[l].register_forward_hook(self.hook_fn('instance', i)) for i, l in enumerate(self.instance_layer_idx)]

    def deactivate_hook(self):
        if self.feature_handler is not None:
            for hook in self.feature_handler:
                hook.remove()
            self.model_hooked_features = None
            self.feature_handler = []
        if self.instance_handler is not None:
            for hook in self.instance_handler:
                hook.remove()
            self.model_hooked_instance = None
            self.instance_handler = []

    def hook_fn(self, featureOrInstance: str, hook_idx: int):

        def hook(m, i, o):
            if featureOrInstance == 'feature':
                self.model_hooked_features[hook_idx] = o
            else:
                self.model_hooked_instance[hook_idx] = o

        return hook

    def get_dis_output_from_hooked_instance(self, batch):
        if self.model_hooked_instance is not None:
            bbox_batch_idx = batch['batch_idx'].unsqueeze(-1)
            bbox = batch['bboxes']
            bbox = box_convert(bbox, 'cxcywh', 'xyxy')
            rois = []
            for fidx, f in enumerate(self.model_hooked_instance):
                f_bbox = bbox * f.shape[-1]
                f_bbox = torch.cat((bbox_batch_idx, f_bbox), dim=-1)
                f_roi = roi_align(f, f_bbox.to(f.device), output_size=self.instance_roi_size[fidx], aligned=True)
                rois.append(f_roi)
            dis_output = self.instance_discriminator_model(rois)
            return dis_output
        else:
            return None

    def get_dis_output_from_hooked_features(self):
        if self.model_hooked_features is not None:
            features = []
            for f in self.model_hooked_features:
                features.append(f)
            dis_output = self.feature_discriminator_model(features)
            return dis_output
        else:
            return None

    def optimizer_step(self, optims: None | list[torch.optim.Optimizer] = None):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.unscale_(o)
        # Clip gradient norm
        max_norm = 10.0
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm)  # clip gradients
        if len(self.additional_models) > 0:
            for m in self.additional_models:
                torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=max_norm * 2)
        # Step optimizers
        self.scaler.step(self.optimizer)
        if optims is not None:
            for o in optims:
                # check if the optimizer has gradients
                if o.param_groups[0]['params'][0].grad is not None:
                    self.scaler.step(o)

        self.scaler.update()
        # Zero optimizers' grads
        self.optimizer.zero_grad()
        if optims is not None:
            for o in optims:
                o.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        self.activate_hook()
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            source_feature_critics, target_feature_critics = None, None
            source_instance_critics, target_instance_critics = None, None
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                    # Custom code here
                    ## 源域图像级对齐
                    source_feature_critics = self.get_dis_output_from_hooked_features()
                    ## 源域实例级对齐
                    source_instance_critics = self.get_dis_output_from_hooked_instance(batch)
                    t_batch = self.get_t_batch() # will update hooked features and instance
                    t_batch = self.preprocess_batch(t_batch)
                    t_loss, t_loss_item = self.model(t_batch)
                    self.loss += t_loss
                    ## 目标域实例级对齐
                    target_feature_critics = self.get_dis_output_from_hooked_features()
                    ## 目标域实例级对齐
                    target_instance_critics = self.get_dis_output_from_hooked_instance(t_batch)

                    if 6 < epoch < self.args.epochs - 50:
                        feature_threshold = 20
                        loss_feature_d = (F.relu(torch.ones_like(source_feature_critics) * feature_threshold + source_feature_critics)).mean()
                        loss_feature_d += (F.relu(torch.ones_like(target_feature_critics) * feature_threshold - target_feature_critics)).mean()

                        targets_threshold = 20
                        loss_targets_d = (F.relu(torch.ones_like(source_instance_critics) * targets_threshold + source_instance_critics)).mean()
                        loss_targets_d += (F.relu(torch.ones_like(target_instance_critics) * targets_threshold - target_instance_critics)).mean()
                    else:
                        loss_feature_d = 0
                        loss_targets_d = 0
                    self.loss += loss_feature_d * 2
                    self.loss += loss_targets_d * 2

                # Backward
                self.scaler.scale(self.loss).backward()
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step(optims=[self.feature_discriminator_model.optim, self.instance_discriminator_model.optim])
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    tb_module.WRITER.add_scalar('train/critic-feature-source', source_feature_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-feature-target', target_feature_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-instance-source', source_instance_critics.mean(), ni)
                    tb_module.WRITER.add_scalar('train/critic-instance-target', target_instance_critics.mean(), ni)
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.deactivate_hook()
                    self.save_model()
                    self.run_callbacks('on_model_save')
                    self.activate_hook()

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')

        self.deactivate_hook()
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
