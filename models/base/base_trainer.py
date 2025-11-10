# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import random
import time
import torch
import numpy as np
from utils.util import Logger, ValueWindow
from torch.utils.data import DataLoader

import torch.nn.functional as F
from transformers import get_inverse_sqrt_schedule, get_constant_schedule

import accelerate
from accelerate.utils import ProjectConfiguration

from models.base.base_sampler import VariableSampler
from torch.utils.data.distributed import DistributedSampler

def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


class BaseTrainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        from torch.utils.tensorboard import SummaryWriter

        self.sw = None
        if self.accelerator.is_main_process:
            try:
                log_dir = os.path.join(self.exp_dir, "tensorboard")
                os.makedirs(log_dir, exist_ok=True)
                self.sw = SummaryWriter(log_dir=log_dir)
                self.logger.info(f"TensorBoard SummaryWriter initialized at {log_dir}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize TensorBoard writer: {e}")
                self.sw = None

        self.time_window = ValueWindow(100)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        self.checkpoint_backup_dir = os.path.join(self.exp_dir, "checkpoint_backup")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.checkpoint_backup_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(
                self.train_dataloader, self.valid_dataloader
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # Resume or Finetune
        try:
            with self.accelerator.main_process_first():
                if args.resume:
                    ## Automatically resume according to the current exprimental name
                    print(
                        "Automatically resuming from latest checkpoint in {}...".format(
                            self.checkpoint_dir
                        )
                    )
                    start = time.monotonic_ns()
                    ckpt_path = self._load_model(
                        checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                    )
                    end = time.monotonic_ns()
                    print(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
        except Exception as e:
            print(e)
            import traceback

            print(traceback.format_exc())
            print("Resume failed")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # self.task_type = "VC"
        # if self.accelerator.is_main_process:
        #     self.logger.info("Task type: {}".format(self.task_type))

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    @staticmethod
    def _set_random_seed(seed):
        r"""Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            all_ckpts = os.listdir(checkpoint_dir)
            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            ls = list(all_ckpts)
            ls = [os.path.join(checkpoint_dir, i) for i in ls]
            ls.sort(key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
            if self.accelerator.is_main_process:
                self.logger.info("Resume from {}".format(checkpoint_path))

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            self.accelerator.load_state(input_dir=checkpoint_path)

            # set epoch and step
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1])
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1])

            if self.accelerator.is_main_process:
                self.logger.info(
                    "Resume from {}, epoch: {}, step: {}".format(
                        checkpoint_path, self.epoch, self.step
                    )
                )

        elif resume_type == "finetune":
            # Load only the model weights
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            if self.accelerator.is_main_process:
                self.logger.info("Load model weights for finetune...")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(
                    p.numel() for p in model[key].parameters() if p.requires_grad
                )
        else:
            model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model_param

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)

    def _build_model(self):
        raise NotImplementedError

    def _build_dataset(self):
        raise NotImplementedError

    def _build_dataloader(self):
        Dataset, Collator = self._build_dataset()

        # -----------------------------
        # 1️⃣ Dataset 인스턴스 생성
        # -----------------------------
        train_dataset = Dataset(cfg=self.cfg, is_valid=False)
        valid_dataset = Dataset(cfg=self.cfg, is_valid=True)

        train_collate = Collator(self.cfg)
        valid_collate = Collator(self.cfg)

        # -----------------------------
        # 2️⃣ Dynamic batchsize (길이 기반)
        # -----------------------------
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")

            t = time.time()
            if self.accelerator.is_main_process:
                print("Start batching...")

            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )

            if self.accelerator.is_main_process:
                info = "Time taken to batch: {:.1f}s, #batches = {}".format(
                    time.time() - t, len(batch_sampler)
                )
                print(info)
                self.logger.info(info)

            np.random.seed(self.cfg.train.random_seed)
            np.random.shuffle(batch_sampler)

            if self.accelerator.is_main_process:
                print(batch_sampler[:1])

            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
                prefetch_factor=32,
            )

        # ✅ Validation: 고정 batch로 생성
            valid_sampler = (
                DistributedSampler(valid_dataset, shuffle=False)
                if self.accelerator.num_processes > 1 else None
            )

            valid_loader = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                shuffle=False,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            self.accelerator.wait_for_everyone()

    # -----------------------------
    # 3️⃣ 일반 batchsize 모드
    # -----------------------------
        else:
            print("Use Normal Batchsize......")

            # ✅ 분산 훈련 샘플러 추가 (shuffle=True)
            train_sampler = (
                DistributedSampler(train_dataset, shuffle=True)
                if self.accelerator.num_processes > 1 else None
            )
            train_loader = DataLoader(
                train_dataset,
                sampler=train_sampler,
                shuffle=(train_sampler is None),
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            # ✅ 분산 검증 샘플러 추가 (중복 평가 방지)
            valid_sampler = (
                DistributedSampler(valid_dataset, shuffle=False)
                if self.accelerator.num_processes > 1 else None
            )
            valid_loader = DataLoader(
                valid_dataset,
                sampler=valid_sampler,
                shuffle=False,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            # num_warmup_steps=self.cfg.train.lr_warmup_steps,  # TODO: need to check wheather need to multiply by num_processes
            num_warmup_steps=self.cfg.train.lr_warmup_steps
            * self.accelerator.num_processes,
        )
        return lr_scheduler

    def _build_criterion(self):
        criteria = dict()
        criteria["l1_loss"] = torch.nn.L1Loss(reduction="mean")
        criteria["l2_loss"] = torch.nn.MSELoss(reduction="mean")
        criteria["ce_loss"] = torch.nn.CrossEntropyLoss(reduction="none")
        return criteria

    def write_summary(self, losses, stats):
        if self.sw is None:
            return
        for key, value in losses.items():
            self.sw.add_scalar(f"Train/{key}", value, self.step)

    def write_valid_summary(self, losses, stats):
        if self.sw is None:
            return
        for key, value in losses.items():
            self.sw.add_scalar(f"Valid/{key}", value, self.step)

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch, is_valid: bool = False):
        raise NotImplementedError

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        # ✅ 분산 샘플러 epoch 설정 추가 (안정성 향상)
        if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
            self.train_dataloader.sampler.set_epoch(self.epoch)

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        # Calculate the number of batches to skip, only skip when resume_skip_steps is enabled in the configuration
        steps_to_skip = 0
        if (
            hasattr(self.cfg.train, "resume_skip_steps")
            and self.cfg.train.resume_skip_steps
            and hasattr(self, "step")
            and self.step > 0
        ):
            steps_to_skip = self.step
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Resume skip steps enabled, skipping first {steps_to_skip} steps..."
                )

            # If dynamic batch size is used, we need to modify batch_sampler
            if self.cfg.train.use_dynamic_batchsize:
                if hasattr(self.train_dataloader, "batch_sampler"):
                    self.train_dataloader.batch_sampler.skip_steps(steps_to_skip)
            # If normal batch size is used, we need to modify sampler
            else:
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(self.epoch)

                    # Calculate the number of samples to skip
                    samples_to_skip = (
                        steps_to_skip
                        * self.cfg.train.batch_size
                        * self.accelerator.num_processes
                    )
                    if isinstance(
                        self.train_dataloader.sampler,
                        torch.utils.data.DistributedSampler,
                    ):
                        self.train_dataloader.sampler.set_start_index(samples_to_skip)
                    elif hasattr(self.train_dataloader.sampler, "skip_samples"):
                        self.train_dataloader.sampler.skip_samples(samples_to_skip)

        # Track the current number of batches processed
        current_batch = steps_to_skip

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            self.current_loss = float(total_loss)
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Steps/Train {}".format(key): loss},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

            current_batch += 1

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))

            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path, safe_serialization=True)
            self.logger.info("Finished saving state.")

            if (
                hasattr(self.cfg.train, "save_checkpoints_backup_steps")
                and self.step % self.cfg.train.save_checkpoints_backup_steps == 0
            ):
                try:
                    backup_path = os.path.join(
                        self.checkpoint_backup_dir, checkpoint_filename
                    )
                    shutil.copytree(path, backup_path)
                    self.logger.info("Saving backup state to {}...".format(backup_path))
                except Exception as e:
                    self.logger.error("Failed to save backup state: {}".format(e))

    def train_loop(self):
        """Train + Validation loop with Early Stopping."""
        self.accelerator.wait_for_everyone()

        # 🧠 Early Stopping 설정
        patience = getattr(self.cfg.train, "early_stopping_patience", 5)  # 개선 없는 epoch 허용 횟수
        best_val_loss = float("inf")
        epochs_no_improve = 0

        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n" + "-" * 32)
                self.logger.info(f"Epoch {self.epoch}")

            # -----------------
            # Train
            # -----------------
            train_total_loss, train_losses = self._train_epoch()
            if self.accelerator.is_main_process:
                for key, loss in train_losses.items():
                    self.logger.info(f"  |- Train/{key}: {loss:.6f}")

            # -----------------
            # Validation
            # -----------------
            valid_total_loss, valid_losses = self._valid_epoch()
            if self.accelerator.is_main_process:
                for key, loss in valid_losses.items():
                    self.logger.info(f"  |- Valid/{key}: {loss:.6f}")
                # ✅ TensorBoard logging 추가
                self.write_valid_summary(valid_losses, {})
            # -----------------
            # Logging
            # -----------------
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss,
                    "Epoch/Valid Loss": valid_total_loss,
                },
                step=self.epoch,
            )

            # -----------------
            # Scheduler Step
            # -----------------
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # -----------------
            # Early Stopping 체크
            # -----------------
            if valid_total_loss < best_val_loss:
                best_val_loss = valid_total_loss
                epochs_no_improve = 0

                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"✅ Validation loss improved to {best_val_loss:.6f}. Saving checkpoint..."
                    )
                    self.current_loss = float(best_val_loss)  # ← 파일명에 val loss 반영
                    self.save_checkpoint()

            else:
                epochs_no_improve += 1
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"⚠️ No improvement in validation loss ({epochs_no_improve}/{patience})"
                    )

            # 조기 종료 조건
            if epochs_no_improve >= patience:
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"🛑 Early stopping triggered after {patience} epochs without improvement."
                    )
                break

            self.epoch += 1

        # -----------------
        # 최종 체크포인트 저장
        # -----------------
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_path = os.path.join(
                self.checkpoint_dir,
                f"final_epoch-{self.epoch:04d}_step-{self.step:07d}",
            )
            self.accelerator.save_state(final_path)
            self.logger.info(f"💾 Final checkpoint saved at {final_path}")

        if self.sw is not None and self.accelerator.is_main_process:
            try:
                self.sw.flush()
                self.sw.close()
            except Exception as e:
                pass

        self.accelerator.end_training()

    def echo_log(self, losses, mode="Training"):
        message = [
            "{} - Epoch {} Step {}: [{:.3f} s/step]".format(
                mode, self.epoch + 1, self.step, self.time_window.average
            )
        ]

        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(str(key) + "=" + str(round(float(losses[key]), 5)))
        self.logger.info(", ".join(message))
    
    def _valid_epoch(self):
        """Validation 루프 (multi-GPU 분산 평균 계산 포함)"""
        if self.valid_dataloader is None:
            return 0.0, {}

        # 각 GPU가 나눠서 검증하도록 sampler 자동 생성
        if hasattr(self.valid_dataloader, "sampler") and hasattr(self.valid_dataloader.sampler, "set_epoch"):
            self.valid_dataloader.sampler.set_epoch(self.epoch)

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = {}
        num_batches = 0
        device = self.accelerator.device

        # ✅ 진입 로그 추가
        if self.accelerator.is_main_process:
            self.logger.info(f"🟡 Starting validation at epoch {self.epoch} "
                            f"({len(self.valid_dataloader)} batches total)")

        with torch.no_grad():
            for i, batch in enumerate(self.valid_dataloader):
                # ✅ 진행률 로그 (100개 단위로)
                if i % 100 == 0 and self.accelerator.is_main_process:
                    self.logger.info(f"[Valid] Processing batch {i}/{len(self.valid_dataloader)}")

                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                total_loss, valid_losses, _ = self._train_step(batch, is_valid=True)

                # 각 GPU별로 loss를 gather해서 평균냄
                gathered_losses = self.accelerator.gather_for_metrics(torch.tensor([total_loss], device=device))
                mean_loss = gathered_losses.mean().item()
                epoch_sum_loss += mean_loss

                for key, value in valid_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    # 개별 loss도 평균 처리
                    gathered = self.accelerator.gather_for_metrics(torch.tensor([value], device=device))
                    epoch_losses[key] += gathered.mean().item()

                num_batches += 1

        # 전체 평균 계산 (GPU별 batch 수 통합)
        num_batches_tensor = torch.tensor([num_batches], device=device)
        gathered_batches = self.accelerator.gather_for_metrics(num_batches_tensor)
        global_num_batches = gathered_batches.sum().item()

        if global_num_batches > 0:
            epoch_sum_loss /= global_num_batches
            for key in epoch_losses.keys():
                epoch_losses[key] /= global_num_batches

        # 모든 프로세스 동기화
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            self.logger.info(f"🟢 Validation complete - avg_loss: {epoch_sum_loss:.6f}")

        return epoch_sum_loss, epoch_losses
