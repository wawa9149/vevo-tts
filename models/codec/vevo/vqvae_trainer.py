# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import yaml
import torchaudio

from models.vc.base.vc_emilia_dataset import VCEmiliaDataset, VCCollator
from models.codec.kmeans.repcodec_model import RepCodec
from models.codec.vevo.vevo_repcodec import VevoRepCodec

from models.base.base_trainer import BaseTrainer


class VQVAETrainer(BaseTrainer):
    def __init__(self, args, cfg):
        super(VQVAETrainer, self).__init__(args, cfg)

        # setup input model (such as SSL features)
        self._build_input_model()

    def _build_model(self):
        if getattr(self.cfg.model.repcodec, "repcodec_type", "amphion") == "amphion":
            model = RepCodec(cfg=self.cfg.model.repcodec)
        else:
            assert self.cfg.model.repcodec.repcodec_type == "vevo"
            with open(self.cfg.model.repcodec.config_path) as fp:
                conf = yaml.load(fp, Loader=yaml.FullLoader)
            model = VevoRepCodec(**conf)
        return model

    @torch.no_grad()
    def _extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
        """
        with torch.no_grad():
            feats, _ = self.hubert.extract_features(
                wavs, lengths=wav_lens, num_layers=output_layer
            )
            feats = feats[-1].squeeze()
        return feats

    def _build_input_model(self):
        # HuBERT Model
        if self.cfg.model.representation_type == "hubert":
            bundle = torchaudio.pipelines.HUBERT_LARGE
            self.hubert = bundle.get_model()
            self.hubert.eval()
            self.hubert.to(self.accelerator.device)

            if getattr(self.cfg.model, "use_norm_feat", False):
                stat = np.load(self.cfg.model.representation_stat_mean_var_path)
                self.feat_norm_mean = torch.tensor(stat["mean"]).to(
                    self.accelerator.device
                )
                self.feat_norm_std = torch.tensor(stat["std"]).to(
                    self.accelerator.device
                )

    def _build_dataset(self):
        return VCEmiliaDataset, VCCollator

    def _train_step(self, batch, is_valid=False):
        """한 step (train 또는 validation) 실행"""
        train_losses = {}
        total_loss = 0.0
        train_stats = {}

        speech = batch["wav"]
        feat = None

        # 🔸 HuBERT feature 추출
        if self.cfg.model.representation_type == "hubert":
            sr = self.cfg.model.representation_sample_rate
            wavs = batch[f"wav_{sr}"]
            wav_lens = batch[f"wav_{sr}_len"]
            feat = self._extract_hubert_feature(wavs, wav_lens)

        # 🔸 Gaussian normalization
        if getattr(self.cfg.model, "use_norm_feat", False):
            feat = (feat - self.feat_norm_mean.to(feat)) / self.feat_norm_std.to(feat)

        if feat is None:
            raise ValueError("❌ feat is None — check representation_type or feature extraction config.")


        # 🔸 모델 forward
        if getattr(self.cfg.model.repcodec, "repcodec_type", "amphion") == "amphion":
            feat_rec, codebook_loss, _ = self.model(feat)
            perplexity = None
        else:
            feat_rec, _, _, vqloss, perplexity = self.model(feat.transpose(1, 2))
            feat_rec = feat_rec.transpose(1, 2)
            codebook_loss = torch.sum(vqloss)

        rec_loss = torch.nn.functional.l1_loss(feat_rec, feat)
        total_loss = rec_loss * 32 + codebook_loss

        train_losses["rec_loss"] = rec_loss
        train_losses["codebook_loss"] = codebook_loss

        if perplexity is not None:
            self.logger.info(f"  |- perplexity: {perplexity.mean().item():.2f}")

        # 🔹 Validation일 때는 backward 하지 않음
        if not is_valid:
            self.optimizer.zero_grad()
            self.accelerator.backward(total_loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
                )
            self.optimizer.step()
            self.scheduler.step()

        for k in train_losses:
            train_losses[k] = train_losses[k].item()

        self.current_loss = total_loss.item()
        return total_loss.item(), train_losses, train_stats
