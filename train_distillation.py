import sys
from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT
sys.path.append("../")

import json, os, math

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L

from src.model import EFVC
from src.model_distillation import EFVC_Destilation

from util.dataset.Vimeo90K import Vimeo90K


class DCVC_TCM_Lit(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        
        self.stage = 0
        self.cfg = cfg
        self._parse_cfg()

        self.teacher = EFVC()
        self.student = EFVC_Destilation()
        self.student.apply(self._init_weights)

        ckpt_teacher = torch.load("log/512/model_epoch58_step710747.pth")
        self.teacher.load_state_dict(ckpt_teacher)

        # 加载部分参数
        stu_dict = self.student.state_dict()
        stu_dict.update(ckpt_teacher)
        self.student.load_state_dict(stu_dict)


        self.sum_count = 0
        self.sum_out = {
            "bpp_mv_y": 0,
            "bpp_mv_z": 0,
            "bpp_y": 0,
            "bpp_z": 0,
            "bpp": 0,

            "MSE": 0,
            "PSNR": 0,

            "loss": 0,
        }
        self.stage = 0


    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        loss = 0
        teacher_feature = None
        student_feature = None

        student_reframe = batch[:, 0, ...].to(self.device)
        teacher_reframe = batch[:, 0, ...].to(self.device)

        for i in range(1, T):
            input_frame = batch[:, i,...].to(self.device)

            if self.stage == 0:
                teacher_out = self.teacher(input_frame, teacher_reframe, teacher_feature)
                teacher_reframe = teacher_out["recon_image"]
                teacher_feature = teacher_out["feature"]

            elif self.stage == 1:
                teacher_out = None
            

            out = self.student(input_frame, student_reframe, student_feature)

            # mse & psnr
            dist_recon = F.mse_loss(input_frame, out["recon_image"])
            out["MSE"] = dist_recon

            loss += self._get_loss(teacher_out, out, self.train_lambda)

            # take recon image as ref image
            student_reframe = out["recon_image"]
            student_feature = out["feature"]

            # log

            self.sum_count += 1
            self.sum_out["bpp_mv_y"] += out["bpp_mv_y"].item()
            self.sum_out["bpp_mv_z"] += out["bpp_mv_z"].item()
            self.sum_out["bpp_y"]    += out["bpp_y"].item()
            self.sum_out["bpp_z"]    += out["bpp_z"].item()
            self.sum_out["bpp"]      += out["bpp"].item()

            self.sum_out["MSE"]      += out["MSE"].item()
            self.sum_out["PSNR"]     += mse2psnr(out["MSE".item()])

            self.sum_out["loss"]     += loss.item()

            # print(self.sum_out)


        # average loss
        loss = loss / (T - 1)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        
        # log
        if self.global_step % 50 == 0:
            for key in self.sum_out.keys():
                self.sum_out[key] /= self.sum_count

            self.sum_out["stage"] = float(self.stage)
            self.sum_out["lr"]    = self.optimizers().optimizer.state_dict()['param_groups'][0]['lr']

            self.log_dict(self.sum_out)

            for key in self.sum_out.keys():
                self.sum_out[key] = 0

            self.sum_count = 0

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr = self.base_lr)

        if self.multi_frame_training:
            milestones = [3, 6, 9, 12]
        
        else:
            if self.stage == 0:
                milestones = [1, 2, 3, 4]
            
            else:
                milestones = [1, 2, 3, 4]

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = opt,
            milestones = milestones,
            gamma = 0.5
        )

        return [opt], [scheduler]


    def on_train_epoch_start(self) -> None:
        self.training_stage()


    def training_stage(self):
        self.stage = 0

        if self.stage == 0:
            self._freeze_teacher()
            self._freeze_student()

        elif self.stage == 1:
            self._unfreeze_student()
            if self.teacher is not None:
                del self.teacher
                self.teacher = None


    def _get_loss(self, teacher_out, student_out, train_lambda):
        teacher_context = teacher_out["context"]
        student_context = student_out["context"]


        if self.stage == 0:
            loss = 0
            for i in range(3):
                loss += F.mse_loss(teacher_context[i], student_context[i])
        
        elif self.stage == 1:
            dist = student_out["MSE"]
            bpp = student_out["bpp"]
            loss = self.train_lambda * dist + bpp


        return loss
            

    def _freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad = False

    def _freeze_student(self):
        for p in self.student.parameters():
            p.requires_grad = False
        
        for p in self.student.warpper.parameters():
            p.requires_grad = True

    def _unfreeze_student(self):
        for p in self.student.parameters():
            p.requires_grad = True

    
    def _parse_cfg(self):
        print(self.cfg)

        self.stage_milestones = self.cfg["training"]["stage_milestones"]
        self.base_lr = self.cfg["training"]["base_lr"]
        self.aux_lr = self.cfg["training"]["aux_lr"]
        self.flow_pretrain_dir = self.cfg["training"]["flow_pretrain_dir"]
        self.train_lambda = self.cfg["training"]["train_lambda"]
        self.multi_frame_training = self.cfg["training"]["multi_frame_training"]

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))
    

if __name__ == "__main__":
    L.seed_everything(3407)

    with open("config_distillation.json") as f:
        config = json.load(f)

    model_module = DCVC_TCM_Lit(config)


    if config["training"]["multi_frame_training"]:
        frame_num = 5
        interval = 1
        batch_size = config["training"]["batch_size"] // 2
    
    else:
        frame_num = 2
        interval = 2
        batch_size = config["training"]["batch_size"]

    train_dataset = Vimeo90K(
        root = config["datasets"]["viemo90k"]["root"], 
        split_file= config["datasets"]["viemo90k"]["split_file"],
        frame_num = frame_num, interval = interval, rnd_frame_group = True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, 
        num_workers = 4, persistent_workers=True, pin_memory = True
    )

    trainer = L.Trainer(
        # devices=2, strategy="ddp_find_unused_parameters_true",
        max_epochs = 60,
        fast_dev_run = True,
    )

    trainer.fit(
        model = model_module,
        train_dataloaders = train_dataloader,
        # ckpt_path = "lightning_logs/version_1/checkpoints/epoch=4-step=40385.ckpt"
    )