import sys, yaml, os, math
sys.path.append("../")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from src.model import EFVC

from util.dataset.Vimeo90K import Vimeo90K


class DCVC_TCM_Lit(L.LightningModule):
    WEIGHT = [0.7, 0.9, 1.1, 1.3]

    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        
        self.stage = 0
        self.cfg = cfg
        self._parse_cfg()

        self.model = EFVC()
        self.model.apply(self._init_weights)
        self.model.optic_flow._load_Spynet(self.flow_pretrain_dir)

        self.mv_modules: list[nn.Module] = [
            self.model.optic_flow,
            self.model.mv_encoder,
            self.model.mv_prior_encoder,

            self.model.bit_estimator_z_mv,

            self.model.mv_prior_decoder,
            self.model.mv_decoder
        ]

        self.sum_count = 0
        self.sum_out = {
            "bpp_mv_y": 0,
            "bpp_mv_z": 0,
            "bpp_y": 0,
            "bpp_z": 0,
            "bpp": 0,

            "ME_MSE": 0,
            "ME_PSNR": 0,

            "MSE": 0,
            "PSNR": 0,

            "loss": 0,
        }

    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        loss = 0
        feature = None
        ref_frame = batch[:, 0, ...].to(self.device)

        for i in range(1, T):
            input_frame = batch[:, i,...].to(self.device)
            out = self.model(input_frame, ref_frame, feature)

            loss += self._get_loss(input_frame, out, self.train_lambda, i - 1)

            # take recon image as ref image
            ref_frame = out["recon_image"]
            feature = out["feature"]


            # log
            self.sum_count += 1
            self.sum_out["bpp_mv_y"] += out["bpp_mv_y"].item()
            self.sum_out["bpp_mv_z"] += out["bpp_mv_z"].item()
            self.sum_out["bpp_y"]    += out["bpp_y"].item()
            self.sum_out["bpp_z"]    += out["bpp_z"].item()
            self.sum_out["bpp"]      += out["bpp"].item()

            self.sum_out["MSE"]      += out["MSE"]
            self.sum_out["PSNR"]     += out["PSNR"]

            self.sum_out["ME_MSE"]   += out["ME_MSE"]
            self.sum_out["ME_PSNR"]  += out["ME_PSNR"]
            self.sum_out["loss"]     += loss.item()

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


    def _get_loss(self, input, output, frame_lambda, frame_idx):
        dist_me = F.mse_loss(input, output["warpped_image"])
        dist_recon = F.mse_loss(input, output["recon_image"])

        output["MSE"] = dist_recon.item()
        output["PSNR"] = mse2psnr(output["MSE"])

        output["ME_MSE"] = dist_me.item()
        output["ME_PSNR"] = mse2psnr(output["ME_MSE"])

        if self.stage == 0:
            dist = dist_me
            rate = output["bpp_mv_y"] + output["bpp_mv_z"]
        
        elif self.stage == 1:
            dist = dist_recon
            rate = 0

        elif self.stage == 2:
            dist = dist_recon
            rate = output["bpp_y"] + output["bpp_z"]

        else:
            dist = dist_recon
            rate = output["bpp"]

        if self.multi_frame_training:
            return self.WEIGHT[frame_idx] * frame_lambda * dist + rate
        
        else:
            return frame_lambda * dist + rate


    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr = self.base_lr)

        if self.multi_frame_training:
            milestones = self.lr_milestones_multi
        
        else:
            milestones = self.lr_milestones


        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer = opt,
            milestones = milestones,
            gamma = self.lr_gamma
        )

        return [opt], [scheduler]

    def on_train_start(self) -> None:
        # lr_scheduler = self.lr_schedulers()
        # for _ in range(10):
        #     lr_scheduler.step()
        
        print("Hack lr", self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
            
    
    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()


    def on_train_epoch_start(self):
        self._training_stage()

        # save last epcoh
        if self.current_epoch in self.stage_milestones:
            self._save_model(
                name = f"model_milestone_stage{self.stage}_epoch{self.current_epoch - 1}.pth",
                folder = "log/model_ckpt"
            )

        if self.stage == 3:
            self._save_model(
                name = f"model_epoch{self.current_epoch - 1}_step{self.global_step}.pth", 
                folder = "log/model_ckpt"
            )


    def _training_stage(self):
        if self.multi_frame_training:
            self._train_all()
            self.stage = 3

            print("train all, stage 3")
            return

    
        self.stage = 0
        for step in self.stage_milestones:
            if self.current_epoch < step:
                break
            else:
                self.stage += 1

        if self.stage == 0:
            self._train_mv()

        elif self.stage == 1:
            self._freeze_mv()

        elif self.stage == 2:
            self._freeze_mv()

        elif self.stage == 3:
            self._train_all()


    def _parse_cfg(self):
        print(self.cfg)

        self.stage_milestones = self.cfg["training"]["stage_milestones"]
        self.base_lr = self.cfg["training"]["base_lr"]

        self.flow_pretrain_dir = self.cfg["training"]["flow_pretrain_dir"]

        self.train_lambda = self.cfg["training"]["train_lambda"]
        self.multi_frame_training = self.cfg["training"]["multi_frame_training"]

        self.lr_milestones_multi = self.cfg["training"]["lr_milestones_multi"]
        self.lr_milestones = self.cfg["training"]["lr_milestones"]
        self.lr_gamma = self.cfg["training"]["lr_gamma"]


    def _freeze_mv(self):
        self._train_all()

        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = False

    def _train_mv(self):
        for p in self.model.parameters():
            p.requires_grad = False

        for m in self.mv_modules:
            for p in m.parameters():
                p.requires_grad = True

    def _train_all(self):
        for p in self.model.parameters():
            p.requires_grad = True


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _save_model(self, folder = "log/model_ckpt", name = None):
        if not os.path.exists(folder):
            os.makedirs(folder)

        if name == None:
            name = "model_ep{}_st{}.pth".format(self.current_epoch, self.global_step)

        torch.save(
            self.model.state_dict(), 
            os.path.join(folder, name)
        )


def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))
    

if __name__ == "__main__":
    L.seed_everything(3407)

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model_module = DCVC_TCM_Lit(config)
    model_module.model.load_state_dict(torch.load("log/model_ckpt/model_epoch48_step395773.pth"))

    if config["training"]["multi_frame_training"]:
        frame_num = 5
        interval = 1
        batch_size = config["training"]["batch_size"] // 2
    
    else:
        frame_num = 2
        interval = 2
        batch_size = config["training"]["batch_size"]

    train_dataset = Vimeo90K(
        root = config["datasets"]["vimeo90k"]["root"], 
        split_file= config["datasets"]["vimeo90k"]["split_file"],
        frame_num = frame_num, interval = interval, rnd_frame_group = True
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True, 
        num_workers = 8, persistent_workers=True, pin_memory = True
    )

    logger = TensorBoardLogger(save_dir = "log", name = config["name"])
    trainer = L.Trainer(
        strategy = "ddp_find_unused_parameters_true",
        max_epochs = 70,
        logger = logger,
        # fast_dev_run = True,
    )

    if config["training"]["resume"]:
        trainer.fit(
            model = model_module,
            train_dataloaders = train_dataloader,
            ckpt_path = config["training"]["ckpt"]
        )

    else:
        trainer.fit(
            model = model_module,
            train_dataloaders = train_dataloader,
        )

