import os
import piq
import torch
import torchvision
from iqa import IQA
from loss import TVLoss, PerceptualLoss, ColorConstancyLoss
from model import UnetTMO, AttentionUnetTMO
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


def save_image(im, p):
    base_dir = os.path.split(p)[0]
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torchvision.utils.save_image(im, p)


@MODEL_REGISTRY
class PSENetV2(LightningModule):
    """
    Enhanced PSENet with the following modifications:
    1. Attention mechanism in the enhancement network
    2. Perceptual loss for better texture preservation
    3. Color constancy loss for natural color reproduction
    4. Adaptive fusion strategy for pseudo ground-truth generation
    5. Multi-scale enhancement for better detail preservation
    """
    def __init__(
        self, 
        tv_w=1.0, 
        perceptual_w=0.5,
        color_w=0.3,
        gamma_lower=-1.0, 
        gamma_upper=1.0, 
        number_refs=4, 
        lr=1e-4,
        use_attention=True,
        use_adaptive_fusion=True,
        afifi_evaluation=False
    ):
        super().__init__()
        self.tv_w = tv_w
        self.perceptual_w = perceptual_w
        self.color_w = color_w
        self.gamma_lower = gamma_lower
        self.gamma_upper = gamma_upper
        self.number_refs = number_refs
        self.afifi_evaluation = afifi_evaluation
        self.lr = lr
        self.use_attention = use_attention
        self.use_adaptive_fusion = use_adaptive_fusion
        
        # Model selection based on attention flag
        if use_attention:
            self.model = AttentionUnetTMO()
        else:
            self.model = UnetTMO()
        
        # Loss functions
        self.mse = torch.nn.MSELoss()
        self.tv = TVLoss()
        self.perceptual_loss = PerceptualLoss()
        self.color_loss = ColorConstancyLoss()
        self.iqa = IQA()
        
        self.saved_input = None
        self.saved_pseudo_gt = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            betas=[0.9, 0.999],
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2, 
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

    def generate_pseudo_gt(self, im):
        """
        Enhanced pseudo ground-truth generation with adaptive fusion
        """
        bs, ch, h, w = im.shape
        
        # Generate exposure variations (same as original)
        underexposed_ranges = torch.linspace(0, self.gamma_upper, steps=self.number_refs + 1).to(im.device)[:-1]
        step_size = self.gamma_upper / self.number_refs
        underexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * step_size + underexposed_ranges[None, :]
        )
        
        overexposed_ranges = torch.linspace(self.gamma_lower, 0, steps=self.number_refs + 1).to(im.device)[:-1]
        step_size = -self.gamma_lower / self.number_refs
        overexposed_gamma = torch.exp(
            torch.rand([bs, self.number_refs], device=im.device) * step_size + overexposed_ranges[None, :]
        )
        
        gammas = torch.cat([underexposed_gamma, overexposed_gamma], dim=1)
        synthetic_references = 1 - (1 - im[:, None]) ** gammas[:, :, None, None, None]
        
        # Add previous iteration output
        with torch.no_grad():
            previous_iter_output = self.model(im)[0].clone()
        
        references = torch.cat([im[:, None], previous_iter_output[:, None], synthetic_references], dim=1)
        nref = references.shape[1]
        
        if self.use_adaptive_fusion:
            # Adaptive fusion using quality scores and spatial attention
            scores = self.iqa(references.view(bs * nref, ch, h, w))
            scores = scores.view(bs, nref, 1, h, w)
            
            # Normalize scores to get weights
            weights = torch.softmax(scores * 10, dim=1)  # Temperature scaling
            
            # Weighted fusion instead of hard selection
            pseudo_gt = torch.sum(references * weights, dim=1)
        else:
            # Original hard selection
            scores = self.iqa(references.view(bs * nref, ch, h, w))
            scores = scores.view(bs, nref, 1, h, w)
            max_idx = torch.argmax(scores, dim=1)
            max_idx = max_idx.repeat(1, ch, 1, 1)[:, None]
            pseudo_gt = torch.gather(references, 1, max_idx).squeeze(1)
        
        return pseudo_gt

    def training_step(self, batch, batch_idx):
        """
        Enhanced training step with additional losses
        """
        nth_input = batch
        nth_pseudo_gt = self.generate_pseudo_gt(batch)
        
        if self.saved_input is not None:
            im = self.saved_input
            pred_im, pred_gamma = self.model(im)
            pseudo_gt = self.saved_pseudo_gt
            
            # Reconstruction loss (MSE)
            reconstruction_loss = self.mse(pred_im, pseudo_gt)
            
            # Total Variation loss (smoothness)
            tv_loss = self.tv(pred_gamma)
            
            # Perceptual loss (texture preservation)
            perceptual = self.perceptual_loss(pred_im, pseudo_gt)
            
            # Color constancy loss (natural colors)
            color_constancy = self.color_loss(pred_im, im)
            
            # Combined loss
            loss = (
                reconstruction_loss + 
                tv_loss * self.tv_w + 
                perceptual * self.perceptual_w +
                color_constancy * self.color_w
            )

            # Logging
            self.log_dict({
                "train_loss/reconstruction": reconstruction_loss,
                "train_loss/tv": tv_loss,
                "train_loss/perceptual": perceptual,
                "train_loss/color_constancy": color_constancy,
                "total_loss": loss
            }, on_epoch=True, on_step=False, prog_bar=True)
            
            if batch_idx == 0:
                visuals = [im, pseudo_gt, pred_im]
                visuals = torchvision.utils.make_grid([v[0] for v in visuals], nrow=3)
                self.logger.experiment.add_image("images", visuals, self.current_epoch)
        else:
            loss = None
            self.log("total_loss", 0, on_epoch=True, on_step=False)
        
        self.saved_input = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            pred_im, pred_gamma = self.model(batch)
            self.logger.experiment.add_images("val_input", batch[:4], self.current_epoch)
            self.logger.experiment.add_images("val_output", pred_im[:4], self.current_epoch)

    def test_step(self, batch, batch_idx, test_idx=0):
        input_im, path = batch[0], batch[-1]
        pred_im, pred_gamma = self.model(input_im)
        
        for i in range(len(path)):
            save_image(pred_im[i], os.path.join(self.logger.log_dir, path[i]))

        if len(batch) == 3:
            gt = batch[1]
            psnr = piq.psnr(pred_im, gt)
            ssim = piq.ssim(pred_im, gt)
            
            self.log("psnr", psnr, on_step=False, on_epoch=True)
            self.log("ssim", ssim, on_step=False, on_epoch=True)
            
            if self.afifi_evaluation:
                assert len(path) == 1, "only support with batch size 1"
                if "N1." in path[0]:
                    self.log("psnr_under", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_under", ssim, on_step=False, on_epoch=True)
                else:
                    self.log("psnr_over", psnr, on_step=False, on_epoch=True)
                    self.log("ssim_over", ssim, on_step=False, on_epoch=True)