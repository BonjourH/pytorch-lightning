# main.py
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision as tv
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor
from lightning.pytorch.profilers import AdvancedProfiler

# --------------------------------
# Step 1: Define a LightningModule
# --------------------------------

class LitAutoEncoder(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
        
        #################### Save hyperparameters ####################
        # self.save_hyperparameters()
        # call this to save only (layer_1_dim=128) to the checkpoint
        # self.save_hyperparameters("layer_1_dim")

        # equivalent
        # self.save_hyperparameters(ignore=["loss_fx", "generator_network"])

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def _shared_eval(self, batch, batch_idx, prefix):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{prefix}_loss", loss)
        return loss

# Initialize WandbLogger
logger = WandbLogger(project="mnist_autoencoder", name="mnist_autoencoder")

# Initialize checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min'
)

# -------------------
# Step 2: Define data
# -------------------
torch.set_float32_matmul_precision('medium')
dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())
train, val = data.random_split(dataset, [55000, 5000])
train_loader = data.DataLoader(train, batch_size=32, num_workers=19)
val_loader = data.DataLoader(val, batch_size=32, num_workers=19)

# -------------------
# Step 3: Train
# -------------------
profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
autoencoder = LitAutoEncoder()
trainer = L.Trainer(logger=logger, 
                    callbacks=[checkpoint_callback], 
                    max_epochs=5, 
                    # profiler=profiler,
                    profiler='simple',
                    num_sanity_val_steps=0)
trainer.fit(autoencoder, train_loader, val_loader)

# To visualize logs and checkpoints with Wandb, run the following in your terminal:
# wandb login
# wandb sync --include-offline
