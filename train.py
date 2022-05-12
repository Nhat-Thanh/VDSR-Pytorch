import argparse
from model import VDSR
import os
import torch
from torch.optim import Adam
from utils.dataset import dataset
from utils.common import PSNR

parser = argparse.ArgumentParser()
parser.add_argument("--epochs",         type=int, default=80, help='-')
parser.add_argument("--batch-size",     type=int, default=64, help='-')
parser.add_argument("--save-best-only", type=int, default=0,  help='-')
parser.add_argument("--save-log",       type=int, default=0,  help='-')
parser.add_argument("--ckpt-dir",       type=str, default="", help='-')


# -----------------------------------------------------------
# global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
epochs = FLAGS.epochs
batch_size = FLAGS.batch_size
save_log = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)

ckpt_dir = FLAGS.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = "checkpoint"

model_path = os.path.join(ckpt_dir, "VDSR.pt")
ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")


# -----------------------------------------------------------
#  Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"
crop_size = 41

train_set = dataset(dataset_dir, "train")
train_set.generate(crop_size, transform=True)
train_set.load_data()

valid_set = dataset(dataset_dir, "validation")
valid_set.generate(crop_size)
valid_set.load_data()


# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vdsr = VDSR(device)

# -----------------------------------------------------------
#  Train
# -----------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    srcnn = VDSR(device)

    optimizer = Adam(vdsr.model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(vdsr.model.parameters(),
    #                             lr=0.01, momentum=0.9, weight_decay=0.0001)
    srcnn.setup(optimizer=optimizer,
                loss=torch.nn.MSELoss(),
                model_path=model_path,
                ckpt_path=ckpt_path,
                metric=PSNR)

    srcnn.load_checkpoint(ckpt_path)
    srcnn.train(train_set, valid_set, epochs=epochs, batch_size=batch_size,
                save_best_only=save_best_only, save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()