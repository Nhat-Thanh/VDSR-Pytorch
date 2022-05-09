from utils.common import *
from model import VDSR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scale",     type=int, default=2,  help='-')
parser.add_argument("--ckpt-path", type=str, default="", help='-')

FLAGS, _ = parser.parse_known_args()

scale = FLAGS.scale
ckpt_path = FLAGS.ckpt_path
if ckpt_path == "":
    ckpt_path = "checkpoint/VDSR.pt"


# -----------------------------------------------------------
# load VDSR model 
# -----------------------------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VDSR(device)
model.load_weights(ckpt_path)


# -----------------------------------------------------------
# init list of images
# -----------------------------------------------------------

ls_data = sorted_list(f"dataset/test/x{scale}/data")
ls_labels = sorted_list(f"dataset/test/x{scale}/labels")


# -----------------------------------------------------------
# preprocess and testing
# -----------------------------------------------------------

sum_psnr = 0
with torch.no_grad():
    for i in range(0, len(ls_data)):
        hr_image = read_image(ls_labels[i])
        lr_image = read_image(ls_data[i])
        bicubic_image = upscale(lr_image, scale)

        hr_image = rgb2ycbcr(hr_image)
        bicubic_image = rgb2ycbcr(bicubic_image)

        hr_image = norm01(hr_image)
        bicubic_image = norm01(bicubic_image)

        bicubic_image = bicubic_image.to(device)
        bicubic_image = torch.unsqueeze(bicubic_image, dim=0)
        sr_image = model.predict(bicubic_image)[0]
        sr_image = sr_image.cpu()

        sum_psnr += PSNR(hr_image, sr_image, max_val=1)

print(sum_psnr / len(ls_data))

