import argparse
from model import VDSR
from utils.common import *

parser = argparse.ArgumentParser()
parser.add_argument('--scale',     type=int, default=2,  help='-')
parser.add_argument("--ckpt-path", type=str, default="", help='-')

FLAGS, unparsed = parser.parse_known_args()

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    raise ValueError("must be 2, 3 or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = "checkpoint/VDSR.pt"

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VDSR(device)
    model.load_weights(ckpt_path)

    ls_data = sorted_list(f"dataset/test/x{scale}/data")
    ls_labels = sorted_list(f"dataset/test/x{scale}/labels")

    sum_psnr = 0
    with torch.no_grad():
        for i in range(0, len(ls_data)):
            lr_image = read_image(ls_data[i])
            lr_image = gaussian_blur(lr_image, sigma=sigma)
            bicubic_image = upscale(lr_image, scale)
            hr_image = read_image(ls_labels[i])

            bicubic_image = rgb2ycbcr(bicubic_image)
            hr_image = rgb2ycbcr(hr_image)

            bicubic_image = norm01(bicubic_image)
            hr_image = norm01(hr_image)

            bicubic_image = torch.unsqueeze(bicubic_image, dim=0).to(device)
            sr_image = model.predict(bicubic_image)[0].cpu()

            sum_psnr += PSNR(hr_image, sr_image, max_val=1)

    print(sum_psnr.numpy() / len(ls_data))

if __name__ == "__main__":
    main()