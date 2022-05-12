from utils.common import *
from model import VDSR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',        type=int,   default=2,                   help='-')
parser.add_argument("--ckpt-path",    type=str,   default="",                  help='-')
parser.add_argument("--image-path",   type=str,   default="dataset/test1.png", help='-')

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path

scale = FLAGS.scale
if scale not in [2, 3, 4]:
    ValueError("must be 2, 3 or 4")

ckpt_path = FLAGS.ckpt_path
if (ckpt_path == "") or (ckpt_path == "default"):
    ckpt_path = f"checkpoint/VDSR-x{scale}.pt"

sigma = 0.3 if scale == 2 else 0.2


# -----------------------------------------------------------
# demo
# -----------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    write_image("bicubic.png", bicubic_image)

    lr_image = gaussian_blur(lr_image, sigma=sigma)
    bicubic_image = upscale(lr_image, scale)
    bicubic_image = rgb2ycbcr(bicubic_image)
    bicubic_image = norm01(bicubic_image)
    bicubic_image = torch.unsqueeze(bicubic_image, dim=0)

    model = VDSR(device)
    model.load_weights(ckpt_path)
    with torch.no_grad():
        bicubic_image = bicubic_image.to(device)
        sr_image = model.predict(bicubic_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = sr_image.type(torch.uint8)
    sr_image = ycbcr2rgb(sr_image)

    write_image("sr.png", sr_image)

if __name__ == "__main__":
    main()

