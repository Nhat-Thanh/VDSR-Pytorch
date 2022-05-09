from utils.common import *
from model import VDSR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scale',      type=int, default=2,                   help='-')
parser.add_argument("--image-path", type=str, default="dataset/test2.png", help='-')
parser.add_argument("--ckpt-path",  type=str, default="",                  help='-')

FLAGS, unparsed = parser.parse_known_args()
image_path = FLAGS.image_path
ckpt_path = FLAGS.ckpt_path
if ckpt_path == "":
    ckpt_path = "checkpoint/VDSR.pt"

scale = FLAGS.scale
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if scale not in [2, 3, 4]:
    ValueError("must be 2, 3 or 4")


# -----------------------------------------------------------
#  read image and save bicubic image
# -----------------------------------------------------------

lr_image = read_image(image_path)
bicubic_image = upscale(lr_image, scale)
write_image("bicubic.png", bicubic_image)


# -----------------------------------------------------------
# preprocess lr image 
# -----------------------------------------------------------

bicubic_image = rgb2ycbcr(bicubic_image)
bicubic_image = norm01(bicubic_image)
bicubic_image = torch.unsqueeze(bicubic_image, dim=0)


# -----------------------------------------------------------
#  predict and save image
# -----------------------------------------------------------

model = VDSR(device)
model.load_weights(ckpt_path)
with torch.no_grad():
    sr_image = model.predict(bicubic_image.to(device))[0]

sr_image = sr_image.cpu()
sr_image = denorm01(sr_image)
sr_image = sr_image.type(torch.uint8)
sr_image = ycbcr2rgb(sr_image)

write_image("sr.png", sr_image)
