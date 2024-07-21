import os
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Literal, Optional

from .src.networks import define_G
#import src.c2pGen
from .src.stupid import MLP_code

from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.backend.util.devices import TorchDevice
from invokeai.invocation_api import BaseInvocation, InputField, InvocationContext, WithMetadata, invocation

GA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/160_net_G_A.pth")
ALIAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/alias_net.pth")

def rescale(image, Rescale=True):
    if not Rescale:
        return image
    if Rescale:
        width, height = image.size
        while width > 4000 or height > 4000:
            image = image.resize((int(width // 2), int(height // 2)), Image.BICUBIC)
            width, height = image.size
        while width < 128 or height < 128:
            image = image.resize((int(width * 2), int(height * 2)), Image.BICUBIC)
            width, height = image.size
        return image

class Model():
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.G_A_net = None
        self.alias_net = None
        self.ref_t = None
        self.cell_size_code = None

    def load(self):
        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])

            G_A_state = torch.load(GA_PATH, map_location=str(self.device))
            for p in list(G_A_state.keys()):
                G_A_state["module."+str(p)] = G_A_state.pop(p)
            self.G_A_net.load_state_dict(G_A_state)

            alias_state = torch.load(ALIAS_PATH, map_location=str(self.device))
            for p in list(alias_state.keys()):
                alias_state["module."+str(p)] = alias_state.pop(p)
            self.alias_net.load_state_dict(alias_state)

            code = torch.tensor(MLP_code, device=self.device).reshape((1, 256, 1, 1))
            self.cell_size_code = self.G_A_net.module.MLP(code)

    def pixelize(self, in_img, cell_size):
        with torch.no_grad():
            in_img = in_img.convert('RGB')
            in_img = rescale(in_img)
            width, height = in_img.size
            cell_size = cell_size
            best_cell_size = 4
            in_img = in_img.resize(((width // cell_size) * best_cell_size, (height // cell_size) * best_cell_size),
                               Image.BICUBIC)
            in_t = process(in_img).to(self.device)

            feature = self.G_A_net.module.RGBEnc(in_t)
            images = self.G_A_net.module.RGBDec(feature, self.cell_size_code)
            out_t = self.alias_net(images)
            return save(out_t, cell_size, best_cell_size)


def process(img):
    ow,oh = img.size

    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)

    left = (ow - nw)//2
    top = (oh - nh)//2
    right = left + nw
    bottom = top + nh

    img = img.crop((left, top, right, bottom))

    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trans(img)[None, :, :, :]

def save(tensor, cell_size, best_cell_size=4):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.resize((img.size[0]//best_cell_size, img.size[1]//best_cell_size), Image.NEAREST)
    img = img.resize((img.size[0]*cell_size, img.size[1]*cell_size), Image.NEAREST)
    #img.save(file)
    return img


@invocation(
    "pixelize",
    title="Pixelize Image",
    tags=["image", "retro", "pixel", "pixel art"],
    category="image",
    version="1.0.0",
)
class PixelizeImageInvocation(BaseInvocation, WithMetadata):
    """Creates 'pixel' 'art' using trained models"""

    image: ImageField = InputField(description="The image to pixelize")
    cell_size: int    = InputField(default=4, ge=2, le=32, 
                                   description="pixel/cell size (min 2 max WHATEVER BRO)")



    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)


        m = Model()
        m.load()
        out_image = m.pixelize(image, self.cell_size)

        del m

        image_dto = context.images.save(image=out_image)

        return ImageOutput.build(image_dto)