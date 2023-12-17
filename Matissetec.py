import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import numpy as np
import subprocess
import os
import random
import nodes

import folder_paths as comfy_paths

# wildcard trick is 100% stolen from pythongossss's
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

WILDCARD = AnyType("*")

class LengthNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("last frame #",)

    FUNCTION = "getLength"
    #OUTPUT_NODE = False
    CATEGORY = "MatisseTec"
    def getLength(self, image):
        return (str(len(image)),)


class RGBAVideoCombine:
    def __init__(self):
        self.base = comfy_paths.output_directory
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "route": ("STRING", {"default": "/api/"}),
                "fps": ("INT", {"default": "10"}),
                "width": ("INT", {"default": "200"}),
                "height": ("INT", {"default": "200"}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (None,)

    FUNCTION = "combine_video"
    CATEGORY = "MatisseTec"
    def combine_video(self, image, route, fps, width, height):
        rand = random.randint(1e5,9e5)
        print(f"in combine with {len(image)} images")
        converted_images = []
        os.makedirs(f"{self.base}{route}temp", exist_ok=True)
        for i, img in enumerate(image):
            if isinstance(img, torch.Tensor):
                img = self.tensor_to_pil(img)

            img = img.convert("RGBA")
            alpha = img.split()[3]
            # mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)
            img_p = Image.new("RGBA", img.size)
            img_p.paste(img, mask=alpha)
            
            frame_filename = f"{self.base}{route}temp/frame_{rand}_{i:03d}.png"
            img_p.save(frame_filename, format="PNG")
            converted_images.append(img_p)
        output_path = f"{self.base}{route}output.gif"
        # Save as an animated GIF using FFmpeg
            # "-framerate", f"{fps}",
        subprocess.run([
            "ffmpeg", "-y",
            "-i", f"{self.base}{route}temp/frame_{rand}_%03d.png",
            "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos,split [o1][o2];[o1] palettegen [p];[o2][p] paletteuse",
            "-loop", "0",
            output_path
        ])

        print("GIF created successfully with FFmpeg.")
        return (float('nan'),)

    def tensor_to_pil(self, tensor):
        # Ensure tensor is in CPU and in the right format
        if tensor.is_cuda:
            tensor = tensor.cpu()
        # If tensor is in the format (Channels, Height, Width), permute to (Height, Width, Channels)
        if tensor.dim() == 3 and tensor.shape[0] < tensor.shape[1] and tensor.shape[0] < tensor.shape[2]:
            tensor = tensor.permute(1, 2, 0)
        # Convert to numpy array
        numpy_array = tensor.numpy()
        # Ensure pixel values are in the range [0, 255]
        if np.issubdtype(numpy_array.dtype, np.floating):
            numpy_array = (numpy_array * 255).astype(np.uint8)
        # Convert numpy array to PIL Image
        image = Image.fromarray(numpy_array)
        return image

class ClipStrings:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"default": "person at the skate park"}),
                "negative": ("STRING", {"default": "nsfw, nude"}),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("Positive Conditioning", "Negative Conditioning")


    FUNCTION = "getConditioning"
    #OUTPUT_NODE = False
    CATEGORY = "MatisseTec"
    def getConditioning(self, positive, negative, clip):
        CLIPTextEncode = nodes.CLIPTextEncode()
        encodedPositive = CLIPTextEncode.encode(clip=clip, text=positive)
        encodedNegative = CLIPTextEncode.encode(clip=clip, text=negative)
        return (encodedPositive[0], encodedNegative[0],)


class FileReader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "fileLocation": ("STRING",{"default": "/example.txt"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file content")


    FUNCTION = "readFile"
    OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    def readFile(self, fileLocation):
        with open(fileLocation, 'r') as f:
            data = f.read()
        return (data,)
   
    @classmethod
    def IS_CHANGED(cls) -> float: return float("nan")


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "list length": LengthNode,
    "rgba video combine": RGBAVideoCombine,
    "clip strings": ClipStrings,
    "file reader": FileReader
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "list length": "List Length 🗃️",
    "rgba video combine": "RGBA Video Combine 🎨📹",
    "clip strings": "clip strings",
    "file reader": "file reader"
}
