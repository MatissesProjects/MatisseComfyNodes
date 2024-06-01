import torch
from PIL import Image
import numpy as np
import subprocess
import os
import random
import nodes
import folder_paths as comfy_paths
import requests

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

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("last frame #",)

    #OUTPUT_NODE = False
    CATEGORY = "MatisseTec"
    FUNCTION = "getLength"
    def getLength(self, image):
        return (len(image),)


class RGBACombine:
    def __init__(self):
        self.base = comfy_paths.output_directory
        self.count = 0
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "route": ("STRING", {"default": "/api/"}),
                "fileName": ("STRING", {"default": "output"}),
                "appendCount": ("BOOLEAN", {"default": False}),
                "fps": ("FLOAT", {"default": "10"}),
                "width": ("INT", {"default": "200"}),
                "height": ("INT", {"default": "200"}),
                "resetCount": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (None,)

    CATEGORY = "MatisseTec"
    FUNCTION = "combine_video"
    def combine_video(self, image, route, fileName, appendCount, fps, width, height, resetCount):
        rand = random.randint(1e5,9e5)
        self.count += 1
        if resetCount:
            self.count = 0
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
        output_path = f"{self.base}{route}{fileName}{self.count}.gif" if appendCount else \
                      f"{self.base}{route}{fileName}.gif"
        # Save as an animated GIF using FFmpeg
            # "-framerate", f"{fps}",
        subprocess.run([
            "ffmpeg", "-y",
            "-i", f"{self.base}{route}temp/frame_{rand}_%03d.png",
            "-framerate", str(fps),  # Set the input framerate
            "-vf", f"scale={width}:{height}:flags=lanczos,split [o1][o2];[o1] palettegen [p];[o2][p] paletteuse",
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
                "positive": ("STRING", {"multiline": True, "default": "person at the skatepark"}),
                "negative": ("STRING", {"multiline": True, "default": "nsfw, nude"}),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("Positive Conditioning", "Negative Conditioning",)


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
    RETURN_NAMES = ("file content",)


    OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    FUNCTION = "readFile"
    def readFile(self, fileLocation):
        with open(fileLocation, 'r') as f:
            data = f.read()
        return (data,)
   
    @classmethod
    def IS_CHANGED(cls) -> float: return float("nan")

class StringConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "stringA": ("STRING",{"default": ""}),
                "stringB": ("STRING",{"default": ""}),
                "stringC": ("STRING",{"default": ""}),
                "stringD": ("STRING",{"default": ""}),
                "delimiter": ("STRING",{"default": ", "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenate",)

    OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    FUNCTION = "concat"
    def concat(self, stringA="", stringB="", stringC="", stringD="", delimiter=", "):
        # Use a list comprehension to filter out empty strings, allowing for kwargs later
        strings = [s for s in [stringA, stringB, stringC, stringD] if s]
        # Join the non-empty strings with the specified delimiter
        final = delimiter.join(strings)
        
        return (final,)

   
    @classmethod
    def IS_CHANGED(cls) -> float: return float("nan")

class ImageSelector():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "imageToSelect": (['first', 'last', 'random'],{"default": "last"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    FUNCTION = "selectImage"

    def selectImage(self, images, imageToSelect):
        # Select the image index based on the input word
        selector = len(images) - 1 if imageToSelect == 'last' else 0 if imageToSelect == 'first' else random.randint(0, len(images) - 1)
        selected_image = images[selector:selector+1]
        return (selected_image,)

    @classmethod
    def IS_CHANGED(cls) -> float: return float("nan")

class ImageClipper():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "imageToRemove": (['first', 'last'],{"default": "first"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    FUNCTION = "selectImages"

    def selectImages(self, images, imageToRemove):
        if imageToRemove == 'last':
            selected_images = images[:-1]
        else:
            selected_images = images[1:]
        return (selected_images,)

    @classmethod
    def IS_CHANGED(cls) -> float: return float("nan")

class ImageTextGenerator():
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",{"default": "dog park"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image desctiption",)
    # OUTPUT_NODE = True
    CATEGORY = "MatisseTec"
    FUNCTION = "generateImage"

    def generateImage(self, text):
        # TODO add enabled flag that will pass back just the text if false
        return (requests.post("https://deepnarrationapi.matissetec.dev/describeImage", json={"prompt": text}).json()['choices'][0]['message']['content'],)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "list length": LengthNode,
    "rgba video combine": RGBACombine,
    "clip strings": ClipStrings,
    "file reader": FileReader,
    "string concat": StringConcat,
    "image selector": ImageSelector,
    "image list clipper": ImageClipper,
    "image text generator": ImageTextGenerator,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "list length": "List Length 🗃️",
    "rgba video combine": "RGBA Video Combine 🎨📹",
    "clip strings": "clip strings",
    "file reader": "file reader",
    "string concat":"string concat",
    "image selector": "image selector",
    "image list clipper": "image list clipper",
    "image text generator": "image text generator",
}
