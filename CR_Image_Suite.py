import os
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import piexif
import scipy.ndimage
import torch
from PIL import ExifTags, Image, ImageOps, ImageSequence
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile, PngInfo

import folder_paths
from comfy_execution.graph import ExecutionBlocker
from .CR_cropandstitch import (
    NODE_CLASS_MAPPINGS as CROP_STITCH_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CROP_STITCH_NODE_DISPLAY_NAME_MAPPINGS,
)


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".gif", ".tiff", ".webp", ".bmp"}


class CR_Advanced_Load:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_folder": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "METADATA_RAW")
    RETURN_NAMES = ("image", "file_name", "metadata_raw")
    FUNCTION = "load_image"
    CATEGORY = "CR Image Suite/Load"

    def load_image(self, image_folder: str, index: int):
        if not image_folder or not image_folder.strip():
            raise ValueError("Image folder path is empty.")

        image_folder = image_folder.strip()
        if not os.path.isdir(image_folder):
            raise ValueError(f"Folder does not exist: {image_folder}")

        supported_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
        files = [
            f
            for f in os.listdir(image_folder)
            if os.path.splitext(f)[1].lower() in supported_exts and os.path.isfile(os.path.join(image_folder, f))
        ]
        if not files:
            raise ValueError(f"No supported images found in: {image_folder}")

        files.sort(key=lambda x: x.lower())
        if index >= len(files):
            index = index % len(files)

        selected_file = files[index]
        file_path = os.path.join(image_folder, selected_file)
        img = Image.open(file_path)
        metadata = self.get_metadata(file_path, img)
        img = ImageOps.exif_transpose(img)

        frames = []
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert("RGB")
            arr = np.array(frame).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr)[None, ...]
            frames.append(tensor)
        image_out = torch.cat(frames, dim=0) if len(frames) > 1 else frames[0]
        name_without_ext = os.path.splitext(selected_file)[0]
        return (image_out, name_without_ext, metadata)

    def get_metadata(self, image_path, img):
        metadata = {}
        stat = os.stat(image_path)
        metadata["fileinfo"] = {
            "filename": Path(image_path).as_posix(),
            "resolution": f"{img.width}x{img.height}",
            "date": str(datetime.fromtimestamp(stat.st_mtime)),
            "size": str(stat.st_size),
        }

        if img.format == "WEBP":
            try:
                exif_data = piexif.load(image_path)
                self.process_exif_data(exif_data, metadata)
            except Exception:
                pass

        if isinstance(img, PngImageFile):
            metadata_from_img = img.info
            for k, v in metadata_from_img.items():
                if k == "workflow":
                    try:
                        metadata["workflow"] = json.loads(v)
                    except Exception:
                        metadata["workflow"] = v
                elif k == "prompt":
                    try:
                        metadata["prompt"] = json.loads(v)
                    except Exception:
                        metadata["prompt"] = v
                else:
                    try:
                        metadata[str(k)] = json.loads(v)
                    except Exception:
                        metadata[str(k)] = str(v)

        if isinstance(img, JpegImageFile) or img.format in ["JPG", "JPEG"]:
            exif = img.getexif()
            if exif:
                for k, v in exif.items():
                    tag = ExifTags.TAGS.get(k, k)
                    tag_name = str(tag)

                    if isinstance(v, bytes) and tag_name.startswith("XP"):
                        try:
                            metadata[tag_name] = v.decode("utf-16le").replace("\x00", "")
                        except Exception:
                            metadata[tag_name] = str(v)
                    elif isinstance(v, bytes) and tag_name == "UserComment":
                        try:
                            prefix = v[:8]
                            data = v[8:]
                            if prefix.startswith(b"UNICODE"):
                                metadata[tag_name] = data.decode("utf-16le", errors="ignore").replace("\x00", "")
                            elif prefix.startswith(b"ASCII"):
                                metadata[tag_name] = data.decode("ascii", errors="ignore").replace("\x00", "")
                            else:
                                try:
                                    metadata[tag_name] = v.decode("utf-8").replace("\x00", "")
                                except Exception:
                                    metadata[tag_name] = v.decode("utf-16le", errors="ignore").replace("\x00", "")
                        except Exception:
                            metadata[tag_name] = str(v)
                    elif v is not None:
                        metadata[tag_name] = str(v)

                for ifd_id in ExifTags.IFD:
                    try:
                        resolve = ExifTags.GPSTAGS if ifd_id == ExifTags.IFD.GPSInfo else ExifTags.TAGS
                        ifd = exif.get_ifd(ifd_id)
                        ifd_name = str(ifd_id.name)
                        if ifd:
                            metadata[ifd_name] = {}
                            for key, value in ifd.items():
                                tag = resolve.get(key, key)
                                metadata[ifd_name][str(tag)] = str(value)
                    except KeyError:
                        pass

        self.extract_camera_info(metadata)
        return metadata

    def process_exif_data(self, exif_data, metadata):
        if "0th" in exif_data:
            if 271 in exif_data["0th"]:
                prompt_data = exif_data["0th"][271].decode("utf-8")
                prompt_data = prompt_data.replace("Prompt:", "", 1)
                try:
                    metadata["prompt"] = json.loads(prompt_data)
                except json.JSONDecodeError:
                    metadata["prompt"] = prompt_data

            if 270 in exif_data["0th"]:
                workflow_data = exif_data["0th"][270].decode("utf-8")
                workflow_data = workflow_data.replace("Workflow:", "", 1)
                try:
                    metadata["workflow"] = json.loads(workflow_data)
                except json.JSONDecodeError:
                    metadata["workflow"] = workflow_data
        metadata.update(exif_data)

    def extract_camera_info(self, metadata):
        camera = {}

        def get_value(tag, default=None):
            return metadata.get(tag, default)

        make = str(get_value("Make", "")).strip()
        model = str(get_value("Model", "")).strip()
        camera["Camera"] = f"{make} {model}".strip() or "Unknown Camera"

        lens = get_value("LensModel") or get_value("LensType") or get_value("Lens")
        if lens and str(lens).strip():
            camera["Lens"] = str(lens).strip()

        fnum = get_value("FNumber")
        if isinstance(fnum, tuple):
            fnum = fnum[0] / fnum[1] if fnum[1] else 0
        if fnum:
            camera["Aperture"] = f"f/{float(fnum):.1f}".rstrip("0").rstrip(".")

        exp = get_value("ExposureTime")
        if isinstance(exp, tuple):
            exp = exp[0] / exp[1] if exp[1] else 0
        if exp:
            if exp < 1:
                camera["Shutter"] = f"1/{int(round(1 / exp))}s"
            else:
                camera["Shutter"] = f"{exp:.2f}s".rstrip("0").rstrip(".")

        iso = get_value("ISOSpeedRatings") or get_value("ISO")
        if iso:
            camera["ISO"] = str(iso)

        focal = get_value("FocalLength")
        if isinstance(focal, tuple):
            focal = focal[0] / focal[1] if focal[1] else 0
        if focal:
            camera["Focal Length"] = f"{float(focal):.0f}mm"

        camera_info = {k: v for k, v in camera.items() if v and str(v).strip()}
        if camera_info:
            metadata["Camera Info"] = camera_info

    @classmethod
    def IS_CHANGED(cls, image_folder: str, index: int):
        if not image_folder or not os.path.isdir(image_folder):
            return "invalid"
        try:
            files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
            files.sort(key=str.lower)
            hash_val = hashlib.md5("".join(files).encode()).hexdigest()
            mtime = max(os.path.getmtime(os.path.join(image_folder, f)) for f in files)
            return f"{hash_val}_{mtime}_{index}"
        except Exception:
            return "error"


class CR_Image_Crop_By_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_left": ("INT", {"default": 64, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "padding_right": ("INT", {"default": 64, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "padding_top": ("INT", {"default": 64, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "padding_bottom": ("INT", {"default": 64, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "return_list": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE_BOUNDS")
    FUNCTION = "crop_by_mask"
    CATEGORY = "CR Image Suite/Mask"

    def crop_by_mask(
        self,
        image,
        mask,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
        return_list=False,
    ):
        image = image.unsqueeze(0) if image.dim() == 3 else image
        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        use_single_mask = len(mask) != len(image)

        cropped_images = []
        all_bounds = []
        for i in range(len(image)):
            mask_idx = 0 if use_single_mask else i
            current_mask = mask[mask_idx]
            rows = torch.any(current_mask, dim=1)
            cols = torch.any(current_mask, dim=0)

            if not torch.any(rows) or not torch.any(cols):
                channels = image[i].shape[-1]
                fallback = torch.zeros((16, 16, channels), dtype=image.dtype, device=image.device)
                cropped_images.append(fallback)
                all_bounds.append([0, 15, 0, 15])
                continue

            row_idx = torch.where(rows)[0]
            col_idx = torch.where(cols)[0]
            rmin = int(row_idx[0].item())
            rmax = int(row_idx[-1].item())
            cmin = int(col_idx[0].item())
            cmax = int(col_idx[-1].item())

            rmin = max(rmin - padding_top, 0)
            rmax = min(rmax + padding_bottom, current_mask.shape[0] - 1)
            cmin = max(cmin - padding_left, 0)
            cmax = min(cmax + padding_right, current_mask.shape[1] - 1)

            all_bounds.append([rmin, rmax, cmin, cmax])
            cropped_images.append(image[i][rmin : rmax + 1, cmin : cmax + 1, :])

        if return_list:
            return (cropped_images, all_bounds)
        return (torch.stack(cropped_images), all_bounds)


class CR_Image_Save:
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": "", "multiline": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_number_padding": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
                "filename_number_start": (["false", "true"],),
                "extension": (["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp"],),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 2400, "step": 1}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100, "step": 1}),
                "optimize_image": (["true", "false"],),
                "lossless_webp": (["false", "true"],),
                "overwrite_mode": (["false", "prefix_as_filename"],),
                "embed_workflow": (["true", "false"],),
                "show_previews": (["true", "false"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "files")
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "CR Image Suite/IO"

    def save_images(
        self,
        images,
        output_path="",
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        extension="png",
        dpi=300,
        quality=100,
        optimize_image="true",
        lossless_webp="false",
        overwrite_mode="false",
        filename_number_padding=4,
        filename_number_start="false",
        embed_workflow="true",
        show_previews="true",
        prompt=None,
        extra_pnginfo=None,
    ):
        delimiter = filename_delimiter
        number_padding = int(filename_number_padding)
        lossless_webp = lossless_webp == "true"
        optimize_image = optimize_image == "true"

        if output_path in [None, "", "none", "."]:
            output_path = self.output_dir
        elif not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)

        os.makedirs(output_path, exist_ok=True)

        file_extension = "." + extension
        if file_extension not in ALLOWED_EXT:
            raise ValueError(f"Unsupported extension `{extension}`. Allowed: {sorted(ALLOWED_EXT)}")

        counter = 1
        if number_padding > 0:
            if filename_number_start == "true":
                pattern = f"(\\d+){re.escape(delimiter)}{re.escape(filename_prefix)}{re.escape(file_extension)}$"
            else:
                pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d+){re.escape(file_extension)}$"
            existing_counters = [
                int(re.search(pattern, filename).group(1))
                for filename in os.listdir(output_path)
                if re.match(pattern, os.path.basename(filename))
            ]
            if existing_counters:
                counter = max(existing_counters) + 1

        results = []
        output_files = []
        for image in images:
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            if extension == "webp":
                img_exif = img.getexif()
                if embed_workflow == "true":
                    workflow_metadata = ""
                    if prompt is not None:
                        img_exif[0x010F] = "Prompt:" + json.dumps(prompt)
                    if extra_pnginfo is not None:
                        for key in extra_pnginfo:
                            workflow_metadata += json.dumps(extra_pnginfo[key])
                    img_exif[0x010E] = "Workflow:" + workflow_metadata
                exif_data = img_exif.tobytes()
            else:
                metadata = PngInfo()
                if embed_workflow == "true":
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for key in extra_pnginfo:
                            metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                exif_data = metadata

            if overwrite_mode == "prefix_as_filename" or number_padding == 0:
                file = f"{filename_prefix}{file_extension}"
            else:
                if filename_number_start == "true":
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"

                while os.path.exists(os.path.join(output_path, file)):
                    counter += 1
                    if filename_number_start == "true":
                        file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                    else:
                        file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"

            output_file = os.path.abspath(os.path.join(output_path, file))
            if extension in ["jpg", "jpeg"]:
                img.save(output_file, quality=quality, optimize=optimize_image, dpi=(dpi, dpi))
            elif extension == "webp":
                img.save(output_file, quality=quality, lossless=lossless_webp, exif=exif_data)
            elif extension == "png":
                img.save(output_file, pnginfo=exif_data, optimize=optimize_image)
            elif extension == "bmp":
                img.save(output_file)
            elif extension == "tiff":
                img.save(output_file, quality=quality, optimize=optimize_image)
            else:
                img.save(output_file, optimize=optimize_image)

            output_files.append(output_file)
            if show_previews == "true":
                subfolder = self.get_subfolder_path(output_file, self.output_dir)
                results.append({"filename": file, "subfolder": subfolder, "type": self.type})

            if overwrite_mode == "false" and number_padding > 0:
                counter += 1

        if show_previews == "true":
            return {"ui": {"images": results, "files": output_files}, "result": (images, output_files)}
        return {"ui": {"images": []}, "result": (images, output_files)}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts) :]
        return os.sep.join(subfolder_parts[:-1])


class CR_Mask_Combine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            },
            "optional": {
                "mask_c": ("MASK",),
                "mask_d": ("MASK",),
                "mask_e": ("MASK",),
                "mask_f": ("MASK",),
            },
        }

    CATEGORY = "CR Image Suite/Mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "combine_masks"

    def combine_masks(self, mask_a, mask_b, mask_c=None, mask_d=None, mask_e=None, mask_f=None):
        masks = [m for m in [mask_a, mask_b, mask_c, mask_d, mask_e, mask_f] if m is not None]
        valid_masks = [m for m in masks if m.shape != (1, 64, 64)]

        if len(valid_masks) == 0:
            return (mask_a,)
        if len(valid_masks) == 1:
            return (valid_masks[0],)

        device = valid_masks[0].device
        valid_masks = [m.to(device) for m in valid_masks]
        combined_mask = torch.sum(torch.stack(valid_masks, dim=0), dim=0)
        combined_mask = torch.clamp(combined_mask, 0, 1)
        return (combined_mask,)


class CR_Convert_Mask_To_Bounding_Box:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "padding": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "CR Image Suite/Mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "convert_mask_to_bounding_box"

    def convert_mask_to_bounding_box(self, mask, padding):
        is_single_mask = mask.dim() == 2
        masks = mask.unsqueeze(0) if is_single_mask else mask
        output = torch.zeros_like(masks)

        for i in range(len(masks)):
            current_mask = masks[i]
            rows = torch.any(current_mask > 0, dim=1)
            cols = torch.any(current_mask > 0, dim=0)

            if not torch.any(rows) or not torch.any(cols):
                continue

            row_idx = torch.where(rows)[0]
            col_idx = torch.where(cols)[0]
            rmin = max(int(row_idx[0].item()) - padding, 0)
            rmax = min(int(row_idx[-1].item()) + padding, current_mask.shape[0] - 1)
            cmin = max(int(col_idx[0].item()) - padding, 0)
            cmax = min(int(col_idx[-1].item()) + padding, current_mask.shape[1] - 1)

            output[i, rmin : rmax + 1, cmin : cmax + 1] = 1.0

        return (output.squeeze(0) if is_single_mask else output,)


class CR_Grow_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "CR Image Suite/Mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "grow_mask"

    def grow_mask(self, mask, expand, tapered_corners):
        if expand == 0:
            return (mask,)

        mask_batch = mask.unsqueeze(0) if mask.dim() == 2 else mask
        src_device = mask_batch.device
        src_dtype = mask_batch.dtype

        corner = 0 if tapered_corners else 1
        kernel = np.array(
            [
                [corner, 1, corner],
                [1, 1, 1],
                [corner, 1, corner],
            ],
            dtype=np.uint8,
        )

        out_masks = []
        iters = abs(int(expand))
        for m in mask_batch:
            arr = m.detach().to(device="cpu", dtype=torch.float32).numpy()
            for _ in range(iters):
                if expand < 0:
                    arr = scipy.ndimage.grey_erosion(arr, footprint=kernel)
                else:
                    arr = scipy.ndimage.grey_dilation(arr, footprint=kernel)
            out_masks.append(torch.from_numpy(arr))

        out = torch.stack(out_masks, dim=0).to(device=src_device, dtype=src_dtype)
        return (out,)


class CR_Blur_Mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "radius": ("INT", {"default": 4, "min": 0, "max": 1024, "step": 1}),
                "iterations": ("INT", {"default": 1, "min": 0, "max": 128, "step": 1}),
                "clamp_result": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "CR Image Suite/Mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "blur_mask"

    def blur_mask(self, mask, radius, iterations, clamp_result):
        if radius <= 0 or iterations <= 0:
            return (mask,)

        mask_batch = mask.unsqueeze(0) if mask.dim() == 2 else mask
        src_device = mask_batch.device
        src_dtype = mask_batch.dtype

        sigma = float(radius)
        passes = int(iterations)
        out_masks = []
        for m in mask_batch:
            arr = m.detach().to(device="cpu", dtype=torch.float32).numpy()
            for _ in range(passes):
                arr = scipy.ndimage.gaussian_filter(arr, sigma=sigma, mode="nearest")
            if clamp_result:
                arr = np.clip(arr, 0.0, 1.0)
            out_masks.append(torch.from_numpy(arr))

        out = torch.stack(out_masks, dim=0).to(device=src_device, dtype=src_dtype)
        return (out,)


class CR_Muter_Switch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "on_true": ("*", {"lazy": True}),
                "on_false": ("*", {"lazy": True}),
            },
        }

    CATEGORY = "CR Image Suite/Logic"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "muter_switch"

    def check_lazy_status(self, enabled, on_true=None, on_false=None):
        if enabled and on_true is None:
            return ["on_true"]
        if not enabled and on_false is None:
            return ["on_false"]
        return []

    @classmethod
    def VALIDATE_INPUTS(cls, enabled=True, on_true=None, on_false=None, **kwargs):
        if on_true is None and on_false is None:
            return "At least one of on_true or on_false must be connected."
        return True

    def muter_switch(self, enabled, on_true=None, on_false=None):
        if enabled:
            return (on_true if on_true is not None else on_false,)
        return (on_false if on_false is not None else on_true,)


class CR_Execution_Muter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("*",),
                "enabled": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "CR Image Suite/Logic"
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "execution_muter"

    def execution_muter(self, input, enabled, verbose=False):
        if enabled:
            return (input,)
        return (ExecutionBlocker("Blocked Execution" if verbose else None),)


class CR_Image_Mask_Switch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "images1": ("IMAGE",),
            },
            "optional": {
                "select_text": ("STRING", {"default": "", "multiline": False}),
                "mask1_opt": ("MASK",),
                "images2_opt": ("IMAGE",),
                "mask2_opt": ("MASK",),
                "images3_opt": ("IMAGE",),
                "mask3_opt": ("MASK",),
                "images4_opt": ("IMAGE",),
                "mask4_opt": ("MASK",),
            },
        }

    CATEGORY = "CR Image Suite/Mask"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "mask")
    FUNCTION = "image_mask_switch"

    @staticmethod
    def _resolve_select(select, select_text):
        if select_text is not None:
            text = str(select_text).strip().lower()
            if text:
                word_map = {
                    "one": 1,
                    "two": 2,
                    "three": 3,
                    "four": 4,
                }
                if text in word_map:
                    return word_map[text]

                match = re.search(r"-?\d+", text)
                if match is not None:
                    parsed = int(match.group(0))
                    return max(1, min(4, parsed))

        return max(1, min(4, int(select)))

    def image_mask_switch(
        self,
        select,
        images1,
        select_text="",
        mask1_opt=None,
        images2_opt=None,
        mask2_opt=None,
        images3_opt=None,
        mask3_opt=None,
        images4_opt=None,
        mask4_opt=None,
    ):
        select = self._resolve_select(select, select_text)

        if select == 1:
            return (images1, mask1_opt)
        if select == 2:
            return (images2_opt, mask2_opt)
        if select == 3:
            return (images3_opt, mask3_opt)
        return (images4_opt, mask4_opt)


NODE_CLASS_MAPPINGS = {
    "CR Advanced Load": CR_Advanced_Load,
    "CR Convert Mask to Bounding Box": CR_Convert_Mask_To_Bounding_Box,
    "CR Image Crop by Mask": CR_Image_Crop_By_Mask,
    "CR Image Save": CR_Image_Save,
    "CR Mask Combine": CR_Mask_Combine,
    "CR Grow Mask": CR_Grow_Mask,
    "CR Blur Mask": CR_Blur_Mask,
    "CR Muter Switch": CR_Muter_Switch,
    "CR Execution Muter": CR_Execution_Muter,
    "CR Image Mask Switch": CR_Image_Mask_Switch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR Advanced Load": "CR Advanced Load",
    "CR Convert Mask to Bounding Box": "Convert Mask to Bounding Box",
    "CR Image Crop by Mask": "CR Image Crop by Mask",
    "CR Image Save": "CR Image Save",
    "CR Mask Combine": "CR Mask Combine",
    "CR Grow Mask": "CR Grow Mask",
    "CR Blur Mask": "CR Blur Mask",
    "CR Muter Switch": "CR Muter Switch",
    "CR Execution Muter": "CR Execution Muter",
    "CR Image Mask Switch": "CR Image Mask Switch",
}

NODE_CLASS_MAPPINGS.update(CROP_STITCH_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(CROP_STITCH_NODE_DISPLAY_NAME_MAPPINGS)
