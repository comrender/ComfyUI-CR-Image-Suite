import math
import nodes
import numpy as np
import torch
import torch.nn.functional as TF
import torchvision.transforms.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
from abc import ABC, abstractmethod

class ProcessorLogic(ABC):
    @abstractmethod
    def rescale_i(self, samples, width, height, algorithm: str):
        pass

    @abstractmethod
    def rescale_m(self, samples, width, height, algorithm: str):
        pass

    @abstractmethod
    def fillholes_iterative_hipass_fill_m(self, samples):
        pass

    @abstractmethod
    def hipassfilter_m(self, samples, threshold):
        pass

    @abstractmethod
    def expand_m(self, samples, pixels):
        pass

    @abstractmethod
    def invert_m(self, samples):
        pass

    @abstractmethod
    def blur_m(self, samples, pixels):
        pass

    @abstractmethod
    def debug_context_location_in_image(self, image, x, y, w, h):
        pass

    @abstractmethod
    def pad_to_multiple(self, value, multiple):
        pass

    @abstractmethod
    def preresize_imm(self, image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
        pass

    @abstractmethod
    def extend_imm(self, image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
        pass

    @abstractmethod
    def batched_findcontextarea_m(self, mask):
        pass

    def findcontextarea_m(self, mask):
        # Default implementation for single masks using the batched version
        # mask is [1, H, W]
        _, x, y, w, h = self.batched_findcontextarea_m(mask)
        context = mask[:, y[0]:y[0]+h[0], x[0]:x[0]+w[0]]
        return context, x[0].item(), y[0].item(), w[0].item(), h[0].item()

    @abstractmethod
    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        pass

    def growcontextarea_m(self, context, mask, x, y, w, h, extend_factor):
        _, nx, ny, nw, nh = self.batched_growcontextarea_m(mask, torch.tensor([x], device=mask.device), torch.tensor([y], device=mask.device), torch.tensor([w], device=mask.device), torch.tensor([h], device=mask.device), extend_factor)
        nx, ny, nw, nh = nx[0].item(), ny[0].item(), nw[0].item(), nh[0].item()
        ctx = mask[:, ny:ny+nh, nx:nx+nw]
        return ctx, nx, ny, nw, nh

    @abstractmethod
    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        pass

    def combinecontextmask_m(self, context, mask, x, y, w, h, optional_context_mask):
        _, nx, ny, nw, nh = self.batched_combinecontextmask_m(mask, torch.tensor([x], device=mask.device), torch.tensor([y], device=mask.device), torch.tensor([w], device=mask.device), torch.tensor([h], device=mask.device), optional_context_mask)
        nx, ny, nw, nh = nx[0].item(), ny[0].item(), nw[0].item(), nh[0].item()
        ctx = mask[:, ny:ny+nh, nx:nx+nw]
        return ctx, nx, ny, nw, nh

    @abstractmethod
    def crop_magic_im(self, image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm, resize_output=True):
        pass

    @abstractmethod
    def stitch_magic_im(self, canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
        pass


class CPUProcessorLogic(ProcessorLogic):
    def rescale_i(self, samples, width, height, algorithm: str):
        # samples shape: [B, H, W, C]
        samples = samples.movedim(-1, 1) # [B, C, H, W]
        algorithm_enum = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
        results = []
        for i in range(samples.shape[0]):
            samples_pil: Image.Image = F.to_pil_image(samples[i].cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(samples_pil))
        samples = torch.stack(results, dim=0)
        samples = samples.movedim(1, -1)
        return samples

    def rescale_m(self, samples, width, height, algorithm: str):
        # samples shape: [B, H, W]
        algorithm_enum = getattr(Image, algorithm.upper())  # i.e. Image.BICUBIC
        results = []
        for i in range(samples.shape[0]):
            samples_pil: Image.Image = F.to_pil_image(samples[i].cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(samples_pil).squeeze(0))
        samples = torch.stack(results, dim=0)
        return samples

    def fillholes_iterative_hipass_fill_m(self, samples):
        thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = []
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            for threshold in thresholds:
                thresholded_mask = mask_np >= threshold
                closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))
            results.append(torch.from_numpy(mask_np.astype(np.float32)))
        return torch.stack(results, dim=0)

    def hipassfilter_m(self, samples, threshold):
        filtered_mask = samples.clone()
        filtered_mask[filtered_mask < threshold] = 0
        return filtered_mask

    def expand_m(self, mask, pixels):
        sigma = pixels / 4
        kernel_size = math.ceil(sigma * 1.5 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        results = []
        for i in range(mask.shape[0]):
            mask_np = mask[i].cpu().numpy()
            dilated_mask = grey_dilation(mask_np, footprint=kernel, mode='reflect')
            results.append(torch.from_numpy(dilated_mask.astype(np.float32)).clamp(0.0, 1.0))
        return torch.stack(results, dim=0)

    def invert_m(self, samples):
        inverted_mask = samples.clone()
        inverted_mask = 1.0 - inverted_mask
        return inverted_mask

    def blur_m(self, samples, pixels):
        sigma = pixels / 4 
        results = []
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            blurred_mask = gaussian_filter(mask_np, sigma=sigma, mode='reflect')
            results.append(torch.from_numpy(blurred_mask).float().clamp(0.0, 1.0))
        return torch.stack(results, dim=0)

    def debug_context_location_in_image(self, image, x, y, w, h):
        debug_image = image.clone()
        debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
        return debug_image

    def pad_to_multiple(self, value, multiple):
        return int(math.ceil(value / multiple) * multiple)

    def preresize_imm(self, image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
        current_width, current_height = image.shape[2], image.shape[1]
        
        if preresize_mode == "ensure minimum resolution":
            if current_width >= preresize_min_width and current_height >= preresize_min_height:
                return image, mask, optional_context_mask

            scale_factor_min_width = preresize_min_width / current_width
            scale_factor_min_height = preresize_min_height / current_height

            scale_factor = max(scale_factor_min_width, scale_factor_min_height)

            target_width = math.ceil(current_width * scale_factor)
            target_height = math.ceil(current_height * scale_factor)

            image = self.rescale_i(image, target_width, target_height, upscale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'bilinear')
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
            
            assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
                f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

        elif preresize_mode == "ensure minimum and maximum resolution":
            if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
                return image, mask, optional_context_mask

            scale_factor_min_width = preresize_min_width / current_width
            scale_factor_min_height = preresize_min_height / current_height
            scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

            scale_factor_max_width = preresize_max_width / current_width
            scale_factor_max_height = preresize_max_height / current_height
            scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

            if scale_factor_min > 1 and scale_factor_max < 1:
                assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
            
            if scale_factor_min > 1:  # We're upscaling to meet min resolution
                scale_factor = scale_factor_min
                rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
            else:  # We're downscaling to meet max resolution
                scale_factor = scale_factor_max
                rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

            if scale_factor >= 1.0:
                target_width = math.ceil(current_width * scale_factor)
                target_height = math.ceil(current_height * scale_factor)
            else:
                target_width = int(current_width * scale_factor)
                target_height = int(current_height * scale_factor)

            image = self.rescale_i(image, target_width, target_height, rescale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
            
            assert preresize_min_width <= target_width <= preresize_max_width, \
                f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
            assert preresize_min_height <= target_height <= preresize_max_height, \
                f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

        elif preresize_mode == "ensure maximum resolution":
            if current_width <= preresize_max_width and current_height <= preresize_max_height:
                return image, mask, optional_context_mask

            scale_factor_max_width = preresize_max_width / current_width
            scale_factor_max_height = preresize_max_height / current_height
            scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

            target_width = int(current_width * scale_factor_max)
            target_height = int(current_height * scale_factor_max)

            image = self.rescale_i(image, target_width, target_height, downscale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

            assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
                f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

        return image, mask, optional_context_mask

    def extend_imm(self, image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
        B, H, W, C = image.shape

        new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
        new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

        assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
        assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

        expanded_image = torch.zeros(B, new_H, new_W, C, device=image.device)
        expanded_mask = torch.ones(B, new_H, new_W, device=mask.device)
        expanded_optional_context_mask = torch.zeros(B, new_H, new_W, device=optional_context_mask.device)

        up_padding = int(H * (extend_up_factor - 1.0))
        down_padding = new_H - H - up_padding
        left_padding = int(W * (extend_left_factor - 1.0))
        right_padding = new_W - W - left_padding

        slice_target_up = max(0, up_padding)
        slice_target_down = min(new_H, up_padding + H)
        slice_target_left = max(0, left_padding)
        slice_target_right = min(new_W, left_padding + W)

        slice_source_up = max(0, -up_padding)
        slice_source_down = min(H, new_H - up_padding)
        slice_source_left = max(0, -left_padding)
        slice_source_right = min(W, new_W - left_padding)

        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
        if up_padding > 0:
            expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
        if down_padding > 0:
            expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
        if left_padding > 0:
            expanded_image[:, :, slice_target_up:slice_target_down, :left_padding] = expanded_image[:, :, slice_target_up:slice_target_down, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
        if right_padding > 0:
            expanded_image[:, :, slice_target_up:slice_target_down, -right_padding:] = expanded_image[:, :, slice_target_up:slice_target_down, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

        expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
        expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

        expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        return expanded_image, expanded_mask, expanded_optional_context_mask

    def batched_findcontextarea_m(self, mask):
        # Optimized GPU implementation or CPU fallback 
        B, H, W = mask.shape
        device = mask.device
        
        # If on GPU, we can use vectorized approach.
        # But for now, let's just use the shared logic that works on both.
        # Wait, I'll use the vectorized one in the NEXT step for GPU specifically.
        # This global step just fixes the placeholder.
        
        x_list, y_list, w_list, h_list = [], [], [], []
        for i in range(B):
            mask_squeezed = mask[i]
            non_zero_indices = torch.nonzero(mask_squeezed)
            if non_zero_indices.numel() == 0:
                bx, by, bw, bh = -1, -1, -1, -1
            else:
                by = torch.min(non_zero_indices[:, 0]).item()
                bx = torch.min(non_zero_indices[:, 1]).item()
                by_max = torch.max(non_zero_indices[:, 0]).item()
                bx_max = torch.max(non_zero_indices[:, 1]).item()
                bw = bx_max - bx + 1
                bh = by_max - by + 1
            x_list.append(bx)
            y_list.append(by)
            w_list.append(bw)
            h_list.append(bh)
        return None, torch.tensor(x_list, device=device), torch.tensor(y_list, device=device), torch.tensor(w_list, device=device), torch.tensor(h_list, device=device)

    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        img_h, img_w = mask.shape[1], mask.shape[2]
        device = mask.device
        
        grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
        grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()
        
        new_x = torch.clamp(x - grow_x, min=0)
        new_y = torch.clamp(y - grow_y, min=0)
        new_x2 = torch.clamp(x + w + grow_x, max=img_w)
        new_y2 = torch.clamp(y + h + grow_y, max=img_h)
        
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        
        empty = (w == -1)
        new_x[empty] = 0
        new_y[empty] = 0
        new_w[empty] = img_w
        new_h[empty] = img_h
        
        return None, new_x, new_y, new_w, new_h

    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        _, ox, oy, ow, oh = self.batched_findcontextarea_m(optional_context_mask)
        
        mask_x_neg1 = (x == -1)
        x_1 = torch.where(mask_x_neg1, ox, x)
        y_1 = torch.where(mask_x_neg1, oy, y)
        w_1 = torch.where(mask_x_neg1, ow, w)
        h_1 = torch.where(mask_x_neg1, oh, h)
        
        mask_ox_neg1 = (ox == -1)
        ox_2 = torch.where(mask_ox_neg1, x_1, ox)
        oy_2 = torch.where(mask_ox_neg1, y_1, oy)
        ow_2 = torch.where(mask_ox_neg1, w_1, ow)
        oh_2 = torch.where(mask_ox_neg1, h_1, oh)
        
        new_x = torch.min(x_1, ox_2)
        new_y = torch.min(y_1, oy_2)
        new_x_max = torch.max(x_1 + w_1, ox_2 + ow_2)
        new_y_max = torch.max(y_1 + h_1, oy_2 + oh_2)
        new_w = new_x_max - new_x
        new_h = new_y_max - new_y
        
        both_empty = (x_1 == -1)
        new_x[both_empty] = -1
        new_y[both_empty] = -1
        new_w[both_empty] = -1
        new_h[both_empty] = -1
        
        return None, new_x, new_y, new_w, new_h

    def crop_magic_im(self, image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm, resize_output=True):
        image = image.clone()
        mask = mask.clone()
        
        # Check for invalid inputs
        if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
            return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

        # Step 1: Pad target dimensions to be multiples of padding
        if padding != 0:
            target_w = self.pad_to_multiple(target_w, padding)
            target_h = self.pad_to_multiple(target_h, padding)

        # Step 2: Calculate target aspect ratio
        target_aspect_ratio = target_w / target_h

        # Step 3: Grow current context area to meet the target aspect ratio
        B, image_h, image_w, C = image.shape
        context_aspect_ratio = w / h
        if context_aspect_ratio < target_aspect_ratio:
            # Grow width to meet aspect ratio
            new_w = int(h * target_aspect_ratio)
            new_h = h
            new_x = x - (new_w - w) // 2
            new_y = y

            # Adjust new_x to keep within bounds
            if new_x < 0:
                shift = -new_x
                if new_x + new_w + shift <= image_w:
                    new_x += shift
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow
            elif new_x + new_w > image_w:
                overflow = new_x + new_w - image_w
                if new_x - overflow >= 0:
                    new_x -= overflow
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow

        else:
            # Grow height to meet aspect ratio
            new_w = w
            new_h = int(w / target_aspect_ratio)
            new_x = x
            new_y = y - (new_h - h) // 2

            # Adjust new_y to keep within bounds
            if new_y < 0:
                shift = -new_y
                if new_y + new_h + shift <= image_h:
                    new_y += shift
                else:
                    overflow = (new_h - image_h) // 2
                    new_y = -overflow
            elif new_y + new_h > image_h:
                overflow = new_y + new_h - image_h
                if new_y - overflow >= 0:
                    new_y -= overflow
                else:
                    overflow = (new_h - image_h) // 2
                    new_y = -overflow

        # Step 3b: When not resizing output, ensure dimensions are at least target dimensions
        # This ensures output_padding works correctly even without resize (Option A: expand context, keep centered)
        if not resize_output:
            if new_w < target_w:
                grow_w = target_w - new_w
                new_x -= grow_w // 2
                new_w = target_w
                # Recalculate bounds
                if new_x < 0:
                    shift = -new_x
                    if new_x + new_w + shift <= image_w:
                        new_x += shift
                    else:
                        new_x = -((new_w - image_w) // 2)
                elif new_x + new_w > image_w:
                    overflow = new_x + new_w - image_w
                    if new_x - overflow >= 0:
                        new_x -= overflow
                    else:
                        new_x = -((new_w - image_w) // 2)
            if new_h < target_h:
                grow_h = target_h - new_h
                new_y -= grow_h // 2
                new_h = target_h
                # Recalculate bounds
                if new_y < 0:
                    shift = -new_y
                    if new_y + new_h + shift <= image_h:
                        new_y += shift
                    else:
                        new_y = -((new_h - image_h) // 2)
                elif new_y + new_h > image_h:
                    overflow = new_y + new_h - image_h
                    if new_y - overflow >= 0:
                        new_y -= overflow
                    else:
                        new_y = -((new_h - image_h) // 2)

        # Step 4: Grow the image to accommodate the new context area
        up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

        expanded_image_w = image_w
        expanded_image_h = image_h

        # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
        if new_x < 0:
            left_padding = -new_x
            expanded_image_w += left_padding
        if new_x + new_w > image_w:
            right_padding = (new_x + new_w - image_w)
            expanded_image_w += right_padding
        # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
        if new_y < 0:
            up_padding = -new_y
            expanded_image_h += up_padding 
        if new_y + new_h > image_h:
            down_padding = (new_y + new_h - image_h)
            expanded_image_h += down_padding

        # Step 5: Create the new image and mask
        expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
        expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

        # Reorder the tensors to match the required dimension format for padding
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Ensure the expanded image has enough room to hold the padded version of the original image
        expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

        # Fill the new extended areas with the edge values of the image
        if up_padding > 0:
            expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = expanded_image[:, :, up_padding:up_padding + 1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
        if down_padding > 0:
            expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = expanded_image[:, :, up_padding + image_h - 1:up_padding + image_h, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
        if left_padding > 0:
            expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
        if right_padding > 0:
            expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

        # Reorder the tensors back to [B, H, W, C] format
        expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Same for the mask
        expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

        # Record the cto values (canvas to original)
        cto_x = left_padding
        cto_y = up_padding
        cto_w = image_w
        cto_h = image_h

        # The final expanded image and mask
        canvas_image = expanded_image
        canvas_mask = expanded_mask

        # Step 6: Crop the image and mask around x, y, w, h
        ctc_x = new_x+left_padding
        ctc_y = new_y+up_padding
        ctc_w = new_w
        ctc_h = new_h

        # Crop the image and mask
        cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # Step 7: Resize image and mask to the target width and height
        if resize_output:
            # Decide which algorithm to use based on the scaling direction
            if target_w > ctc_w or target_h > ctc_h:  # Upscaling
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
            else:  # Downscaling
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

        return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h

    def stitch_magic_im(self, canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
        canvas_image = canvas_image.clone()
        inpainted_image = inpainted_image.clone()
        mask = mask.clone()

        # Resize inpainted image and mask to match the context size
        B, h, w, _ = inpainted_image.shape
        if ctc_w > w or ctc_h > h:  # Upscaling
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
        else:  # Downscaling
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

        # Clamp mask to [0, 1] and expand to match image channels
        resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [B, H, W, 1]

        # Extract the canvas region we're about to overwrite
        canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # Blend: new = mask * inpainted + (1 - mask) * canvas
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

        # Paste the blended region back onto the canvas
        canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

        # Final crop to get back the original image area
        output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

        return output_image


class GPUProcessorLogic(ProcessorLogic):
    def rescale_i(self, samples, width, height, algorithm: str):
        # samples shape: [B, H, W, C]
        mode = algorithm.lower()
        
        # CPU works better, fallback to CPU for rescaling
        original_device = samples.device
        samples = samples.movedim(-1, 1)  # [B, C, H, W]
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            samples_pil: Image.Image = F.to_pil_image(samples[i].float().cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(samples_pil))
        samples = torch.stack(results, dim=0).to(original_device)
        samples = samples.movedim(1, -1)
        return samples
        
        #samples = samples.movedim(-1, 1)  # [B, C, H, W]
        #samples = TF.interpolate(samples, size=(height, width), mode=mode, align_corners=False if mode not in ['nearest', 'area'] else None)
        #samples = samples.movedim(1, -1)
        #return samples

    def rescale_m(self, samples, width, height, algorithm: str):
        # samples shape: [B, H, W]
        mode = algorithm.lower()
        
        # CPU works better, fallback to CPU for rescaling
        original_device = samples.device
        algorithm_enum = getattr(Image, algorithm.upper())
        results = []
        for i in range(samples.shape[0]):
            samples_pil: Image.Image = F.to_pil_image(samples[i].float().cpu()).resize((width, height), algorithm_enum)
            results.append(F.to_tensor(samples_pil).squeeze(0))
        samples = torch.stack(results, dim=0).to(original_device)
        return samples
        
        #samples = samples.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        #samples = TF.interpolate(samples, size=(height, width), mode=mode, align_corners=False if mode not in ['nearest', 'area'] else None)
        #samples = samples.squeeze(1)
        #return samples

    def fillholes_iterative_hipass_fill_m(self, samples):
        # We want this to always run in CPU for simplicity of implementation.
        # Just convert whatever inputs from GPU to CPU at the beginning of the function,
        # then convert them back to GPU at the end of the function.
        # The implementation is verbatim from CPUProcessorLogic.

        thresholds = [1, 0.99, 0.97, 0.95, 0.93, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        results = []
        original_device = samples.device
        for i in range(samples.shape[0]):
            mask_np = samples[i].cpu().numpy()
            for threshold in thresholds:
                thresholded_mask = mask_np >= threshold
                closed_mask = binary_closing(thresholded_mask, structure=np.ones((3, 3)), border_value=1)
                filled_mask = binary_fill_holes(closed_mask)
                mask_np = np.maximum(mask_np, np.where(filled_mask != 0, threshold, 0))
            results.append(torch.from_numpy(mask_np.astype(np.float32)))
        return torch.stack(results, dim=0).to(original_device)

    def hipassfilter_m(self, samples, threshold):
        filtered_mask = samples.clone()
        filtered_mask[filtered_mask < threshold] = 0
        return filtered_mask

    def expand_m(self, mask, pixels):
        # Dilation can be approximated with max pooling
        sigma = pixels / 4
        kernel_size = math.ceil(sigma * 1.5 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        padding = kernel_size // 2
        
        # mask is [B, H, W] -> [B, 1, H, W]
        mask_in = mask.unsqueeze(1)
        
        # Reflect padding to avoid border transparency
        mask_padded = TF.pad(mask_in, (padding, padding, padding, padding), mode='reflect')
        
        # MaxPool2d is equivalent to dilation with a square kernel of 1s
        dilated = TF.max_pool2d(mask_padded, kernel_size=kernel_size, stride=1, padding=0)
        
        return dilated.squeeze(1)

    def invert_m(self, samples):
        inverted_mask = samples.clone()
        inverted_mask = 1.0 - inverted_mask
        return inverted_mask

    def blur_m(self, samples, pixels):
        sigma = pixels / 4
        # Gaussian blur implementation on GPU
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        
        # Create gaussian kernel
        x = torch.arange(kernel_size, device=samples.device, dtype=samples.dtype) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (x / sigma).pow(2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.expand(1, 1, kernel_size, kernel_size)
        
        mask_in = samples.unsqueeze(1)
        pad = kernel_size // 2
        
        # Reflect padding before convolution so the border stays solid
        mask_padded = TF.pad(mask_in, (pad, pad, pad, pad), mode='reflect')
        blurred = TF.conv2d(mask_padded, kernel_2d, padding=0, groups=1)
        
        return blurred.squeeze(1).clamp(0.0, 1.0)

    def debug_context_location_in_image(self, image, x, y, w, h):
        debug_image = image.clone()
        debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
        return debug_image

    def pad_to_multiple(self, value, multiple):
        return int(math.ceil(value / multiple) * multiple)

    def preresize_imm(self, image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
        current_width, current_height = image.shape[2], image.shape[1]
        
        if preresize_mode == "ensure minimum resolution":
            if current_width >= preresize_min_width and current_height >= preresize_min_height:
                return image, mask, optional_context_mask

            scale_factor_min_width = preresize_min_width / current_width
            scale_factor_min_height = preresize_min_height / current_height

            scale_factor = max(scale_factor_min_width, scale_factor_min_height)

            target_width = math.ceil(current_width * scale_factor)
            target_height = math.ceil(current_height * scale_factor)

            image = self.rescale_i(image, target_width, target_height, upscale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'bilinear')
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
            
            assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
                f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

        elif preresize_mode == "ensure minimum and maximum resolution":
            if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
                return image, mask, optional_context_mask

            scale_factor_min_width = preresize_min_width / current_width
            scale_factor_min_height = preresize_min_height / current_height
            scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

            scale_factor_max_width = preresize_max_width / current_width
            scale_factor_max_height = preresize_max_height / current_height
            scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

            if scale_factor_min > 1 and scale_factor_max < 1:
                assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
            
            if scale_factor_min > 1:  # We're upscaling to meet min resolution
                scale_factor = scale_factor_min
                rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
            else:  # We're downscaling to meet max resolution
                scale_factor = scale_factor_max
                rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

            if scale_factor >= 1.0:
                target_width = math.ceil(current_width * scale_factor)
                target_height = math.ceil(current_height * scale_factor)
            else:
                target_width = int(current_width * scale_factor)
                target_height = int(current_height * scale_factor)

            image = self.rescale_i(image, target_width, target_height, rescale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
            
            assert preresize_min_width <= target_width <= preresize_max_width, \
                f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
            assert preresize_min_height <= target_height <= preresize_max_height, \
                f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

        elif preresize_mode == "ensure maximum resolution":
            if current_width <= preresize_max_width and current_height <= preresize_max_height:
                return image, mask, optional_context_mask

            scale_factor_max_width = preresize_max_width / current_width
            scale_factor_max_height = preresize_max_height / current_height
            scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

            target_width = int(current_width * scale_factor_max)
            target_height = int(current_height * scale_factor_max)

            image = self.rescale_i(image, target_width, target_height, downscale_algorithm)
            mask = self.rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
            optional_context_mask = self.rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

            assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
                f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

        return image, mask, optional_context_mask

    def extend_imm(self, image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
        B, H, W, C = image.shape

        new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
        new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

        assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
        assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

        expanded_image = torch.zeros(B, new_H, new_W, C, device=image.device)
        expanded_mask = torch.ones(B, new_H, new_W, device=mask.device)
        expanded_optional_context_mask = torch.zeros(B, new_H, new_W, device=optional_context_mask.device)

        up_padding = int(H * (extend_up_factor - 1.0))
        down_padding = new_H - H - up_padding
        left_padding = int(W * (extend_left_factor - 1.0))
        right_padding = new_W - W - left_padding

        slice_target_up = max(0, up_padding)
        slice_target_down = min(new_H, up_padding + H)
        slice_target_left = max(0, left_padding)
        slice_target_right = min(new_W, left_padding + W)

        slice_source_up = max(0, -up_padding)
        slice_source_down = min(H, new_H - up_padding)
        slice_source_left = max(0, -left_padding)
        slice_source_right = min(W, new_W - left_padding)

        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
        if up_padding > 0:
            expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
        if down_padding > 0:
            expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
        if left_padding > 0:
            expanded_image[:, :, slice_target_up:slice_target_down, :left_padding] = expanded_image[:, :, slice_target_up:slice_target_down, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
        if right_padding > 0:
            expanded_image[:, :, slice_target_up:slice_target_down, -right_padding:] = expanded_image[:, :, slice_target_up:slice_target_down, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

        expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
        expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

        expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        return expanded_image, expanded_mask, expanded_optional_context_mask

    def batched_findcontextarea_m(self, mask):
        # Optimized GPU implementation using parallel max/where
        B, H, W = mask.shape
        device = mask.device
        
        # Find which rows and columns have any mask content
        any_y = mask.max(dim=2).values > 0. # [B, H]
        any_x = mask.max(dim=1).values > 0. # [B, W]
        
        def get_min_max(any_dim, size):
            indices = torch.arange(size, device=device).unsqueeze(0).expand(B, -1)
            # Use large value for min where it's False, -1 for max where it's False
            min_indices = torch.where(any_dim, indices, torch.tensor(size, device=device))
            max_indices = torch.where(any_dim, indices, torch.tensor(-1, device=device))
            
            b_min = torch.min(min_indices, dim=1).values
            b_max = torch.max(max_indices, dim=1).values
            
            # Handle cases where the whole row is False (no mask content for that batch item)
            empty = ~any_dim.any(dim=1)
            b_min[empty] = -1
            b_max[empty] = -1
            
            return b_min, b_max

        y_min, y_max = get_min_max(any_y, H)
        x_min, x_max = get_min_max(any_x, W)
        
        w = torch.where(x_min >= 0, x_max - x_min + 1, torch.tensor(-1, device=device))
        h = torch.where(y_min >= 0, y_max - y_min + 1, torch.tensor(-1, device=device))
        
        return None, x_min, y_min, w, h

    def batched_growcontextarea_m(self, mask, x, y, w, h, extend_factor):
        img_h, img_w = mask.shape[1], mask.shape[2]
        device = mask.device
        
        grow_x = (w.float() * (extend_factor - 1.0) / 2.0).round().long()
        grow_y = (h.float() * (extend_factor - 1.0) / 2.0).round().long()
        
        new_x = torch.clamp(x - grow_x, min=0)
        new_y = torch.clamp(y - grow_y, min=0)
        new_x2 = torch.clamp(x + w + grow_x, max=img_w)
        new_y2 = torch.clamp(y + h + grow_y, max=img_h)
        
        new_w = new_x2 - new_x
        new_h = new_y2 - new_y
        
        empty = (w == -1)
        new_x[empty] = 0
        new_y[empty] = 0
        new_w[empty] = img_w
        new_h[empty] = img_h
        
        return None, new_x, new_y, new_w, new_h

    def batched_combinecontextmask_m(self, mask, x, y, w, h, optional_context_mask):
        _, ox, oy, ow, oh = self.batched_findcontextarea_m(optional_context_mask)
        
        mask_x_neg1 = (x == -1)
        x_1 = torch.where(mask_x_neg1, ox, x)
        y_1 = torch.where(mask_x_neg1, oy, y)
        w_1 = torch.where(mask_x_neg1, ow, w)
        h_1 = torch.where(mask_x_neg1, oh, h)
        
        mask_ox_neg1 = (ox == -1)
        ox_2 = torch.where(mask_ox_neg1, x_1, ox)
        oy_2 = torch.where(mask_ox_neg1, y_1, oy)
        ow_2 = torch.where(mask_ox_neg1, w_1, ow)
        oh_2 = torch.where(mask_ox_neg1, h_1, oh)
        
        new_x = torch.min(x_1, ox_2)
        new_y = torch.min(y_1, oy_2)
        new_x_max = torch.max(x_1 + w_1, ox_2 + ow_2)
        new_y_max = torch.max(y_1 + h_1, oy_2 + oh_2)
        new_w = new_x_max - new_x
        new_h = new_y_max - new_y
        
        both_empty = (x_1 == -1)
        new_x[both_empty] = -1
        new_y[both_empty] = -1
        new_w[both_empty] = -1
        new_h[both_empty] = -1
        
        return None, new_x, new_y, new_w, new_h

    def crop_magic_im(self, image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm, resize_output=True):
        image = image.clone()
        mask = mask.clone()
        
        # Check for invalid inputs
        if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
            return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

        # Step 1: Pad target dimensions to be multiples of padding
        if padding != 0:
            target_w = self.pad_to_multiple(target_w, padding)
            target_h = self.pad_to_multiple(target_h, padding)

        # Step 2: Calculate target aspect ratio
        target_aspect_ratio = target_w / target_h

        # Step 3: Grow current context area to meet the target aspect ratio
        B, image_h, image_w, C = image.shape
        context_aspect_ratio = w / h
        if context_aspect_ratio < target_aspect_ratio:
            # Grow width to meet aspect ratio
            new_w = int(h * target_aspect_ratio)
            new_h = h
            new_x = x - (new_w - w) // 2
            new_y = y

            # Adjust new_x to keep within bounds
            if new_x < 0:
                shift = -new_x
                if new_x + new_w + shift <= image_w:
                    new_x += shift
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow
            elif new_x + new_w > image_w:
                overflow = new_x + new_w - image_w
                if new_x - overflow >= 0:
                    new_x -= overflow
                else:
                    overflow = (new_w - image_w) // 2
                    new_x = -overflow

        else:
            # Grow height to meet aspect ratio
            new_w = w
            new_h = int(w / target_aspect_ratio)
            new_x = x
            new_y = y - (new_h - h) // 2

            # Adjust new_y to keep within bounds
            if new_y < 0:
                shift = -new_y
                if new_y + new_h + shift <= image_h:
                    new_y += shift
                else:
                    overflow = (new_h - image_h) // 2
                    new_y = -overflow
            elif new_y + new_h > image_h:
                overflow = new_y + new_h - image_h
                if new_y - overflow >= 0:
                    new_y -= overflow
                else:
                    overflow = (new_h - image_h) // 2
                    new_y = -overflow

        # Step 3b: When not resizing output, ensure dimensions are at least target dimensions
        # This ensures output_padding works correctly even without resize (Option A: expand context, keep centered)
        if not resize_output:
            if new_w < target_w:
                grow_w = target_w - new_w
                new_x -= grow_w // 2
                new_w = target_w
                # Recalculate bounds
                if new_x < 0:
                    shift = -new_x
                    if new_x + new_w + shift <= image_w:
                        new_x += shift
                    else:
                        new_x = -((new_w - image_w) // 2)
                elif new_x + new_w > image_w:
                    overflow = new_x + new_w - image_w
                    if new_x - overflow >= 0:
                        new_x -= overflow
                    else:
                        new_x = -((new_w - image_w) // 2)
            if new_h < target_h:
                grow_h = target_h - new_h
                new_y -= grow_h // 2
                new_h = target_h
                # Recalculate bounds
                if new_y < 0:
                    shift = -new_y
                    if new_y + new_h + shift <= image_h:
                        new_y += shift
                    else:
                        new_y = -((new_h - image_h) // 2)
                elif new_y + new_h > image_h:
                    overflow = new_y + new_h - image_h
                    if new_y - overflow >= 0:
                        new_y -= overflow
                    else:
                        new_y = -((new_h - image_h) // 2)

        # Step 4: Grow the image to accommodate the new context area
        up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

        expanded_image_w = image_w
        expanded_image_h = image_h

        # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
        if new_x < 0:
            left_padding = -new_x
            expanded_image_w += left_padding
        if new_x + new_w > image_w:
            right_padding = (new_x + new_w - image_w)
            expanded_image_w += right_padding
        # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
        if new_y < 0:
            up_padding = -new_y
            expanded_image_h += up_padding 
        if new_y + new_h > image_h:
            down_padding = (new_y + new_h - image_h)
            expanded_image_h += down_padding

        # Step 5: Create the new image and mask
        expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
        expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

        # Reorder the tensors to match the required dimension format for padding
        image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Ensure the expanded image has enough room to hold the padded version of the original image
        expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

        # Fill the new extended areas with the edge values of the image
        if up_padding > 0:
            expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = expanded_image[:, :, up_padding:up_padding + 1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
        if down_padding > 0:
            expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = expanded_image[:, :, up_padding + image_h - 1:up_padding + image_h, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
        if left_padding > 0:
            expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
        if right_padding > 0:
            expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

        # Reorder the tensors back to [B, H, W, C] format
        expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

        # Same for the mask
        expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

        # Record the cto values (canvas to original)
        cto_x = left_padding
        cto_y = up_padding
        cto_w = image_w
        cto_h = image_h

        # The final expanded image and mask
        canvas_image = expanded_image
        canvas_mask = expanded_mask

        # Step 6: Crop the image and mask around x, y, w, h
        ctc_x = new_x+left_padding
        ctc_y = new_y+up_padding
        ctc_w = new_w
        ctc_h = new_h

        # Crop the image and mask
        cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]
        cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # Step 7: Resize image and mask to the target width and height
        if resize_output:
            # Decide which algorithm to use based on the scaling direction
            if target_w > ctc_w or target_h > ctc_h:  # Upscaling
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
            else:  # Downscaling
                cropped_image = self.rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
                cropped_mask = self.rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

        return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h

    def stitch_magic_im(self, canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
        canvas_image = canvas_image.clone()
        inpainted_image = inpainted_image.clone()
        mask = mask.clone()

        # Resize inpainted image and mask to match the context size
        B, h, w, _ = inpainted_image.shape
        if ctc_w > w or ctc_h > h:  # Upscaling
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
        else:  # Downscaling
            resized_image = self.rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
            resized_mask = self.rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)

        # Clamp mask to [0, 1] and expand to match image channels
        resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [B, H, W, 1]

        # Extract the canvas region we're about to overwrite
        canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

        # Blend: new = mask * inpainted + (1 - mask) * canvas
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

        # Paste the blended region back onto the canvas
        canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blended

        # Final crop to get back the original image area
        output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]

        return output_image

class CR_InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "padding_pixels": ("INT", {"default": 32, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
            }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "CR Image Suite/Inpaint"
    DESCRIPTION = "Crops an image around a mask into a square region with configurable pixel padding."
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")
 
    def _build_square_bbox(self, x, y, w, h, padding_pixels):
        padded_x = x - padding_pixels
        padded_y = y - padding_pixels
        padded_w = w + (padding_pixels * 2)
        padded_h = h + (padding_pixels * 2)
        side = max(1, padded_w, padded_h)
        center_x = padded_x + (padded_w / 2.0)
        center_y = padded_y + (padded_h / 2.0)
        square_x = int(math.floor(center_x - (side / 2.0)))
        square_y = int(math.floor(center_y - (side / 2.0)))
        return square_x, square_y, side

    def inpaint_crop(self, image, mask, padding_pixels):
        image = image.clone()
        mask = mask.clone()
        processor = CPUProcessorLogic()

        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"

        if mask.shape[0] > 1 and image.shape[0] == 1:
            image = image.expand(mask.shape[0], -1, -1, -1).clone()
        if image.shape[0] > 1 and mask.shape[0] == 1:
            mask = mask.expand(image.shape[0], -1, -1).clone()

        if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
            if torch.count_nonzero(mask) == 0:
                mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"

        result_stitcher = {
            'downscale_algorithm': 'bilinear',
            'upscale_algorithm': 'bicubic',
            'blend_pixels': 0,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        result_image = []
        result_mask = []

        for i in range(image.shape[0]):
            sub_image = image[i:i+1]
            sub_mask = mask[i:i+1]
            _, bx, by, bw, bh = processor.batched_findcontextarea_m(sub_mask)

            if bx[0] == -1:
                x, y, w, h = 0, 0, sub_image.shape[2], sub_image.shape[1]
            else:
                x, y, w, h = bx[0].item(), by[0].item(), bw[0].item(), bh[0].item()

            square_x, square_y, square_side = self._build_square_bbox(x, y, w, h, int(padding_pixels))

            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = processor.crop_magic_im(
                sub_image,
                sub_mask,
                square_x,
                square_y,
                square_side,
                square_side,
                square_side,
                square_side,
                0,
                "bilinear",
                "bicubic",
                resize_output=False,
            )

            result_stitcher['canvas_to_orig_x'].append(cto_x)
            result_stitcher['canvas_to_orig_y'].append(cto_y)
            result_stitcher['canvas_to_orig_w'].append(cto_w)
            result_stitcher['canvas_to_orig_h'].append(cto_h)
            result_stitcher['canvas_image'].append(canvas_image.cpu())
            result_stitcher['cropped_to_canvas_x'].append(ctc_x)
            result_stitcher['cropped_to_canvas_y'].append(ctc_y)
            result_stitcher['cropped_to_canvas_w'].append(ctc_w)
            result_stitcher['cropped_to_canvas_h'].append(ctc_h)
            result_stitcher['cropped_mask_for_blend'].append(cropped_mask.cpu())
            result_image.append(cropped_image.squeeze(0).cpu())
            result_mask.append(cropped_mask.squeeze(0).cpu())

        return (result_stitcher, torch.stack(result_image, dim=0), torch.stack(result_mask, dim=0))


class CR_InpaintStitchImproved:
    """
    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            }
        }

    CATEGORY = "CR Image Suite/Inpaint"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"


    def inpaint_stitch(self, stitcher, inpainted_image):
        inpainted_image = inpainted_image.clone()
        results = []
        device = torch.device("cpu")
        processor = CPUProcessorLogic()

        # Pre-move stitcher data to device to avoid moving in loop
        for key in ['canvas_image', 'cropped_mask_for_blend']:
            if key in stitcher:
                stitcher[key] = [t.to(device) if torch.is_tensor(t) else t for t in stitcher[key]]

        batch_size = inpainted_image.shape[0]
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, "Stitch batch size doesn't match image batch size"
        override = False
        if len(stitcher['cropped_to_canvas_x']) != batch_size and len(stitcher['cropped_to_canvas_x']) == 1:
            override = True
        
        for i in range(batch_size):
            one_image = inpainted_image[i:i+1]
            
            one_stitcher = {}
            for key in ['downscale_algorithm', 'upscale_algorithm', 'blend_pixels']:
                one_stitcher[key] = stitcher[key]
            for key in ['canvas_to_orig_x', 'canvas_to_orig_y', 'canvas_to_orig_w', 'canvas_to_orig_h', 'canvas_image', 'cropped_to_canvas_x', 'cropped_to_canvas_y', 'cropped_to_canvas_w', 'cropped_to_canvas_h', 'cropped_mask_for_blend']:
                if override: 
                    one_stitcher[key] = stitcher[key][0]
                else:
                    one_stitcher[key] = stitcher[key][i]

            one_image, = self.inpaint_stitch_single_image(one_stitcher, one_image, processor)
            results.append(one_image.squeeze(0))

        result_batch = torch.stack(results, dim=0)
        result_batch = result_batch.cpu()

        return (result_batch,)

    def inpaint_stitch_single_image(self, stitcher, inpainted_image, processor):
        downscale_algorithm = stitcher.get('downscale_algorithm', 'bilinear')
        upscale_algorithm = stitcher.get('upscale_algorithm', 'bicubic')
        canvas_image = stitcher['canvas_image']

        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']

        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']

        mask = stitcher['cropped_mask_for_blend']  # shape: [1, H, W]

        output_image = processor.stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)

        return (output_image,)

# Mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "CR Inpaint Crop": CR_InpaintCropImproved,
    "CR Inpaint Stitch": CR_InpaintStitchImproved,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CR Inpaint Crop": "CR Inpaint Crop",
    "CR Inpaint Stitch": "CR Inpaint Stitch",
}
