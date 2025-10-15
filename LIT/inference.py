# Copyright 2024 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable
import math

import monai
import torch
import numpy as np
from generative.inferers import DiffusionInferer
from tqdm import tqdm
#from torch.amp import autocast # previous: from torch.cuda.amp import autocast

from LIT.utils import *



class InpaintingInferer():

    def __init__(self, inference_steps, scheduler, diffusion_model):
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_inference_steps=inference_steps)
        self.inference_steps = inference_steps
        self.model = diffusion_model


    @torch.no_grad()
    def __call__(self, mask: torch.Tensor, image_masked: torch.Tensor, 
                 num_resample_steps=10, num_resample_jumps=5, get_intermediates=False,
                 scale_factor=None,
                 *args, **kwargs):
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        assert(set(np.unique(mask)) == set((0,1))), 'mask is not binary but has values {}'.format(np.unique(mask))
        assert(mask.device == image_masked.device), 'mask and input must be on the same device'
        assert(image_masked.device == self.model.device), 'model and inputs must be on the same device'
        device = image_masked.device

        self.model.eval()
        intermediates = []

        #timesteps = torch.Tensor((self.inference_steps,),device=device).long()
        image_inpainted = torch.randn(image_masked.shape, device=device)


        for t in tqdm(self.scheduler.timesteps):
            last_step = t == 0
            # skip resampling at last step and do jumps
            if last_step or t % num_resample_jumps != 0: samplings_at_cuttent_t = 1
            else: samplings_at_cuttent_t = num_resample_steps

            for u in range(samplings_at_cuttent_t):
                last_sample_step = u == (samplings_at_cuttent_t - 1)
                
                # get the known region at t-1 (forward diffusion to get t-1)
                if not last_step:
                    image_inpainted_backward_known = self.sample_forward_diffusion(image_masked, t-1)
                else:
                    image_inpainted_backward_known = image_masked # is this skipping the last denoising?

                # perform a denoising step to get the unknown region at t-1 (backward diffusion to get t-1)
                # NOTE: consider whether this should be done at the last step or not
                image_backward_forward_unknown = self.diffusion_backward(image_inpainted, t, sf=scale_factor)

                # fuse known and unknown regions
                image_inpainted = torch.where(
                    mask == 0, image_inpainted_backward_known, image_backward_forward_unknown
                )

                #if t % num_resample_jumps == 0 and not last_step:
                # if we are doing resampling add noise and go again
                if not last_sample_step: 
                    # perform resampling (forward & backward)
                    # for u in range(num_resample_steps):
                    #     image_inpainted = self.resample(image_inpainted, t)
                    image_inpainted = self.diffusion_forward(image_inpainted, t-1)


            if get_intermediates and t % 50 == 0:
                intermediates.append(image_inpainted)

        # fuse known and unknown regions (only required in case of resampling during t=0)
        image_inpainted = torch.where(mask == 0, image_masked, image_inpainted)

        if get_intermediates:
            return image_inpainted, intermediates
        else:
            return image_inpainted
        
    def inpainting_step(self, image_masked, mask, t):
        image_inpainted_backward_known = self.sample_forward_diffusion(image_masked, t-1)
        image_backward_forward_unknown = self.diffusion_backward(image_inpainted, t)
        # fuse known and unknown regions
        image_inpainted = torch.where(
            mask == 0, image_inpainted_backward_known, image_backward_forward_unknown
        )
        return image_inpainted

    def sample_forward_diffusion(self, image, t): # add noise at 
        noise = torch.randn((image.shape), device=image.device)
        #timesteps_prev = torch.tensor((t,), device=image.device).long()
        # sqrt(alpha) * sample + (1-sqrt(alpha)) * noise
        noised_image = self.scheduler.add_noise(original_samples=image, noise=noise, timesteps=t)
        return noised_image
    
    def diffusion_forward(self, image, t):
        noise = torch.randn((image.shape),device=image.device)
        # sqrt(1-beta) * image + sqrt(beta) * noise
        image_inpainted = (torch.sqrt(1 - self.scheduler.betas[t]) * image + torch.sqrt(self.scheduler.betas[t]) * noise)

        return image_inpainted


    def diffusion_backward(self, image, t, sf=None): # TODO: add jumps
        # the model outputs the noise
        #if sf is None:
        #    model_output = self.model(image, timesteps=torch.tensor((t,),device=image.device).long())
        #else:
        model_output = self.model(image, scale_factors=sf, timesteps=torch.tensor((t,),device=image.device).long())
        # use model output to denoise and add noise again
        image_inpainted_prev_unknown, _ = self.scheduler.step(model_output, t.item(), image)

        return image_inpainted_prev_unknown


    # def resample(self, image, t, sf): # forward and back
    #     image_forward = self.diffusion_forward(image, t)
    #     image_backward = self.diffusion_backward(image_forward, t+1, sf)
    #     return image_backward



class SliceWiseInpaintingInferer(InpaintingInferer):

    def __init__(self, dimensions, diffusion_model, scheduler, inference_steps):
        super().__init__(inference_steps, scheduler, diffusion_model)
        self.dimension = dimensions
        self.slice_thickness = diffusion_model.in_channels

    def get_slice_from_volume(self, volume, slice_cut, dimension): # TODO: make sure channel dimension is always first
        threed_to_twod_slice = [slice(None), slice(None), slice(None)]
        threed_to_twod_slice[dimension] = slice(slice_cut-self.slice_thickness//2, slice_cut+self.slice_thickness//2+1)
        threed_to_twod_slice = tuple(threed_to_twod_slice)

        volume = volume[threed_to_twod_slice]
        return volume
    
    @staticmethod
    def slice_selector(start_idx, end_idx, dimension):
        selected_slice = [slice(None), slice(None), slice(None)]
        selected_slice[dimension] = slice(start_idx, end_idx)
        selected_slice = tuple(selected_slice)
        return selected_slice


    
    def get_inference_slices(self, mask, image_masked, dimension, offset=0):
        chosen_slices = np.arange(self.slice_thickness // 2 - offset, 
                                  image_masked.shape[dimension]- self.slice_thickness // 2 + offset, 
                                  self.slice_thickness)
        
        image_inpainted = torch.zeros_like(image_masked)


        batched_slices = []
        batched_masks = []
        batch_slice_indices = []

        for slice_index in chosen_slices:
            image_masked_slice = self.get_slice_from_volume(image_masked, slice_index, dimension)
            mask_slice = self.get_slice_from_volume(mask, slice_index, dimension)

            if (mask_slice == 0).all():

                sl = self.slice_selector(slice_index-self.slice_thickness//2, slice_index+self.slice_thickness//2+1, dimension)
                image_inpainted[sl] = image_masked_slice
            else:
                batched_slices.append(image_masked_slice)
                batched_masks.append(mask_slice)
                batch_slice_indices.append(slice_index)

        batched_slices = torch.stack(batched_slices, dim=0)
        batched_masks = torch.stack(batched_masks, dim=0)

        return batched_slices, batched_masks, batch_slice_indices, image_inpainted
        
        
    
    def __call__(self, mask: torch.Tensor, image_masked: torch.Tensor,
                    batch_size=1,
                    num_resample_steps=10, num_resample_jumps=5, 
                    get_intermediates=False, 
                    scale_factor=None,
                    *args, **kwargs):

        intermediates = []

        batched_slices, batched_masks, batch_slice_indices, image_inpainted = self.get_inference_slices(mask, image_masked, self.dimension)

        
        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]

        for i in range(math.ceil(len(batched_slices) / batch_size)):

            slices_batch = get_batch(batched_slices, i)
            masks_batch = get_batch(batched_masks, i)
            slice_indices_batch = get_batch(batch_slice_indices, i)

            start_idx = slice_indices_batch[0] - self.slice_thickness // 2
            end_idx = slice_indices_batch[-1] + self.slice_thickness // 2 + 1

            print(f'Inpainting slices {start_idx} to {end_idx}')

            # put channel dimension first 
            slices_batch = torch.swapaxes(slices_batch, 1, self.dimension+1)
            masks_batch = torch.swapaxes(masks_batch, 1, self.dimension+1)
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.dimension)
            
            image_inpainted_slice = super().__call__(masks_batch, slices_batch, 
                                                    num_resample_steps, num_resample_jumps, 
                                                    get_intermediates=get_intermediates, scale_factor=scale_factor, 
                                                    *args, **kwargs)
            
            

            if get_intermediates:
                image_inpainted_slice, interm = image_inpainted_slice
                intermediates.append(interm)
            
            image_inpainted_slice = torch.cat([img for img in image_inpainted_slice], dim=0)
            
            image_inpainted[start_idx:end_idx] = image_inpainted_slice

            # put channel dimension back
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.dimension)

        if get_intermediates:
            # return all intermediates for first slice of the first batch
            #return image_inpainted, torch.stack(intermediates[0])[:,0]

            # return all intermediates for the middle slice of the middle batch
            intermediates = intermediates[len(intermediates)//2] # select middle batch
            intermediates = torch.stack(intermediates) # collect all intermediates
            intermediates = intermediates[:, intermediates.shape[1]//2] # select middle slice
            return image_inpainted, intermediates
        else:
            return image_inpainted
        

class TwoAndHalfDInpaintingInferer(SliceWiseInpaintingInferer):

    def __init__(self, diffusion_model_dict, scheduler, inference_steps):
        super().__init__(None, list(diffusion_model_dict.values())[0], scheduler, inference_steps)
        #super().super().__init__(inference_steps, scheduler, None)
        self.diffusion_model_dict = diffusion_model_dict
        # map planes to dimensions assuming RAS orientation
        self.plane_to_dimension = {'sagittal': 0, 'axial': 1, 'coronal': 2}
        self.dimension_to_plane = {0: 'sagittal', 1: 'axial', 2: 'coronal'}


    def view_agg_inference(self, image_masked: torch.Tensor, mask: torch.Tensor, 
                           batch_size: int, inference_slices: dict,
                           num_resample_steps: int, num_resample_jumps: int,
                           get_intermediates: bool, scale_factor=None,
                           verbose=True):
        
        #image_inpainted = torch.randn(image_masked.shape, device=image_masked.device)

        # set mask to region to noise
        image_inpainted = torch.where(
                    mask == 0, image_masked, torch.randn(image_masked.shape, device=image_masked.device)
        )

        #plot_batch(image_inpainted.unsqueeze(0), '/groups/ag-reuter/projects/fastsurfer-tumor/monai_generative/diffusion_inpainting/experiments/inference_viewagg/images/initial_noised.png', slice_cut=[100, 84, 140])

        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]
        intermediates = []

        if verbose:
            progress_bar = tqdm(self.scheduler.timesteps)
        else:
            progress_bar = iter(self.scheduler.timesteps)
        
        for t in progress_bar:
            batch_outputs = []

            current_plane = self.dimension_to_plane[int(t%3)] # alternate between sagittal, axial, coronal
            self.model = self.diffusion_model_dict[current_plane] #.eval() # TODO: this may not be needed and might cause overhead (same for other eval calls)
            batched_slices, batched_masks, batch_slice_indices, _ = inference_slices[current_plane]

            

            # put channel dimension first 
            batched_slices = torch.swapaxes(batched_slices, 1, self.plane_to_dimension[current_plane]+1)
            batched_masks = torch.swapaxes(batched_masks, 1, self.plane_to_dimension[current_plane]+1)
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.plane_to_dimension[current_plane])

            # select region matching batches from already partially inpainted image
            image_inpainted_slice = []
            for idx in batch_slice_indices:
                image_inpainted_slice.append(image_inpainted[idx - self.slice_thickness//2:idx + self.slice_thickness//2+1])
            image_inpainted_slice = torch.stack(image_inpainted_slice, dim=0)

            # get checked image area (for logging)
            start_idx = batch_slice_indices[0] - self.slice_thickness // 2
            end_idx = batch_slice_indices[-1] + self.slice_thickness // 2 + 1

            # assert(len(batched_slices)*self.slice_thickness == end_idx - start_idx), 'batched_slices and image_inpainted_slice have different lengths'

            for i in range(math.ceil(len(batched_slices) / batch_size)): # iterate over batches
                # get current batch
                slices_batch = get_batch(batched_slices, i)
                masks_batch = get_batch(batched_masks, i)
                #slice_indices_batch = get_batch(batch_slice_indices, i)
                image_inpainted_slice_batch = get_batch(image_inpainted_slice, i)

                #print(f'Inpainting slices {start_idx} to {end_idx}, plane {current_plane}, timestep {t}')
                if verbose:
                    progress_bar.set_description(f'Inpainting slices {start_idx} to {end_idx}, plane {current_plane}, timestep {t}')
                
                d = self.denoise(t, masks_batch, slices_batch, image_inpainted_slice_batch,
                                                num_resample_steps, num_resample_jumps, scale_factor=scale_factor)
                batch_outputs.append(d)
                
            image_inpainted_slice_denoised = torch.cat([img for img in batch_outputs], dim=0)

            # unbind to remove batch dimension
            #image_inpainted[start_idx:end_idx] = torch.cat(torch.unbind(image_inpainted_slice_denoised)) 
            for i, idx in enumerate(batch_slice_indices):
                image_inpainted[idx - self.slice_thickness//2:idx + self.slice_thickness//2+1] = image_inpainted_slice_denoised[i]

            # put channel dimension back
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.plane_to_dimension[current_plane])

            if get_intermediates and t % 50 == 0:
                intermediates.append(image_inpainted.cpu())



        self.model = None # sanitize

        if get_intermediates:
            return image_inpainted, torch.stack(intermediates).unsqueeze(1)
        else:
            return image_inpainted


    @torch.no_grad()
    def denoise(self, t, mask: torch.Tensor, image_masked: torch.Tensor, image_inpainted: torch.Tensor,
                 num_resample_steps=10, num_resample_jumps=5, scale_factor=None):    
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """
        #assert(set(np.unique(mask)) == set((0,1))), 'mask is not binary but has values {}'.format(np.unique(mask))
        #assert(mask.device), 'mask and input must be on the same device'
        #assert(mask.device), 'model and inputs must be on the same device'

        #timesteps = torch.Tensor((self.inference_steps,),device=device).long()
        
        # DBG_PATH = '/groups/ag-reuter/projects/fastsurfer-tumor/monai_generative/diffusion_inpainting/experiments/inference_viewagg/images/'
        #plot_batch(image_inpainted, DBG_PATH+ 'dbg_pre_denoise.png', slice_cut=[100, 84, 140])


    
        last_step = t == 0

        # get known region at t
        image_inpainted = torch.where(
            mask == 0, self.sample_forward_diffusion(image_masked, t), image_inpainted
        )

        #plot_batch(self.sample_forward_diffusion(image_masked, t), DBG_PATH+ 'dbg_noise_only.png', slice_cut=[100, 84, 140])
        #plot_batch(image_inpainted, DBG_PATH+ 'dbg_noised.png', slice_cut=[100, 84, 140])

        # skip resampling at last step and do jumps
        if last_step or t % num_resample_jumps != 0: samplings_at_cuttent_t = 1
        else: samplings_at_cuttent_t = num_resample_steps

        for u in range(samplings_at_cuttent_t):
            last_sample_step = u == (samplings_at_cuttent_t - 1)
            
            # get the known region at t-1 (forward diffusion to get t-1)
            if not last_step:
                image_inpainted_backward_known = self.sample_forward_diffusion(image_masked, t-1)
            else:
                image_inpainted_backward_known = image_masked # is this skipping the last denoising?

            # perform a denoising step to get the unknown region at t (backward diffusion to get t-1)
            # NOTE: consider whether this should be done at the last step or not
            image_backward_forward_unknown = self.diffusion_backward(image_inpainted, t, sf=scale_factor)
            

            #plot_batch(image_backward_forward_unknown, DBG_PATH+ 'dbg_after_denoise.png', slice_cut=[100, 84, 140])

            # fuse known and unknown regions
            image_inpainted = torch.where(
                mask == 0, image_inpainted_backward_known, image_backward_forward_unknown
            )

            #plot_batch(image_inpainted, DBG_PATH+ 'dbg_denoised.png', slice_cut=[100, 84, 140])

            #if t % num_resample_jumps == 0 and not last_step:
            # if we are doing resampling add noise and go again
            if not last_sample_step: 
                # perform resampling (forward & backward)
                # for u in range(num_resample_steps):
                #     image_inpainted = self.resample(image_inpainted, t)
                image_inpainted = self.diffusion_forward(image_inpainted, t-1)

        # fuse known and unknown regions (only required in case of resampling during t=0)
        image_inpainted = torch.where(mask == 0, image_masked, image_inpainted)

        return image_inpainted

    def __call__(self, mask: torch.Tensor, image_masked: torch.Tensor, batch_size=1, 
                 num_resample_steps=10, num_resample_jumps=5, get_intermediates=False, scale_factor=None):

        if mask.dtype == torch.bool:
            mask = mask.int()
        else:
            assert(set(np.unique(mask)) == set((0,1))), 'mask is not binary but has values {}'.format(np.unique(mask))

        # unpack meta tensors (operating on torch tensors is faster)
        if isinstance(image_masked, monai.data.meta_tensor.MetaTensor):
            image_masked = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(image_masked, meta=None)
        if isinstance(mask, monai.data.meta_tensor.MetaTensor):
            mask = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(mask, meta=None)

        inference_slices = {}
        for plane, dim in self.plane_to_dimension.items():
            inference_slices[plane] = self.get_inference_slices(mask, image_masked, dim)
    
        return self.view_agg_inference(image_masked, mask, batch_size, inference_slices,
                                  num_resample_steps, num_resample_jumps, get_intermediates, scale_factor)
    


class OffsetTwoAndHalfDInpaintingInferer(TwoAndHalfDInpaintingInferer):


    def view_agg_inference(self, image_masked: torch.Tensor, mask: torch.Tensor, 
                           batch_size: int,
                           num_resample_steps: int, num_resample_jumps: int,
                           get_intermediates: bool, scale_factor=None,
                           verbose=True):
        
        # set mask to region to noise
        image_inpainted = torch.where(
                    mask == 0, image_masked, torch.randn(image_masked.shape, device=image_masked.device)
        )

        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]
        intermediates = []

        if verbose:
            progress_bar = tqdm(self.scheduler.timesteps)
        else:
            progress_bar = iter(self.scheduler.timesteps)
        
        for t in progress_bar:
            batch_outputs = []

            current_plane = self.dimension_to_plane[int(t%3)] # alternate between sagittal, axial, coronal
            self.model = self.diffusion_model_dict[current_plane] # .eval()
            batched_slices, batched_masks, batch_slice_indices, _ = self.get_inference_slices(mask, image_masked, int(t%3), 
                                                                        offset=(self.slice_thickness//2) * (t.item() % 2)) # add offset to alternate slicings

            

            # put channel dimension first 
            batched_slices = torch.swapaxes(batched_slices, 1, self.plane_to_dimension[current_plane]+1)
            batched_masks = torch.swapaxes(batched_masks, 1, self.plane_to_dimension[current_plane]+1)
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.plane_to_dimension[current_plane])

            # select region matching batches from already partially inpainted image
            image_inpainted_slice = []
            for idx in batch_slice_indices:
                image_inpainted_slice.append(image_inpainted[idx - self.slice_thickness//2:idx + self.slice_thickness//2+1])
            image_inpainted_slice = torch.stack(image_inpainted_slice, dim=0)

            # get checked image area (for logging)
            start_idx = batch_slice_indices[0] - self.slice_thickness // 2
            end_idx = batch_slice_indices[-1] + self.slice_thickness // 2 + 1

            # assert(len(batched_slices)*self.slice_thickness == end_idx - start_idx), 'batched_slices and image_inpainted_slice have different lengths'

            for i in range(math.ceil(len(batched_slices) / batch_size)): # iterate over batches
                # get current batch
                slices_batch = get_batch(batched_slices, i)
                masks_batch = get_batch(batched_masks, i)
                #slice_indices_batch = get_batch(batch_slice_indices, i)
                image_inpainted_slice_batch = get_batch(image_inpainted_slice, i)

                #print(f'Inpainting slices {start_idx} to {end_idx}, plane {current_plane}, timestep {t}')
                if verbose:
                    progress_bar.set_description(f'Inpainting slices {start_idx} to {end_idx}, plane {current_plane}, timestep {t}')
                
                d = self.denoise(t, masks_batch, slices_batch, image_inpainted_slice_batch,
                                                num_resample_steps, num_resample_jumps, scale_factor=scale_factor)
                batch_outputs.append(d)
                
            image_inpainted_slice_denoised = torch.cat([img for img in batch_outputs], dim=0)

            # unbind to remove batch dimension
            #image_inpainted[start_idx:end_idx] = torch.cat(torch.unbind(image_inpainted_slice_denoised)) 
            for i, idx in enumerate(batch_slice_indices):
                image_inpainted[idx - self.slice_thickness//2:idx + self.slice_thickness//2+1] = image_inpainted_slice_denoised[i]

            # put channel dimension back
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.plane_to_dimension[current_plane])

            if get_intermediates and t % 50 == 0:
                intermediates.append(image_inpainted.cpu())



        self.model = None # sanitize

        if get_intermediates:
            return image_inpainted, torch.stack(intermediates).unsqueeze(1)
        else:
            return image_inpainted


    def __call__(self, mask: torch.Tensor, image_masked: torch.Tensor, batch_size=1, 
                 num_resample_steps=10, num_resample_jumps=5, get_intermediates=False, scale_factor=None):

        if mask.dtype == torch.bool:
            mask = mask.int()
        else:
            assert(set(np.unique(mask)) == set((0,1))), 'mask is not binary but has values {}'.format(np.unique(mask))

        # unpack meta tensors (operating on torch tensors is faster)
        if isinstance(image_masked, monai.data.meta_tensor.MetaTensor):
            image_masked = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(image_masked, meta=None)
        if isinstance(mask, monai.data.meta_tensor.MetaTensor):
            mask = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(mask, meta=None)
    
        return self.view_agg_inference(image_masked, mask, batch_size,
                                  num_resample_steps, num_resample_jumps, get_intermediates, scale_factor)
    

class AnomalyInferer(TwoAndHalfDInpaintingInferer):


    def __call__(self, image: torch.Tensor, batch_size=1, starting_t=0,
                 num_resample_steps=10, num_resample_jumps=5, get_intermediates=False, scale_factor=None):

        # unpack meta tensors (operating on torch tensors is faster)
        if isinstance(image, monai.data.meta_tensor.MetaTensor):
            image = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(image, meta=None)

        inference_slices = {}
        for plane, dim in self.plane_to_dimension.items():
            inference_slices[plane] = self.get_inference_slices(torch.ones_like(image), image, dim)
    
        return self.view_agg_inference(image, batch_size, inference_slices, starting_t,
                                  num_resample_steps, num_resample_jumps, get_intermediates, scale_factor)


    def view_agg_inference(self, image: torch.Tensor,
                           batch_size: int, inference_slices: dict, starting_t: int,
                           num_resample_steps: int, num_resample_jumps: int,
                           get_intermediates: bool, scale_factor=None,
                           verbose=True):
        
        # set mask to region to noise
        image_denoised = self.sample_forward_diffusion(image, torch.tensor(starting_t))
        
        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]
        intermediates = []

        timesteps = self.scheduler.timesteps[len(self.scheduler.timesteps)-starting_t-1:]

        if verbose:
            progress_bar = tqdm(timesteps)
        else:
            progress_bar = iter(timesteps)
        
        for t in progress_bar:
            batch_outputs = []

            current_plane = self.dimension_to_plane[int(t%3)] # alternate between sagittal, axial, coronal
            self.model = self.diffusion_model_dict[current_plane] #.eval()
            batched_slices, batched_masks, batch_slice_indices, _ = inference_slices[current_plane]

            # put channel dimension first 
            batched_slices = torch.swapaxes(batched_slices, 1, self.plane_to_dimension[current_plane]+1)
            image_denoised = torch.swapaxes(image_denoised, 0, self.plane_to_dimension[current_plane])

            # select region matching batches from already partially inpainted image
            image_denoised_slice = []
            for idx in batch_slice_indices:
                image_denoised_slice.append(image_denoised[idx - self.slice_thickness//2:idx + self.slice_thickness//2+1])
            image_denoised_slice = torch.stack(image_denoised_slice, dim=0)

            # get checked image area
            start_idx = batch_slice_indices[0] - self.slice_thickness // 2
            end_idx = batch_slice_indices[-1] + self.slice_thickness // 2 + 1

            for i in range(math.ceil(len(batched_slices) / batch_size)): # iterate over batches
                # get current batch
                image_denoised_slice_batch = get_batch(image_denoised_slice, i)

                if verbose:
                    progress_bar.set_description(f'Denoising slices {start_idx} to {end_idx}, plane {current_plane}, timestep {t}')
                
                d = self.denoise(t, image_denoised_slice_batch,
                                                num_resample_steps, num_resample_jumps, scale_factor=scale_factor)
                batch_outputs.append(d)
                
            image_inpainted_slice_denoised = torch.cat([img for img in batch_outputs], dim=0)
            # unbind to remove batch dimension
            image_denoised[start_idx:end_idx] = torch.cat(torch.unbind(image_inpainted_slice_denoised)) 

            # put channel dimension back
            image_denoised = torch.swapaxes(image_denoised, 0, self.plane_to_dimension[current_plane])

            if get_intermediates and t % (len(timesteps) // 10) == 0: # save 30 intermediate outputs
                intermediates.append(image_denoised.cpu())



        self.model = None # sanitize

        if get_intermediates:
            return image_denoised, torch.stack(intermediates).unsqueeze(1)
        else:
            return image_denoised
        

    
    @torch.no_grad()
    def denoise(self, t, image: torch.Tensor,
                 num_resample_steps=10, num_resample_jumps=5, scale_factor=None):    
        """
        Run inference on `inputs` with the `network` model.

        Args:
            inputs: input of the model inference.
            network: model for inference.
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.
        """

        image_denoised = image

    
        last_step = t == 0

        # skip resampling at last step and do jumps
        if last_step or t % num_resample_jumps != 0: samplings_at_cuttent_t = 1
        else: samplings_at_cuttent_t = num_resample_steps

        for u in range(samplings_at_cuttent_t):
            last_sample_step = u == (samplings_at_cuttent_t - 1)

            # perform a denoising step to get the unknown region at t (backward diffusion to get t-1)
            image_denoised = self.diffusion_backward(image_denoised, t, sf=scale_factor)

            # if we are doing resampling add noise and go again
            if not last_sample_step: 
                # perform resampling (forward & backward)
                image_denoised = self.diffusion_forward(image_denoised, t-1)

        return image_denoised




class DiffusionInfererVINN(DiffusionInferer):
    
    def __call__(
        self,
        inputs: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        scale_factors: torch.Tensor = None,
        condition = None,
        mode: str = "crossattn",
        
    ) -> torch.Tensor:
        """
        Implements the forward pass for a supervised training iteration.

        Args:
            inputs: Input image to which noise is added.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the input.
            timesteps: random timesteps.
            condition: Conditioning for network input.
            mode: Conditioning mode for the network.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps.long())
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None

        #if scale_factors is None:
        #    prediction = diffusion_model(x=noisy_image, timesteps=timesteps, context=condition)
        #else:
        prediction = diffusion_model(x=noisy_image, scale_factors=scale_factors, timesteps=timesteps, context=condition)
            

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise: torch.Tensor,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler,
        scale_factors: torch.Tensor = None,
        save_intermediates: bool = False,
        intermediate_steps: int = 100,
        conditioning: torch.Tensor = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []

        kwargs = {}
        if scale_factors is not None:
            kwargs["scale_factors"] = scale_factors
        if mode == "concat" and conditioning is not None:
            model_input = torch.cat([image, conditioning], dim=1)
            kwargs["context"] = None
        elif mode == "crossattn" and conditioning is not None:
            kwargs["context"] = conditioning
            model_input = image
        else:
            kwargs["context"] = None
            model_input = image

        for t in progress_bar:
            # 1. predict noise model_output
            model_output = diffusion_model(
                model_input, 
                timesteps=torch.Tensor((t,), device=torch.device('cpu')), #input_noise.device), 
                **kwargs
            )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image
        

    @torch.no_grad()
    def sample_backward_forward(
        self,
        input_noise: torch.Tensor,
        precond_img: torch.Tensor,
        t_start: int,
        diffusion_model: Callable[..., torch.Tensor],
        scheduler,
        scale_factors: torch.Tensor = None,
        save_intermediates: bool = False,
        intermediate_steps: int = 100,
        conditioning: torch.Tensor = None,
        mode: str = "crossattn",
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            mode: Conditioning mode for the network.
            verbose: if true, prints the progression bar of the sampling process.
        """
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        intermediates = []

        kwargs = {}
        if scale_factors is not None:
            kwargs["scale_factors"] = scale_factors
        if mode == "concat" and conditioning is not None:
            model_input = torch.cat([image, conditioning], dim=1)
            kwargs["context"] = None
        elif mode == "crossattn" and conditioning is not None:
            kwargs["context"] = conditioning
            model_input = image
        else:
            kwargs["context"] = None
            model_input = image

        # noise preconditioning image to t_start
        image = scheduler.add_noise(original_samples=precond_img, noise=image, timesteps=torch.Tensor((t_start-1,), device=input_noise.device).long())
        if verbose:
            progress_bar = tqdm(scheduler.timesteps[:-t_start])
        else:
            progress_bar = iter(scheduler.timesteps[:-t_start])

        for t in progress_bar:
            # 1. predict noise model_output
            model_output = diffusion_model(
                model_input, 
                timesteps=torch.Tensor((t,), device=input_noise.device), 
                **kwargs
            )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image
