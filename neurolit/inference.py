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


import math
import warnings
from collections.abc import Callable

# Suppress FutureWarning about deprecated cuda.cudart module
# Must be before torch/monai imports to catch the warning during import
warnings.filterwarnings("ignore", category=FutureWarning, message=".*cuda.cudart.*")  # noqa: E402

import monai  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from monai.inferers import DiffusionInferer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from neurolit.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


class InpaintingInferer:
    """Coordinate diffusion-based inpainting iterations.

    This class manages the inference process for diffusion-based inpainting,
    coordinating forward and backward diffusion steps to fill in masked regions.

    Parameters
    ----------
    inference_steps : int
        Number of denoising timesteps to execute.
    scheduler : monai.inferers.DiffusionInferer
        Scheduler that defines the diffusion timestep sequence.
    diffusion_model : torch.nn.Module
        Model used to predict noise residuals at each timestep.
    """

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
        """Inpaint masked regions by alternating forward and backward diffusion.

        This method performs diffusion-based inpainting by iteratively denoising
        the image while preserving known regions defined by the mask.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor where zeros indicate known voxels.
        image_masked : torch.Tensor
            Image tensor with masked regions that need inpainting.
        num_resample_steps : int, optional
            Number of resampling loops per timestep, by default 10.
        num_resample_jumps : int, optional
            Number of timesteps to skip before resampling, by default 5.
        get_intermediates : bool, optional
            Whether to record intermediate outputs, by default False.
        scale_factor : Any, optional
            Optional scaling factors passed to the diffusion model.
        *args : Any
            Additional positional arguments forwarded to downstream calls.
        **kwargs : Any
            Additional keyword arguments forwarded to downstream calls.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            Denoised tensor, optionally paired with intermediate buffers.
        """
        assert(set(np.unique(mask)) == set((0,1))), f'mask is not binary but has values {np.unique(mask)}'
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

    def sample_forward_diffusion(self, image, t): # add noise at 
        """Add noise to `image` at timestep `t`.

        Parameters
        ----------
        image : torch.Tensor
            Tensor to perturb for forward diffusion.
        t : int | torch.Tensor
            Current timestep identifier.

        Returns
        -------
        torch.Tensor
            Noised tensor for the current timestep.
        """
        noise = torch.randn((image.shape), device=image.device)
        # sqrt(alpha) * sample + (1-sqrt(alpha)) * noise
        noised_image = self.scheduler.add_noise(original_samples=image, noise=noise, timesteps=t)
        return noised_image
    
    def diffusion_forward(self, image, t):
        """Apply a single forward diffusion update using the scheduler betas.

        Parameters
        ----------
        image : torch.Tensor
            Current reconstruction tensor.
        t : int | torch.Tensor
            Current timestep index.

        Returns
        -------
        torch.Tensor
            Tensor after applying forward diffusion noise.
        """
        noise = torch.randn((image.shape),device=image.device)
        # sqrt(1-beta) * image + sqrt(beta) * noise
        image_inpainted = (torch.sqrt(1 - self.scheduler.betas[t]) * image + torch.sqrt(self.scheduler.betas[t]) * noise)

        return image_inpainted


    def diffusion_backward(self, image, t, sf=None): # TODO: add jumps
        """Denoise `image` at timestep `t` with the diffusion model prediction.

        Parameters
        ----------
        image : torch.Tensor
            Tensor to denoise.
        t : torch.Tensor
            Timestep tensor provided to the scheduler.
        sf : Any, optional
            Optional scale factors forwarded to the diffusion model, by default None.

        Returns
        -------
        torch.Tensor
            Reconstruction for timestep ``t-1``.
        """
        # the model outputs the noise
        model_output = self.model(image, scale_factors=sf, timesteps=torch.tensor((t,),device=image.device).long())
        # use model output to denoise and add noise again
        image_inpainted_prev_unknown, _ = self.scheduler.step(model_output, t.item(), image)

        return image_inpainted_prev_unknown


    # def resample(self, image, t, sf): # forward and back
    #     image_forward = self.diffusion_forward(image, t)
    #     image_backward = self.diffusion_backward(image_forward, t+1, sf)
    #     return image_backward



class SliceWiseInpaintingInferer(InpaintingInferer):

    """Process the volume one slice batch at a time along a fixed axis.

    Parameters
    ----------
    dimensions : int
        Dimension index along which to extract slices.
    diffusion_model : torch.nn.Module
        Model that predicts diffusion noise for each slice.
    scheduler : monai.inferers.DiffusionInferer
        Scheduler controlling the timestep sequence.
    inference_steps : int
        Number of diffusion timesteps to run for each slice.
    """

    def __init__(self, dimensions, diffusion_model, scheduler, inference_steps):
        super().__init__(inference_steps, scheduler, diffusion_model)
        self.dimension = dimensions
        self.slice_thickness = diffusion_model.in_channels

    def get_slice_from_volume(self, volume, slice_cut, dimension): # TODO: make sure channel dimension is always first
        """Extract a slab centered at `slice_cut`.

        Parameters
        ----------
        volume : torch.Tensor
            Volume tensor to sample from.
        slice_cut : int
            Center index of the desired slice block.
        dimension : int
            Spatial dimension to slice along.

        Returns
        -------
        torch.Tensor
            Extracted slice tensor with thickness matching the model channels.
        """
        threed_to_twod_slice = [slice(None), slice(None), slice(None)]
        threed_to_twod_slice[dimension] = slice(slice_cut-self.slice_thickness//2, slice_cut+self.slice_thickness//2+1)
        threed_to_twod_slice = tuple(threed_to_twod_slice)

        volume = volume[threed_to_twod_slice]
        return volume
    
    @staticmethod
    def slice_selector(start_idx, end_idx, dimension):
        """Build a slab tuple for the requested range.

        Parameters
        ----------
        start_idx : int
            Starting slice index (inclusive).
        end_idx : int
            Ending slice index (exclusive).
        dimension : int
            Spatial axis to apply the slice.

        Returns
        -------
        tuple[slice]
            Tuple that can be used to index the volume.
        """
        selected_slice = [slice(None), slice(None), slice(None)]
        selected_slice[dimension] = slice(start_idx, end_idx)
        selected_slice = tuple(selected_slice)
        return selected_slice


    
    def get_inference_slices(self, mask, image_masked, dimension, offset=0):
        """Collect slice batches and indices for inference along `dimension`.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor for the entire volume.
        image_masked : torch.Tensor
            Volume tensor with masked regions to inpaint.
        dimension : int
            Spatial axis for extracting slices.
        offset : int, optional
            Offset applied to avoid checkerboard artifacts, by default 0.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]
            Batched slices, batched masks, the slice centers, and the pre-populated output tensor.
        """
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
        """Inpaint the volume slice-wise.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask volume indicating known voxels.
        image_masked : torch.Tensor
            Prefiltered volume with masked regions to inpaint.
        batch_size : int, optional
            Number of slices to process per batch, by default 1.
        num_resample_steps : int, optional
            Resampling loops per timestep, by default 10.
        num_resample_jumps : int, optional
            Timesteps between resampling events, by default 5.
        get_intermediates : bool, optional
            Whether to collect intermediate reconstructions, by default False.
        scale_factor : Any, optional
            Scale factors forwarded to the diffusion model, by default None.
        *args : Any
            Additional positional arguments forwarded to the parent class.
        **kwargs : Any
            Additional keyword arguments forwarded to the parent class.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Reconstructed volume, optionally with intermediate slices.
        """
        intermediates = []

        batched_slices, batched_masks, batch_slice_indices, image_inpainted = self.get_inference_slices(mask, image_masked, self.dimension)

        
        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]

        for i in range(math.ceil(len(batched_slices) / batch_size)):

            slices_batch = get_batch(batched_slices, i)
            masks_batch = get_batch(batched_masks, i)
            slice_indices_batch = get_batch(batch_slice_indices, i)

            start_idx = slice_indices_batch[0] - self.slice_thickness // 2
            end_idx = slice_indices_batch[-1] + self.slice_thickness // 2 + 1

            logger.info(f'Inpainting slices {start_idx} to {end_idx}')

            # put channel dimension first 
            slices_batch = torch.swapaxes(slices_batch, 1, self.dimension+1)
            masks_batch = torch.swapaxes(masks_batch, 1, self.dimension+1)
            image_inpainted = torch.swapaxes(image_inpainted, 0, self.dimension)
            
            image_inpainted_slice = super().__call__(masks_batch, slices_batch, 
                                                    num_resample_steps, num_resample_jumps, 
                                                    *args, get_intermediates=get_intermediates, scale_factor=scale_factor, 
                                                    **kwargs)
            
            

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

    """Aggregate slice-wise inpainting across axial, sagittal, and coronal views.

    Parameters
    ----------
    diffusion_model_dict : dict[str, torch.nn.Module]
        Mapping from plane names to their diffusion models.
    scheduler : monai.inferers.DiffusionInferer
        Scheduler controlling the diffusion timesteps.
    inference_steps : int
        Number of diffusion steps to perform per slice.
    """

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
        """Inpaint the volume by switching views and offsets.

        Parameters
        ----------
        image_masked : torch.Tensor
            Input tensor with masked regions to reconstruct.
        mask : torch.Tensor
            Binary mask tensor indicating the known region.
        batch_size : int
            Number of slices to process per batch.
        inference_slices : dict
            Precomputed slice batches and masks for each plane.
        num_resample_steps : int
            Number of resampling loops per timestep.
        num_resample_jumps : int
            Number of timesteps to skip before each resampling.
        get_intermediates : bool
            Whether to collect intermediate reconstructions.
        scale_factor : Any, optional
            Scale factors passed to the diffusion model, by default None.
        verbose : bool, optional
            Whether to show a progress bar, by default True.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Reconstructed volume or tuple with collected intermediates if requested.
        """
        #image_inpainted = torch.randn(image_masked.shape, device=image_masked.device)

        # set mask to region to noise
        image_inpainted = torch.where(
                    mask == 0, image_masked, torch.randn(image_masked.shape, device=image_masked.device)
        )

        #plot_batch(image_inpainted.unsqueeze(0), 
        # '/groups/ag-reuter/projects/fastsurfer-tumor/monai_generative/diffusion_inpainting/experiments/inference_viewagg/images/initial_noised.png',
        # slice_cut=[100, 84, 140])

        get_batch = lambda x, i: x[i*batch_size:(i+1)*batch_size]
        intermediates = []

        if verbose:
            progress_bar = tqdm(self.scheduler.timesteps)
        else:
            progress_bar = iter(self.scheduler.timesteps)
        
        for t in progress_bar:
            batch_outputs = []

            current_plane = self.dimension_to_plane[int(t%3)] # alternate between sagittal, axial, coronal
            # TODO: this may not be needed and might cause overhead (same for other eval calls)
            self.model = self.diffusion_model_dict[current_plane] #.eval() 
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
        """Refine the unknown regions through backward and forward diffusion.

        Parameters
        ----------
        t : torch.Tensor
            Current timestep identifier.
        mask : torch.Tensor
            Binary mask delineating known voxels.
        image_masked : torch.Tensor
            Reference tensor containing the known region.
        image_inpainted : torch.Tensor
            Current reconstruction that is being denoised.
        num_resample_steps : int, optional
            Number of resampling loops applied per timestep.
        num_resample_jumps : int, optional
            Number of timesteps to jump before a resampling iteration.
        scale_factor : Any, optional
            Optional scaling factors provided to the diffusion model.

        Returns
        -------
        torch.Tensor
            Tensor after the current denoising iteration.
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

        """Prepare slice batches for all planes and run view-aggregated inference.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor that highlights the unknown regions.
        image_masked : torch.Tensor
            Volume tensor with the known regions preserved.
        batch_size : int, optional
            Batch size for slice inference, by default 1.
        num_resample_steps : int, optional
            Resampling iterations per timestep, by default 10.
        num_resample_jumps : int, optional
            Timesteps between resampling rounds, by default 5.
        get_intermediates : bool, optional
            Whether to collect intermediate outputs, by default False.
        scale_factor : Any, optional
            Optional scaling factors forwarded to the diffusion models.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Reconstructed volume, optionally paired with intermediates.
        """
        if mask.dtype == torch.bool:
            mask = mask.int()
        else:
            assert(set(np.unique(mask)) == set((0,1))), f'mask is not binary but has values {np.unique(mask)}'

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

    """Two-and-a-half dimensional inferer that alternates slicing offsets.

    Parameters
    ----------
    diffusion_model_dict : dict[str, torch.nn.Module]
        Mapping from plane names to their diffusion models.
    scheduler : monai.inferers.DiffusionInferer
        Scheduler controlling the diffusion timesteps.
    inference_steps : int
        Number of timesteps per slice inference.
    """

    def view_agg_inference(self, image_masked: torch.Tensor, mask: torch.Tensor, 
                           batch_size: int,
                           num_resample_steps: int, num_resample_jumps: int,
                           get_intermediates: bool, scale_factor=None,
                           verbose=True):
        """Inpaint the volume using offset slicing to avoid checkerboarding.

        Parameters
        ----------
        image_masked : torch.Tensor
            Input tensor with masked regions to reconstruct.
        mask : torch.Tensor
            Binary mask tensor indicating the known region.
        batch_size : int
            Number of slices processed per batch.
        num_resample_steps : int
            Number of resampling loops per timestep.
        num_resample_jumps : int
            Timesteps between resampling events.
        get_intermediates : bool
            Whether to collect intermediate outputs.
        scale_factor : Any, optional
            Optional scaling factors for the diffusion model.
        verbose : bool, optional
            Whether progress information is displayed.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Reconstructed volume, optionally paired with intermediates.
        """
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
            # add offset to alternate slicings to avoid checkerboard pattern
            batched_slices, batched_masks, batch_slice_indices, _ = self.get_inference_slices(mask, image_masked, int(t%3), 
                                                                        offset=(self.slice_thickness//2) * (t.item() % 2)) 

            

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

        """Prepare slices with alternating offsets and perform view aggregation.

        Parameters
        ----------
        mask : torch.Tensor
            Binary mask tensor identifying regions to reconstruct.
        image_masked : torch.Tensor
            Image tensor with masked regions preserved.
        batch_size : int, optional
            Number of slices per batch, by default 1.
        num_resample_steps : int, optional
            Resampling loops per timestep, by default 10.
        num_resample_jumps : int, optional
            Timesteps between resampling events, by default 5.
        get_intermediates : bool, optional
            Whether to collect intermediate outputs, by default False.
        scale_factor : Any, optional
            Optional scaling factors passed to the diffusion model.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Reconstructed volume, optionally with intermediates.
        """

        if mask.dtype == torch.bool:
            mask = mask.int()
        else:
            assert(set(np.unique(mask)) == set((0,1))), f'mask is not binary but has values {np.unique(mask)}'

        # unpack meta tensors (operating on torch tensors is faster)
        if isinstance(image_masked, monai.data.meta_tensor.MetaTensor):
            image_masked = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(image_masked, meta=None)
        if isinstance(mask, monai.data.meta_tensor.MetaTensor):
            mask = monai.data.meta_tensor.MetaTensor.ensure_torch_and_prune_meta(mask, meta=None)
    
        return self.view_agg_inference(image_masked, mask, batch_size,
                                  num_resample_steps, num_resample_jumps, get_intermediates, scale_factor)
    

class AnomalyInferer(TwoAndHalfDInpaintingInferer):

    """Detect anomalies by detecting deviations from the expected noise distribution.

    Parameters
    ----------
    diffusion_model_dict : dict[str, torch.nn.Module]
        Mapping from plane names to their diffusion models.
    scheduler : monai.inferers.DiffusionInferer
        Scheduler controlling the diffusion timesteps.
    inference_steps : int
        Number of timesteps to run per reconstruction.
    """


    def __call__(self, image: torch.Tensor, batch_size=1, starting_t=0,
                 num_resample_steps=10, num_resample_jumps=5, get_intermediates=False, scale_factor=None):
        """Generate anomaly maps by denoising noise-injected copies of `image`.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor used as a reference for anomalous regions.
        batch_size : int, optional
            Batch size for slice inference, by default 1.
        starting_t : int, optional
            Starting timestep for denoising, by default 0.
        num_resample_steps : int, optional
            Resampling loops per timestep, by default 10.
        num_resample_jumps : int, optional
            Timesteps between resampling rounds, by default 5.
        get_intermediates : bool, optional
            Whether to collect intermediate outputs, by default False.
        scale_factor : Any, optional
            Optional scaling factors forwarded to the diffusion model.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Denoised tensor, optionally paired with intermediate samples.
        """

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
        """Denoise the volume starting from `starting_t` for anomaly detection.

        Parameters
        ----------
        image : torch.Tensor
            Image tensor to initialize the denoising process.
        batch_size : int
            Number of slices processed per batch.
        inference_slices : dict
            Precomputed slices for each anatomical plane.
        starting_t : int
            Timestep index from which denoising begins.
        num_resample_steps : int
            Number of resampling loops per timestep.
        num_resample_jumps : int
            Timesteps between resampling rounds.
        get_intermediates : bool
            Whether to collect intermediate outputs.
        scale_factor : Any, optional
            Optional scale factors, by default None.
        verbose : bool, optional
            Whether to show progress updates.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            Denoised volume or tuple with intermediates when requested.
        """
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
        """Denoise `image` for `starting_t` forward iterations.

        Parameters
        ----------
        t : torch.Tensor
            Current timestep identifier.
        image : torch.Tensor
            Tensor being denoised.
        num_resample_steps : int, optional
            Number of resampling loops per timestep.
        num_resample_jumps : int, optional
            Timesteps between resampling events.
        scale_factor : Any, optional
            Optional scaling factors for the diffusion model.

        Returns
        -------
        torch.Tensor
            Tensor after the denoising iteration.
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

    """Sampler that extends MONAI's DiffusionInferer to support VINN conditioning.

    The implementation handles VINN scale factors, optional conditioning modes, and
    exposes helper sampling routines used during training and inference.
    """
    
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
        """Implement the VINN forward pass for a training iteration.

        Parameters
        ----------
        inputs : torch.Tensor
            Input image tensor to corrupt.
        diffusion_model : Callable[..., torch.Tensor]
            Model predicting the noise residual.
        noise : torch.Tensor
            Random noise tensor with the same shape as ``inputs``.
        timesteps : torch.Tensor
            Timestep identifiers for the scheduler.
        scale_factors : torch.Tensor, optional
            Optional VINN scale factors to condition the model.
        condition : Any, optional
            Conditioning tensor provided to the diffusion model.
        mode : str, optional
            Conditioning mode, either ``crossattn`` or ``concat``.

        Returns
        -------
        torch.Tensor
            Predicted noise residuals at the given timesteps.
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
        """Sample from the VINN diffusion model using the provided scheduler.

        Parameters
        ----------
        input_noise : torch.Tensor
            Random noise tensor with the shape of the desired sample.
        diffusion_model : Callable[..., torch.Tensor]
            Model used for sampling.
        scheduler : Any
            Diffusion scheduler; if ``None`` uses the inferer's scheduler.
        scale_factors : torch.Tensor, optional
            Optional scale factors passed to the diffusion model.
        save_intermediates : bool, optional
            Whether to return intermediate tensors, by default False.
        intermediate_steps : int, optional
            Interval between saved intermediates when ``save_intermediates`` is True.
        conditioning : torch.Tensor, optional
            Optional conditioning tensor for the diffusion model.
        mode : str, optional
            Conditioning mode, either ``crossattn`` or ``concat``.
        verbose : bool, optional
            Whether to show a progress bar during sampling.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            Sampled tensor, optionally paired with accumulated intermediates.
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
        """Precondition the volume and then sample backward and forward with VINN.

        Parameters
        ----------
        input_noise : torch.Tensor
            Random noise tensor similar to the desired output shape.
        precond_img : torch.Tensor
            Image used to precondition the schedule before sampling begins.
        t_start : int
            Timestep index to start the backward-forward sampling from.
        diffusion_model : Callable[..., torch.Tensor]
            VINN diffusion model used for prediction.
        scheduler : Any
            Scheduler to step through diffusion timesteps.
        scale_factors : torch.Tensor, optional
            Optional scaling factors for the diffusion model.
        save_intermediates : bool, optional
            Whether to return intermediate tensors, by default False.
        intermediate_steps : int, optional
            Interval between recorded intermediates when enabled.
        conditioning : torch.Tensor, optional
            Optional conditioning tensor for the diffusion model.
        mode : str, optional
            Conditioning mode, either ``crossattn`` or ``concat``.
        verbose : bool, optional
            Whether to display progress.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            Sampled tensor, optionally paired with saved intermediates.
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
