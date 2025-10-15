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

import os
from typing import Union, List, Any

import torch
import numpy as np
import matplotlib.pyplot as plt





def plot_batch(image_batch: torch.Tensor, image_path: str, slice_cut: Union[None, List[int]] = None) -> None:
    """Plot a batch of 2D or 3D images with different slice views.
    
    Args:
        image_batch: Input batch of images to plot. Shape should be:
            - For 3D: (batch_size, channels, H, W, D)
            - For 2D: (batch_size, channels, H, W)
        image_path: Path where to save the output plot
        slice_cut: List of indices where to cut the slices. If None, uses middle slice.
            For 3D: [x, y, z] indices
            For 2D: [x, y] indices
    """
    dim = len(image_batch.shape) - 2
    BATCH_SIZE = image_batch.shape[0]

    if slice_cut is None:
        slice_cut = image_batch.shape[3] // 2

    if dim == 3:
        slices = [[(i, 0, slice(None), slice(None), slice_cut[2]) for i in range(BATCH_SIZE)],
                  [(i, 0, slice(None), slice_cut[1], slice(None)) for i in range(BATCH_SIZE)],
                  [(i, 0, slice_cut[0], slice(None), slice(None)) for i in range(BATCH_SIZE)]]
    elif dim == 2:
        middle_slice = image_batch.shape[1] // 2
        slices = [[(i, middle_slice, slice(None), slice(None)) for i in range(BATCH_SIZE)],
                  [(i, slice(None), slice_cut[0], slice(None)) for i in range(BATCH_SIZE)],
                  [(i, slice(None), slice(None), slice_cut[1]) for i in range(BATCH_SIZE)]]
    else:
        raise ValueError("Invalid dimension")

    fig, axs = plt.subplots(3, BATCH_SIZE, figsize=(1.5*BATCH_SIZE, 3))

    if BATCH_SIZE == 1:
        axs = np.expand_dims(axs, axis=1)

    for s, ax in zip(slices, axs):
        for i in range(BATCH_SIZE):
            ax[i].imshow(image_batch[s[i]].detach().cpu(), vmin=0, vmax=1, cmap="gray")
            #plt.axis("off")
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    plt.tight_layout()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_curve(n_epochs: int, epoch_loss_list: List[float], 
                       val_epoch_loss_list: List[float], args: Any, 
                       VAL_INTERVAL: int) -> None:
    """Plot training and validation learning curves.
    
    Args:
        n_epochs: Total number of epochs
        epoch_loss_list: List of training losses for each epoch
        val_epoch_loss_list: List of validation losses (recorded every VAL_INTERVAL)
        args: Arguments object containing output directory information
        VAL_INTERVAL: Interval at which validation was performed
    """
    plt.style.use("seaborn-v0_8")
    plt.title("Learning Curves", fontsize=20)
    plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
    plt.plot(
        np.linspace(VAL_INTERVAL, n_epochs, int(n_epochs / VAL_INTERVAL)),
        val_epoch_loss_list,
        color="C1",
        linewidth=2.0,
        label="Validation",
    )
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.legend(prop={"size": 14})
    # log scale
    plt.yscale("log")
    plt.savefig(os.path.join(args.out_dir,'images/learning_curves.png'), dpi=100, bbox_inches='tight')
    plt.close()

def plot_scheduler(scheduler: Any, output_file: str) -> None:
    """Plot the scheduler's alpha values over time.
    
    Args:
        scheduler: Diffusion scheduler object containing alphas_cumprod
        output_file: Path where to save the output plot
    """
    plt.plot(scheduler.alphas_cumprod.cpu(), color=(2 / 255, 163 / 255, 163 / 255), linewidth=2)
    plt.xlabel("Timestep [t]")
    plt.ylabel("alpha cumprod")
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()

def plot_inpainting(val_image: torch.Tensor, val_image_masked: torch.Tensor, 
                   val_image_inpainted: torch.Tensor, out_file: str,
                   SLICE_CUT: torch.Tensor, cut_dim: int = 0) -> None:
    """Plot original, masked, and inpainted images side by side.
    
    Args:
        val_image: Original image tensor
        val_image_masked: Masked version of the image
        val_image_inpainted: Inpainted result
        out_file: Path where to save the output plot
        SLICE_CUT: Index where to cut the slice for visualization
        cut_dim: Dimension along which to take the slice (0=sagittal, 1=coronal, 2=axial)
    """
    if len(val_image.shape) == 5:
        val_image = val_image[:, 0, ...]
        val_image_masked = val_image_masked[:, 0, ...]
        val_image_inpainted = val_image_inpainted[:, 0, ...]

    slicing = [0, slice(None), slice(None), slice(None)]
    slicing[cut_dim+1] = SLICE_CUT[cut_dim].item()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(val_image[slicing].cpu(), cmap="gray")
    axs[0].set_title("Original image")
    axs[0].axis("off")
    axs[1].imshow(val_image_masked[slicing].cpu(), cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Masked image")
    axs[2].imshow(val_image_inpainted[slicing].cpu(), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Inpainted image")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()