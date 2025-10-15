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
# =========================================================================
# Adapted from 
# https://github.com/Project-MONAI/GenerativeModels
# which has the following license:
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse

import torch
import torch.nn.functional as F
import monai
from monai.data import DataLoader
from monai import transforms
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.networks.schedulers import DDPMScheduler
from LIT.networks.DiffusionUnet import DiffusionModelUNetVINN
from LIT.inference import DiffusionInfererVINN
from LIT.utils import plot_batch, plot_scheduler
from LIT.data.datasets import get_base_dataset, SlicedDataset


def argument_parser():
    parser = argparse.ArgumentParser(description='Train a 3D DDPM model')
    parser.add_argument('--out_dir', type=str, default='debug_run', help='experiment output directory')
    parser.add_argument('--slice_dim', type=str, default=None, help='slice dimension for 2D training')
    parser.add_argument('--no_vinn', default=False, help='use VINN', action='store_true')
    return parser.parse_args()


def get_transforms(IMAGE_SHAPE, PATCH_DATASET, FIXED_SIZE, ISVINN=True):
    tr = [
        transforms.LoadImaged(keys=['image'],reader="nibabelreader", image_only=True, dtype=torch.float16, ensure_channel_first=True),
        transforms.ScaleIntensityd(keys=['image']),
    ]
    if PATCH_DATASET: tr.append(transforms.RandSpatialCropd(keys=["image"], roi_size=IMAGE_SHAPE, random_size=False, random_center=True))#Identityd(keys=['image'])
    if FIXED_SIZE: tr.append(transforms.Resized(keys=['image'],spatial_size=IMAGE_SHAPE))

    return transforms.Compose(tr)

if __name__ == "__main__":
    set_determinism(42)
    
    # network
    s = 256
    IMAGE_SHAPE = (s, s, s)
    INTERNAL_SHAPE = (128,128) # only applicable for VINN
    FP16 = False
    TWO_D = True
    SLICE_THICKNESS = 7
    NUM_REVERSE_DIFFUSION_STEPS = 30
    
    # data
    DUMMY_DATA = False
    DATASET_SIZE = 'big' # one of ['big', 'small', 'standard']
    PATCH_DATASET = False
    FIXED_SIZE = False

    # training
    BATCH_SIZE = 30 # original batch size: 44 
    EPOCHS = 50
    DATA_PARALLEL = True
    VAL_INTERVAL = 5 # = save interval

    # logging
    TQDM_ENABLED = True
    SLICE_CUT = (IMAGE_SHAPE[0] // 2, IMAGE_SHAPE[1] // 2, IMAGE_SHAPE[2] // 2)

    device = torch.device("cuda")

    args = argument_parser()

    IS_VINN = not args.no_vinn
    print('Using VINN:', IS_VINN)
    SLICING_DIMENSION = args.slice_dim
    print('Using slicing dimension:', SLICING_DIMENSION)
    
    if SLICING_DIMENSION == 'sagittal':
        SLICING_DIMENSION = 0
    elif SLICING_DIMENSION == 'axial':
        SLICING_DIMENSION = 1
    elif SLICING_DIMENSION == 'coronal':
        SLICING_DIMENSION = 2

    if IS_VINN:
        internal_res_mm = 256 / INTERNAL_SHAPE[0]

        def get_scale_factors(images):
            if not FIXED_SIZE:
                sf = internal_res_mm / images.meta['delta']
                if sf.dim() == 1:
                    sf = sf.unsqueeze(0)
                # pop DIM if TWO_D
                if TWO_D and SLICING_DIMENSION == 0: sf = sf[:, 1:]
                elif TWO_D and SLICING_DIMENSION == 1: sf = sf[:, [0, 2]]
                elif TWO_D and SLICING_DIMENSION == 2: sf = sf[:, :-1]
                return sf
            else:
                return torch.ones((images.shape[0], 2 if TWO_D else 3)) # 1mm resolution with IMAGE_SHAPE size
            


    model = DiffusionModelUNetVINN(
        spatial_dims=3 if not TWO_D else 2,
        internal_size=INTERNAL_SHAPE,
        in_channels=1 if not TWO_D else SLICE_THICKNESS,
        out_channels=1 if not TWO_D else SLICE_THICKNESS,
        num_channels=[4, 16, 32, 64]                 if not TWO_D else [128, 256, 512],#[256, 256, 512],
        attention_levels=[False, False, False, True] if not TWO_D else [False, False, True],
        num_head_channels=[0, 0, 0, 64]              if not TWO_D else [0, 0, 512],
        num_res_blocks=2,
        norm_num_groups=4,
        use_fp16_VINN=FP16,
        is_vinn=IS_VINN,
        interpolation_mode='trilinear' if not TWO_D else 'bilinear',
    )

    assert(not (BATCH_SIZE == 1 and DATA_PARALLEL)), 'BATCH_SIZE == 1 and DATA_PARALLEL is not supported'


    # create datasets
    trs = get_transforms(IMAGE_SHAPE, PATCH_DATASET, FIXED_SIZE, ISVINN=IS_VINN)
    train_dataset, val_dataset = get_base_dataset(DATASET_SIZE, trs)

    if TWO_D: # make slices
        train_dataset = SlicedDataset(train_dataset, thickness=SLICE_THICKNESS, ax=SLICING_DIMENSION)#, slice_per_img=IMAGE_SHAPE[0]//SLICE_THICKNESS)
        val_dataset = SlicedDataset(val_dataset, thickness=SLICE_THICKNESS, ax=SLICING_DIMENSION)

    N_WORKERS = BATCH_SIZE if BATCH_SIZE <= 32 else 32
    collate_fn = lambda x: monai.data.pad_list_data_collate(x, mode='constant', constant_values=0, method='end')#, dtype=torch.float32)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, persistent_workers=True, prefetch_factor=2, shuffle=True, collate_fn=collate_fn) # drop_last=True
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, persistent_workers=True, prefetch_factor=2, shuffle=False, collate_fn=collate_fn) # drop_last=True

    print('Train image shape:', next(iter(train_loader))['image'].shape)
    print('Val image shape:', next(iter(val_loader))['image'].shape)



    # setup logging
    if not os.path.exists(os.path.join(args.out_dir,'images')):        
        print('creating output directory:', os.path.join(args.out_dir, 'images'))
        os.makedirs(os.path.join(args.out_dir,'images'))
        print(args.out_dir)

    # plot input images
    check_data = monai.utils.misc.first(train_loader)
    plot_batch(check_data['image'], image_path=os.path.join(args.out_dir,'images/input_images.png'), slice_cut=SLICE_CUT)

    # use multi-gpu
    model = model.to(device)
    if DATA_PARALLEL:
        print('Using {} GPUs'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model,device_ids=range(torch.cuda.device_count()))#, device_ids=[0, 1, 2, 3])

    scheduler = DDPMScheduler(num_train_timesteps=NUM_REVERSE_DIFFUSION_STEPS, schedule="scaled_linear_beta", beta_start=0.0005, beta_end=0.0195)

    #plot_scheduler(scheduler, output_file=os.path.join(args.out_dir,'images/scheduler_alpha_cumprod.png'))


    inferer = DiffusionInfererVINN(scheduler)
    optimizer = torch.optim.AdamW(params=model.parameters(), weight_decay=1e-5, lr=5e-5)
    scaler = GradScaler()

    total_start = time.time()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        if TQDM_ENABLED:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
        else:
            progress_bar = enumerate(train_loader)

        
        for step, batch in progress_bar:
            images = batch["image"]
            if IS_VINN: scale_factors = get_scale_factors(images)
            if FP16: images = images.to(torch.float16)
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=True, cache_enabled=False):
                # Generate random noise
                noise = torch.randn_like(images).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                if not IS_VINN:
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                else:
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, scale_factors=scale_factors)

                loss = F.mse_loss(noise_pred.float(), noise.float())
            # end autocast

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            if TQDM_ENABLED: progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        if (epoch + 1) % VAL_INTERVAL == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                if IS_VINN: scale_factors =  get_scale_factors(images)
                if FP16: images = images.to(torch.float16)
                noise = torch.randn_like(images).to(device)
                with autocast(enabled=True, cache_enabled=True), torch.no_grad():
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    if not IS_VINN:
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    else:
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps, scale_factors=scale_factors)

                    try:
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                    except Exception as e:
                        print(e)
                        import pdb; pdb.set_trace()

                val_epoch_loss += val_loss.item()
                if TQDM_ENABLED: progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

            # Sampling image during training
            image = torch.randn((1, 1, *IMAGE_SHAPE)) if not TWO_D else torch.randn((1, SLICE_THICKNESS, *IMAGE_SHAPE[1:]))
            
            image = image.to(device)
            scheduler.set_timesteps(num_inference_steps=NUM_REVERSE_DIFFUSION_STEPS)
            with autocast(enabled=True, cache_enabled=True), torch.no_grad(): # TODO: add no_grad()? maybe remove autocast?
                if IS_VINN:
                    scale_factors = torch.ones((1, 2 if TWO_D else 3), device=image.device) / internal_res_mm if not FIXED_SIZE else torch.ones((1, 3), device=image.device) # 1mm resolution with IMAGE_SHAPE size
                else:
                    scale_factors = torch.ones((1, 3), device=image.device)
                image = inferer.sample(input_noise=image, scale_factors=scale_factors, diffusion_model=model, scheduler=scheduler)
                #else:
                #    image = inferer.sample(input_noise=image, scale_factors=None,          diffusion_model=model, scheduler=scheduler)

            plot_batch(image, image_path=os.path.join(args.out_dir,f'images/epoch_{epoch+1}_image.png'), slice_cut=SLICE_CUT)
            torch.save(model, os.path.join(args.out_dir,f'model_epoch_{epoch+1}.pth'))

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")

    torch.save(model, os.path.join(args.out_dir,f'model_final.pth'))
