Training
========

This guide explains how to train your own LIT models on custom data.

Overview
--------

LIT uses a Denoising Diffusion Probabilistic Model (DDPM) architecture with a U-Net backbone. The model is trained separately for three orthogonal views (axial, coronal, and sagittal).

Prerequisites
-------------

Before training, ensure you have:

1. **Training data:** T1-weighted brain MRI images
2. **Sufficient compute:** GPU with at least 16 GB VRAM recommended
3. **Storage:** Adequate space for datasets and checkpoints
4. **Environment:** Docker container or local installation with all dependencies

Data Preparation
----------------

Conforming Images
~~~~~~~~~~~~~~~~~

All training images must be conformed to a standard space using the provided conform script:

.. code-block:: bash

   python3 LIT/data/conform.py \\
       --input raw_image.nii.gz \\
       --output conformed_image.nii.gz

The conform script:

* Resamples to 1mm isotropic voxels
* Reorients to standard RAS orientation
* Crops or pads to a consistent size
* Normalizes intensity values

Dataset Structure
~~~~~~~~~~~~~~~~~

Organize your conformed data in a directory structure:

.. code-block:: text

   training_data/
   ├── subject_001/
   │   └── T1w_conformed.nii.gz
   ├── subject_002/
   │   └── T1w_conformed.nii.gz
   └── subject_003/
       └── T1w_conformed.nii.gz

Training Configuration
----------------------

Basic Training
~~~~~~~~~~~~~~

To train a model for one view (e.g., axial):

.. code-block:: bash

   python3 LIT/train_ddpm.py \\
       --data_dir /path/to/training_data \\
       --output_dir /path/to/output \\
       --view axial \\
       --batch_size 16 \\
       --num_epochs 1000 \\
       --learning_rate 1e-4

**Key Parameters:**

* ``--data_dir``: Directory containing conformed training images
* ``--output_dir``: Where to save model checkpoints and logs
* ``--view``: Which view to train (axial, coronal, or sagittal)
* ``--batch_size``: Batch size (adjust based on GPU memory)
* ``--num_epochs``: Number of training epochs
* ``--learning_rate``: Learning rate for optimizer

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For more control over training:

.. code-block:: bash

   python3 LIT/train_ddpm.py \\
       --data_dir /path/to/training_data \\
       --output_dir /path/to/output \\
       --view axial \\
       --batch_size 16 \\
       --num_epochs 1000 \\
       --learning_rate 1e-4 \\
       --num_workers 4 \\
       --save_interval 50 \\
       --validation_split 0.1 \\
       --timesteps 1000 \\
       --beta_start 0.0001 \\
       --beta_end 0.02

**Additional Parameters:**

* ``--num_workers``: Number of data loading workers
* ``--save_interval``: Save checkpoint every N epochs
* ``--validation_split``: Fraction of data for validation
* ``--timesteps``: Number of diffusion timesteps
* ``--beta_start``: Starting beta value for noise schedule
* ``--beta_end``: Ending beta value for noise schedule

Using Docker for Training
--------------------------

You can use the same Docker image for training. Mount your data directory and run the training script:

.. code-block:: bash

   docker run --gpus all \\
       -v /path/to/training_data:/data \\
       -v /path/to/output:/output \\
       --rm deepmi/lit:latest \\
       python3 /opt/LIT/train_ddpm.py \\
           --data_dir /data \\
           --output_dir /output \\
           --view axial \\
           --batch_size 16 \\
           --num_epochs 1000

Training All Views
------------------

To train models for all three views, run training three times:

.. code-block:: bash

   # Train axial model
   python3 LIT/train_ddpm.py \\
       --data_dir /path/to/data \\
       --output_dir /path/to/output/axial \\
       --view axial \\
       --batch_size 16 \\
       --num_epochs 1000
   
   # Train coronal model
   python3 LIT/train_ddpm.py \\
       --data_dir /path/to/data \\
       --output_dir /path/to/output/coronal \\
       --view coronal \\
       --batch_size 16 \\
       --num_epochs 1000
   
   # Train sagittal model
   python3 LIT/train_ddpm.py \\
       --data_dir /path/to/data \\
       --output_dir /path/to/output/sagittal \\
       --view sagittal \\
       --batch_size 16 \\
       --num_epochs 1000

Monitoring Training
-------------------

Training Logs
~~~~~~~~~~~~~

The training script outputs logs to stdout and saves them to the output directory:

.. code-block:: text

   output/
   ├── checkpoints/
   │   ├── model_epoch_0050.pt
   │   ├── model_epoch_0100.pt
   │   └── ...
   ├── logs/
   │   └── training.log
   └── config.yaml

Key Metrics
~~~~~~~~~~~

Monitor these metrics during training:

* **Loss:** Should decrease over time
* **Validation loss:** Should track training loss without diverging
* **Sample quality:** Periodically generate samples to check visual quality

Sample Generation
~~~~~~~~~~~~~~~~~

To generate samples during training for quality assessment:

.. code-block:: python

   from neuro_lit.inference import sample_from_model
   
   # After training for a few epochs
   samples = sample_from_model(
       model,
       num_samples=5,
       device='cuda'
   )

Using Custom Models
-------------------

After training, you can use your custom models with LIT:

Replace Model Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~

Place your trained models in the weights directory:

.. code-block:: bash

   cp output/axial/checkpoints/model_final.pt ~/.local/share/LIT/weights/model_axial.pt
   cp output/coronal/checkpoints/model_final.pt ~/.local/share/LIT/weights/model_coronal.pt
   cp output/sagittal/checkpoints/model_final.pt ~/.local/share/LIT/weights/model_sagittal.pt

Specify Model Paths
~~~~~~~~~~~~~~~~~~~

Or specify custom model paths when running inference:

.. code-block:: bash

   inpaint-image \\
       --input_image T1w.nii.gz \\
       --mask_image mask.nii.gz \\
       --out_dir output \\
       --model_axial output/axial/checkpoints/model_final.pt \\
       --model_coronal output/coronal/checkpoints/model_final.pt \\
       --model_sagittal output/sagittal/checkpoints/model_final.pt

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

1. **Use high-quality images:** 1mm isotropic T1-weighted MRI
2. **Sufficient diversity:** Include various scanners, protocols, and populations
3. **Quality control:** Manually review all training images
4. **Consistent preprocessing:** Always use the conform.py script

Training Strategy
~~~~~~~~~~~~~~~~~

1. **Start with pre-trained models:** Fine-tune rather than train from scratch if possible
2. **Use validation set:** Monitor for overfitting
3. **Save checkpoints frequently:** Keep every 50-100 epochs
4. **Train for sufficient epochs:** At least 500-1000 epochs
5. **Batch size:** Use largest batch size that fits in GPU memory

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

If results are not satisfactory:

1. **Learning rate:** Try 1e-3, 1e-4, or 1e-5
2. **Batch size:** Larger batches (if memory allows) can stabilize training
3. **Timesteps:** More timesteps (e.g., 2000) may improve quality
4. **Beta schedule:** Adjust beta_start and beta_end for noise schedule

Troubleshooting
---------------

Out of Memory
~~~~~~~~~~~~~

**Problem:** CUDA out of memory during training

**Solutions:**

* Reduce batch size
* Reduce number of timesteps
* Use gradient accumulation
* Train on a GPU with more memory

Poor Sample Quality
~~~~~~~~~~~~~~~~~~~

**Problem:** Generated samples look unrealistic

**Solutions:**

* Train for more epochs
* Increase dataset size and diversity
* Check data preprocessing and conforming
* Tune hyperparameters (learning rate, beta schedule)

Training Not Converging
~~~~~~~~~~~~~~~~~~~~~~~

**Problem:** Loss not decreasing or training unstable

**Solutions:**

* Reduce learning rate
* Check that data is properly normalized
* Verify data loading and augmentation
* Increase batch size
* Add gradient clipping

Model Comparison
----------------

To compare different models:

.. code-block:: python

   from neuro_lit.inference import compare_models
   
   models = {
       'model_v1': 'checkpoints/model_v1.pt',
       'model_v2': 'checkpoints/model_v2.pt',
   }
   
   results = compare_models(
       models,
       test_images=['test1.nii.gz', 'test2.nii.gz'],
       test_masks=['mask1.nii.gz', 'mask2.nii.gz']
   )

Metrics for Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Common metrics for evaluating inpainting quality:

* **Visual quality:** Subjective assessment by experts
* **PSNR:** Peak Signal-to-Noise Ratio
* **SSIM:** Structural Similarity Index
* **Downstream task performance:** Test on segmentation or surface reconstruction

For Research
------------

If you train custom models for research:

1. **Document all parameters** used for training
2. **Report dataset statistics** (size, scanner types, populations)
3. **Provide model checkpoints** for reproducibility
4. **Benchmark against baseline** using standard test sets
5. **Share training code** modifications if any

Contributing Models
-------------------

If you train models that improve upon the defaults:

1. Open an issue on the `GitHub repository <https://github.com/Deep-MI/LIT>`_
2. Provide details about training data and parameters
3. Share benchmark results
4. Discuss with maintainers about inclusion

The LIT team welcomes community contributions of improved models!

