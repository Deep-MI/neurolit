# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
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

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from platformdirs import user_data_dir
from tqdm import tqdm

from neurolit._version import get_version_with_hash

B2SHARE_FILES_URL = "https://b2share.fz-juelich.de/api/files/e649d98a-dada-4b00-98c6-fea142f68b87"
ZENODO_RECORD_URL = "https://zenodo.org/records/14510136/files"

MODEL_URLS = {
    "model_coronal.pt": [
        f"{B2SHARE_FILES_URL}/model_coronal.pt",
        f"{ZENODO_RECORD_URL}/model_coronal.pt?download=1",
    ],
    "model_axial.pt": [
        f"{B2SHARE_FILES_URL}/model_axial.pt",
        f"{ZENODO_RECORD_URL}/model_axial.pt?download=1",
    ],
    "model_sagittal.pt": [
        f"{B2SHARE_FILES_URL}/model_sagittal.pt",
        f"{ZENODO_RECORD_URL}/model_sagittal.pt?download=1",
    ],
}


def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: str | Path,
        urls: list[str],
        verbose: bool = False,
        show_progress: bool = True,
        position: int | None = None,
) -> None:
    """
    Download a checkpoint file with progress bar.

    Raises a ``RequestException`` if the remote file cannot be fetched or the
    download is incomplete. Propagates ``OSError`` if the checkpoint cannot be
    written to disk.

    Parameters
    ----------
    checkpoint_name : str
        Name of checkpoint.
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : list[str]
        List of URLs of checkpoint hosting sites.
    verbose : bool
        Whether to print verbose output.
    show_progress : bool
        Whether to show download progress bar.
    position : int | None
        Progress bar position for parallel downloads.
    """
    checkpoint_path = Path(checkpoint_path)
    partial_path = checkpoint_path.with_name(f"{checkpoint_path.name}.part")
    last_error: requests.exceptions.RequestException | None = None
    for url in urls:
        try:
            if verbose:
                print(f"Downloading checkpoint {checkpoint_name} from {url}")
            with requests.get(
                url,
                verify=True,
                timeout=(5, None),  # (connect timeout: 5 sec, read timeout: None)
                stream=True,  # Stream the download for progress tracking
            ) as response:
                # Raise error if file does not exist:
                response.raise_for_status()

                # Get total file size from headers
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192  # 8 KB chunks
                bytes_written = 0

                # Write file with progress bar to a temporary path first.
                with open(partial_path, "wb") as f:
                    if show_progress and total_size > 0:
                        with tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=checkpoint_name,
                            ncols=80,
                            position=position,
                            leave=True,
                        ) as pbar:
                            for chunk in response.iter_content(chunk_size=block_size):
                                if chunk:  # filter out keep-alive chunks
                                    f.write(chunk)
                                    bytes_written += len(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # Fallback without progress bar
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                bytes_written += len(chunk)

                if total_size > 0 and bytes_written != total_size:
                    raise requests.exceptions.RequestException(
                        f"Incomplete download for {checkpoint_name}: expected {total_size} bytes, received {bytes_written}."
                    )

            partial_path.replace(checkpoint_path)
            return

        except requests.exceptions.RequestException as e:
            last_error = e
            partial_path.unlink(missing_ok=True)
            if verbose:
                print(f"Failed downloading checkpoint {checkpoint_name} from {url} ({type(e).__name__}): {e}")
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                    print(f"Response code: {e.response.status_code}")
        except OSError:
            partial_path.unlink(missing_ok=True)
            raise

    if last_error is not None:
        raise last_error

    links = ', '.join(u.removeprefix('https://')[:22] + "..." for u in urls)
    raise requests.exceptions.RequestException(
        f"Failed downloading the checkpoint {checkpoint_name} from {links}."
    )


def check_and_download_ckpts(
        checkpoint_path: Path | str,
        urls: list[str],
        verbose: bool = False,
        show_progress: bool = True,
        position: int | None = None,
) -> None:
    """
    Check and download a checkpoint file, if it does not exist.

    Parameters
    ----------
    checkpoint_path : Path, str
        Path of the file in which the checkpoint will be saved.
    urls : list[str]
        URLs of checkpoint hosting site.
    verbose : bool
        Whether to print verbose output.
    show_progress : bool
        Whether to show download progress bar.
    position : int | None
        Progress bar position for parallel downloads.
    """
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
    # Download checkpoint file from url if it does not exist
    if not checkpoint_path.exists():
        if not show_progress:
            print(f"Downloading checkpoint {checkpoint_path} from {urls}")
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls, verbose, show_progress, position)


def fallback_multiple_urls(
        checkpoint_path: Path | str,
        urls: list[str],
        verbose: bool = False,
        show_progress: bool = True,
        position: int | None = None,
) -> None:
    last_error: Exception | None = None
    for url in urls:
        try:
            check_and_download_ckpts(checkpoint_path, [url], verbose, show_progress, position)
            if Path(checkpoint_path).exists():
                return
        except Exception as e:
            tqdm.write(f"Tried downloading {checkpoint_path} from {url} but failed")
            tqdm.write(str(e))
            last_error = e

    links = ', '.join(u.removeprefix('https://')[:22] + "..." for u in urls)
    raise requests.exceptions.RequestException(
        f"Failed downloading the checkpoint {Path(checkpoint_path).name} from {links}."
    ) from last_error


def main(argv=None):
    import sys
    
    # Use platformdirs for consistent cross-platform data directory
    # This works for both pip and git installations
    weights_dir = Path(user_data_dir("LIT", "Deep-MI")) / "weights"

    parser = argparse.ArgumentParser(
        description="Download neuroLIT checkpoints (T1w lesion inpainting models)",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default download location:
  {weights_dir}

If you use neuroLIT for research publications, please cite:

Pollak C, Kuegler D, Bauer T, Rueber T, Reuter M, FastSurfer-LIT: Lesion Inpainting Tool for Whole
  Brain MRI Segmentation with Tumors, Cavities and Abnormalities, Imaging Neuroscience 2025.
  https://doi.org/10.1162/imag_a_00446
"""
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    parser.add_argument("-v", "--version", action="version", version=get_version_with_hash(), help="Print version number and exit")
    parser.parse_args(argv)

    # Ensure weights directory exists
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"Download location: {weights_dir}")
    
    
    # Check which models already exist
    missing_models = []
    for model_name in MODEL_URLS.keys():
        model_path = weights_dir / model_name
        if not model_path.exists():
            missing_models.append(model_name)
    
    
    if not missing_models:
        print("\nAll models are already downloaded!")
        print("=" * 60)
        return
    
    print(f"\nModels to download ({len(missing_models)}/{len(MODEL_URLS)}):")
    for model in missing_models:
        print(f"{model}")
    
    print("\nDownloading models (this may take several minutes)...")
    print()  # Empty line before progress bars
    
    def download_model(model_name: str, position: int) -> str:
        model_path = weights_dir / model_name
        fallback_multiple_urls(
            str(model_path),
            urls=MODEL_URLS[model_name],
            verbose=False,
            show_progress=True,
            position=position,
        )
        return model_name

    # Download missing checkpoints in parallel.
    success = True
    max_workers = min(len(missing_models), len(MODEL_URLS))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_model, model_name, position): model_name
            for position, model_name in enumerate(missing_models)
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"\nDownload failed for {model_name}: {e}")
                success = False

    missing_models = [model_name for model_name in MODEL_URLS if not (weights_dir / model_name).exists()]
    if missing_models:
        print("\nMissing model files after download:")
        for model_name in missing_models:
            print(model_name)
        success = False

    if not success:
        print("=" * 30 + " exiting with errors" + "=" * 30)
        sys.exit(1)
    
    print("All models downloaded successfully!")
    print(f"Models location: {weights_dir}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
