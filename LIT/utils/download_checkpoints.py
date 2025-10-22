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

from pathlib import Path
import requests
from tqdm import tqdm
from platformdirs import user_data_dir

def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: str | Path,
        urls: list[str],
        verbose: bool = False,
        show_progress: bool = True,
) -> None:
    """
    Download a checkpoint file with progress bar.

    Raises an HTTPError if the file is not found or the server is not reachable.

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
    """
    response = None
    for url in urls:
        try:
            if verbose:
                print(f"Downloading checkpoint {checkpoint_name} from {url}")
            response = requests.get(
                url + "/" + checkpoint_name,
                verify=True,
                timeout=(5, None),  # (connect timeout: 5 sec, read timeout: None)
                stream=True,  # Stream the download for progress tracking
            )
            # Raise error if file does not exist:
            response.raise_for_status()
            break

        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"Server {url} not reachable ({type(e).__name__}): {e}")
            if isinstance(e, requests.exceptions.HTTPError):
                if verbose:
                    print(f"Response code: {e.response.status_code}")

    if response is None:
        links = ', '.join(u.removeprefix('https://')[:22] + "..." for u in urls)
        raise requests.exceptions.RequestException(
            f"Failed downloading the checkpoint {checkpoint_name} from {links}."
        )
    else:
        response.raise_for_status()  # Raise error if no server is reachable

    # Get total file size from headers
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 KB chunks
    
    # Write file with progress bar
    with open(checkpoint_path, "wb") as f:
        if show_progress and total_size > 0:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc=checkpoint_name,
                ncols=80,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            # Fallback without progress bar
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)


def check_and_download_ckpts(checkpoint_path: Path | str, urls: list[str], verbose: bool = False, show_progress: bool = True) -> None:
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
    """
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
    # Download checkpoint file from url if it does not exist
    if not checkpoint_path.exists():
        if not show_progress:
            print(f"Downloading checkpoint {checkpoint_path} from {urls}")
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls, verbose, show_progress)


def fallback_multiple_urls(checkpoint_name: str, urls: list[str], verbose: bool = False, show_progress: bool = True) -> None:
    for url in urls:
        try:
            check_and_download_ckpts(checkpoint_name, [url], verbose, show_progress)
        except Exception as e:
            print(f"Tried downloading {checkpoint_name} from {url} but failed")
            print(e)


def main():
    import sys
    
    # Use platformdirs for consistent cross-platform data directory
    # This works for both pip and git installations
    weights_dir = Path(user_data_dir("LIT", "Deep-MI")) / "weights"
    
    # Ensure weights directory exists
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"LIT Model Download Utility")
    print(f"=" * 60)
    print(f"Download location: {weights_dir}")
    print(f"=" * 60)
    
    # Model URLs
    models = {
        "model_coronal.pt": "https://zenodo.org/records/14510136/files/model_coronal.pt?download=1",
        "model_axial.pt": "https://zenodo.org/records/14510136/files/model_axial.pt?download=1",
        "model_sagittal.pt": "https://zenodo.org/records/14510136/files/model_sagittal.pt?download=1",
    }
    
    # Check which models already exist
    existing_models = []
    missing_models = []
    for model_name in models.keys():
        model_path = weights_dir / model_name
        if model_path.exists():
            existing_models.append(model_name)
        else:
            missing_models.append(model_name)
    
    if existing_models:
        print(f"\nModels already downloaded ({len(existing_models)}/{len(models)}):")
        for model in existing_models:
            print(f"  ✓ {model}")
    
    if not missing_models:
        print("\n✓ All models are already downloaded!")
        print(f"\nModels location: {weights_dir}")
        return
    
    print(f"\nModels to download ({len(missing_models)}/{len(models)}):")
    for model in missing_models:
        print(f"  • {model}")
    
    print("\nDownloading models (this may take several minutes)...")
    print()  # Empty line before progress bars
    
    # Download missing checkpoints
    success = True
    for model_name, url in models.items():
        model_path = weights_dir / model_name
        if model_path.exists():
            continue
        try:
            fallback_multiple_urls(str(model_path), urls=[url], verbose=False, show_progress=True)
        except Exception as e:
            print(f"\n✗ Download failed for {model_name}: {e}")
            success = False
    
    if not success:
        sys.exit(1)
    
    print(f"\n{'=' * 60}")
    print(f"✓ All models downloaded successfully!")
    print(f"Models location: {weights_dir}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
