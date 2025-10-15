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

def download_checkpoint(
        checkpoint_name: str,
        checkpoint_path: str | Path,
        urls: list[str],
        verbose: bool = False,
) -> None:
    """
    Download a checkpoint file.

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

    with open(checkpoint_path, "wb") as f:
        f.write(response.content)


def check_and_download_ckpts(checkpoint_path: Path | str, urls: list[str], verbose: bool = False) -> None:
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
    """
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)
    # Download checkpoint file from url if it does not exist
    if not checkpoint_path.exists():
        print(f"Downloading checkpoint {checkpoint_path} from {urls}")
        # create dir if it does not exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_checkpoint(checkpoint_path.name, checkpoint_path, urls, verbose)


def fallback_multiple_urls(checkpoint_name: str, urls: list[str], verbose: bool = False) -> None:
    for url in urls:
        try:
            check_and_download_ckpts(checkpoint_name, [url], verbose)
        except Exception as e:
            print(f"Tried downloading {checkpoint_name} from {url} but failed")


def main():
    # "https://github.com/Deep-MI/LIT/releases/download/v0.5.0/model_coronal.pt"
    # "https://github.com/Deep-MI/LIT/releases/download/v0.5.0/model_axial.pt"
    # "https://github.com/Deep-MI/LIT/releases/download/v0.5.0/model_sagittal.pt"
    fallback_multiple_urls("weights/model_coronal.pt", urls=["https://zenodo.org/records/14510136/files/model_coronal.pt?download=1"], verbose=False)
    fallback_multiple_urls("weights/model_axial.pt", urls=["https://zenodo.org/records/14510136/files/model_axial.pt?download=1"], verbose=False)
    fallback_multiple_urls("weights/model_sagittal.pt", urls=["https://zenodo.org/records/14510136/files/model_sagittal.pt?download=1"], verbose=False)

if __name__ == "__main__":
    main()
