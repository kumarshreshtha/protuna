"""Utilities for downloading remote files from URLs."""

import pathlib

import requests
import tqdm


def download_from_url(url: str, save_path: pathlib.Path) -> pathlib.Path:
    """Download a file from a URL and save it to a local path.

    Args:
        url: The HTTP(S) URL to download from.
        save_path: The local destination path to write the downloaded file.

    Returns:
        The ``save_path`` where the file was written.

    Raises:
        requests.HTTPError: If the HTTP request returns an unsuccessful status.
    """
    save_path.parent.mkdir(exist_ok=True, parents=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(save_path, "wb") as f,
            tqdm.tqdm(total=total, unit="B", unit_scale=True) as pbar,
        ):
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))
    return save_path
