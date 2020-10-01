from pathlib import Path
import wget
from urllib.parse import urlparse
from torch.hub import _get_torch_home
import os

LINKS = {
    "TPS-ResNet-BiLSTM-Attn.pth": "https://www.dropbox.com/s/it6hm424ncitunr/TPS-ResNet-BiLSTM-Attn.pth?dl=1"
}
torch_cache_home = _get_torch_home()
CACHE_PATH = os.path.join(torch_cache_home, "ocr")

def get_from_cache(
    url,
    cache_dir=None,
    ):
    """
    Given a URL, look for the corresponding file in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    Return:
        None in case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
        Local path (string) otherwise
    """
    if cache_dir is None:
        cache_dir = CACHE_PATH
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.split("/")[-1].split(".pth")[0] + ".pth"
    if not os.path.isfile(os.path.join(cache_dir, filename)):
        wget.download(url, cache_dir)
    return os.path.join(cache_dir, filename)

def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def cached_path(url_or_filename,
                cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = CACHE_PATH
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))
    return output_path
