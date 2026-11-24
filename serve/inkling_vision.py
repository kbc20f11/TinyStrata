"""Request images for Inkling's in-process vision encoder."""

from pathlib import Path
import tempfile
from urllib.parse import quote


# The supported Vision shard uses 40-pixel tiles and a 4096-pixel side cap.
def tile_count(width: int, height: int) -> int:
    if max(width, height) > 4096:
        scale = 4096 / max(width, height)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    return ((width + 39) // 40) * ((height + 39) // 40)


def stage(images: list[bytes]) -> tuple[tempfile.TemporaryDirectory, list[str]]:
    directory = tempfile.TemporaryDirectory(prefix="tinystrata-inkling-images-", ignore_cleanup_errors=True)
    paths = []
    try:
        for index, data in enumerate(images):
            path = Path(directory.name) / f"{index}.img"
            path.write_bytes(data)
            paths.append(quote(str(path), safe="/\\:"))
    except BaseException:
        directory.cleanup()
        raise
    return directory, paths
