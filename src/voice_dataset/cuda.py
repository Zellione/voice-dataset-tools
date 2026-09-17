import ctypes
from pathlib import Path
import site


def preload_cuda_libraries() -> None:
    library_dirs = []

    for package_dir in site.getsitepackages():
        package_dir = Path(package_dir)

        library_dirs.extend([
            package_dir / "nvidia" / "cublas" / "lib",
            package_dir / "nvidia" / "cudnn" / "lib",
        ])

    libraries = [
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
    ]

    for library in libraries:
        for directory in library_dirs:
            path = directory / library

            if path.exists():
                ctypes.CDLL(
                    str(path),
                    mode=ctypes.RTLD_GLOBAL,
                )
                break
