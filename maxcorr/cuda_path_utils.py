import os
import logging
import platform
import site
import sys
from pathlib import Path

_IS_CUDA_PATHS_SET = False


def setup_cuda_paths():
    """
    Dynamically adds NVIDIA library paths to the runtime search path.
    Handles Linux (LD_LIBRARY_PATH) and Windows (os.add_dll_directory).
    Bypasses macOS (no CUDA support).
    """
    global _IS_CUDA_PATHS_SET

    if _IS_CUDA_PATHS_SET:
        return

    # Silence TF Logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["ABSL_LOGGING_LEVEL"] = "error"
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    system = platform.system()

    # Mac doesn't use CUDA (uses MPS on Apple Silicon)
    if system == "Darwin":
        _IS_CUDA_PATHS_SET = True
        return

    # Reliably find site-packages (works for standard and virtual envs like uv)
    site_packages = site.getsitepackages()
    if hasattr(site, "getusersitepackages"):
        site_packages.append(site.getusersitepackages())

    nvidia_libs = []
    for sp in site_packages:
        sp_path = Path(sp)
        # Find all nvidia subdirectories containing 'lib' or 'bin' (Windows uses bin for DLLs)
        nvidia_libs.extend(sp_path.glob("nvidia/*/lib"))
        nvidia_libs.extend(sp_path.glob("nvidia/*/bin"))  # For Windows

    if not nvidia_libs:
        _IS_CUDA_PATHS_SET = True
        return

    lib_paths = [str(p.absolute()) for p in set(nvidia_libs)]

    # OS-Specific Linking
    if system == "Linux":
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join(lib_paths)
        os.environ["LD_LIBRARY_PATH"] = (
            f"{new_ld}:{current_ld}" if current_ld else new_ld
        )

    elif system == "Windows":
        current_path = os.environ.get("PATH", "")
        new_path = ";".join(lib_paths)
        os.environ["PATH"] = f"{new_path};{current_path}" if current_path else new_path

        # Python 3.8+ on Windows requires explicit DLL directory registration
        if sys.version_info >= (3, 8):
            for p in lib_paths:
                try:
                    os.add_dll_directory(p)
                except Exception:
                    pass

    # Lock the function so it never runs again
    _IS_CUDA_PATHS_SET = True
