# **"Bootstrap Pattern - Installation Concept for Later"**


### The Strategy: "The Parasitic Venv"

You design your library to have a tiny "host" footprint. When imported, it checks for a hidden `.venv` inside its own directory (or user cache). If missing, it builds it.

Here is the complete implementation code for your library.

#### 1\. Directory Structure

```text
inferenceutils/
├── __init__.py          # The entry point
├── bootstrap.py         # The installer logic
└── _core.py             # Your actual library code (lazy loaded)
```

#### 2\. The `bootstrap.py` (The Installer)

This module handles the creation of the isolated environment and the "injection" of that environment into the current Python process.

```python
# inferenceutils/bootstrap.py
import sys
import os
import venv
import subprocess
import site
from pathlib import Path

# Define where our "Parasitic Venv" lives
# Option A: Inside the package (easier for portable builds)
# Option B: In ~/.cache/inferenceutils (better for permission safety)
VENV_PATH = Path(__file__).parent / ".runtime_venv"
IS_WINDOWS = sys.platform == "win32"
PYTHON_EXE = VENV_PATH / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")

def _get_site_packages(venv_path):
    """Find the site-packages directory of the venv we just created."""
    if IS_WINDOWS:
        return venv_path / "Lib" / "site-packages"
    else:
        # On Linux/Mac, it's lib/pythonX.Y/site-packages
        lib = venv_path / "lib"
        # Find the pythonX.Y folder
        py_dirs = list(lib.glob("python*"))
        if not py_dirs:
            raise RuntimeError("Could not find python directory in venv")
        return py_dirs[0] / "site-packages"

def ensure_environment():
    """
    1. Checks if venv exists.
    2. If not, creates it.
    3. Installs llama-cpp-python with CORRECT flags.
    4. Adds the venv to the CURRENT process's sys.path.
    """
    if VENV_PATH.exists():
        # Already installed? Just activate and return.
        # (You might want a version check here in production)
        _activate_runtime()
        return

    print("📦 InferenceUtils: First-run setup detected.")
    print("🚀 Creating isolated hardware-optimized environment...")
    
    # 1. Create the bare venv
    venv.create(VENV_PATH, with_pip=True)

    # 2. Detect Hardware (Reuse your system_info logic here!)
    #    For brevity, I'll pretend we detected Metal
    #    In reality, you'd import your local hardware detection module
    from .hardware.system_info import system_info
    hw = system_info()
    
    cmake_args = []
    if hw.gpu.apple:
        print("🍏 Apple Silicon detected. Enabling Metal...")
        cmake_args = ["-DGGML_METAL=on"]
    elif hw.gpu.nvidia:
        print("🟢 NVIDIA GPU detected. Enabling CUDA...")
        cmake_args = ["-DGGML_CUDA=on"]
    
    # 3. Install dependencies into the VENV (not the user's global python)
    print("🔧 Compiling inference engine (this may take a minute)...")
    
    env = os.environ.copy()
    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)
        # Force a rebuild to ensure flags are respected
        env["PIP_NO_CACHE_DIR"] = "1"

    subprocess.check_call([
        str(PYTHON_EXE), "-m", "pip", "install", 
        "llama-cpp-python", 
        "--force-reinstall", "--no-binary", "llama-cpp-python"
    ], env=env)

    print("✅ Setup complete.")
    _activate_runtime()

def _activate_runtime():
    """Injects the venv packages into the CURRENT python process."""
    site_pkg = _get_site_packages(VENV_PATH)
    
    # site.addsitedir() is better than sys.path.append() because 
    # it handles .pth files (which some packages rely on)
    site.addsitedir(str(site_pkg))
    
    # Optional: Re-trigger importlib to ensure packages are seen
    import importlib
    importlib.invalidate_caches()
```

#### 3\. The `__init__.py` (The Hook)

This is where the magic happens. You hide the bootstrap logic so the user never sees it—they just import and wait a few seconds.

```python
# inferenceutils/__init__.py
import sys

# 1. Run the bootstrap BEFORE importing anything else
from .bootstrap import ensure_environment

try:
    ensure_environment()
except Exception as e:
    print(f"❌ Critical Error: Failed to initialize inference runtime.\n{e}")
    sys.exit(1)

# 2. NOW we can import the libraries that depend on the venv
#    Because we just added the venv to sys.path, this import will work!
try:
    import llama_cpp
except ImportError:
    # Fallback if something went weird with path injection
    raise ImportError("Failed to load llama_cpp even after bootstrap.")

# 3. Export your public API
from .hardware import system_info
# ...
```

### Why this is safer than `pip install`

If you just ran `pip install` in the user's main environment, you risk:

1.  **Breaking their dependencies:** If they have `numpy==1.21` and you force upgrade it, you break their app.
2.  **Permission Denied:** If they installed Python as root/admin, your script will crash.
3.  **Conflict Hell:** If they already installed `llama-cpp-python` (CPU version), `pip` won't reinstall it unless you force it, which is rude.

By using a **Private Venv**, you own the sandbox. You can force-reinstall, mess with versions, and compile custom binaries without touching the user's main `pandas` or `torch` installation.

### The One Risk: "Shared Library Hell"

This approach works perfectly for Python-only dependencies. It works *mostly* well for C-extensions (`.so` / `.dll`).

However, if the user's main Python process has already loaded a DLL (e.g., `libcuda.so`) via `torch`, and your private venv tries to load a *different version* of `libcuda.so` via `llama-cpp`, you might get a segfault.

**Mitigation:** Since you are building a "Middleware" library, this is acceptable. Most users will use you to *manage* the inference, so you are likely the first/only thing loading the LLM backend.

### User Experience

**User:**

```python
import inferenceutils
# Output: 
# 📦 InferenceUtils: First-run setup detected.
# 🚀 Creating isolated hardware-optimized environment...
# 🍏 Apple Silicon detected. Enabling Metal...
# 🔧 Compiling inference engine...
# ✅ Setup complete.

model = inferenceutils.load(...) # Works instantly
```

This effectively turns your library into a **Self-Installing Appliance.**