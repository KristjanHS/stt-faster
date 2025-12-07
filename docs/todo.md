# GPU enablement action plan (CTranslate2/faster-whisper)

- [ ] Reconfirm WSL CUDA visibility: ensure `/usr/lib/wsl/lib/libcuda.so*` exists, `nvidia-smi` works, and `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib` plus the installed CUDA lib64 (e.g., `/usr/local/cuda-12.x/lib64`); run `ldconfig -p | grep -E 'libcuda|libcudart'`.
- [ ] Remove any CPU-only installs: in the project venv, uninstall `ctranslate2` and `faster-whisper`, then reinstall from PyPI with uv using the stock GPU-capable wheels (`uv pip install --force-reinstall --no-cache-dir "ctranslate2==4.6.2" "faster-whisper"`).
- [ ] Verify CUDA detection in the same venv: run a short Python snippet to print `ctranslate2.__file__` and `ctranslate2.get_cuda_device_count()`; if it is 0, double-check that the snippet is executed with `LD_LIBRARY_PATH` set as above.
- [ ] If CUDA count stays 0, sanitize environment: temporarily unset Windows CUDA PATH entries in WSL, export `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.x/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`, rerun the detection snippet, and confirm counts >0.
- [ ] Rerun GPU diagnostics (`.venv/bin/python -m pytest tests/integration/test_gpu_diagnostics.py -q`); ensure the CUDA device count and tiny-model-on-GPU tests pass.
