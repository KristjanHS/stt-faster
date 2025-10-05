# python - <<'PY'
import ctypes
ctypes.CDLL("libcudnn_ops.so.9")
print("cuDNN 9 OK")
#PY