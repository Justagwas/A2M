from __future__ import annotations

RUNTIME_INFO_CHECKING = 'Still checking ONNX Runtime support.'

RUNTIME_WARNING_MISSING = (
    'ONNX Runtime is missing in this installation.\n'
    'Please reinstall A2M to restore conversion support.'
)

RUNTIME_LABEL_CHECKING = 'Checking ONNX Runtime support...'

RUNTIME_LABEL_MISSING = (
    'ONNX Runtime is missing in this installation.\n'
    'Reinstall A2M to restore conversion support.'
)

RUNTIME_LABEL_GPU_NO_PACK = (
    'GPU selected, but no GPU runtime pack is installed.\n'
    'Install CUDA/DirectML runtime pack to enable GPU.'
)

RUNTIME_LABEL_GPU_UNAVAILABLE = (
    'GPU runtime is installed, but acceleration is unavailable on this machine.\n'
    'A2M is currently using CPU fallback.'
)

RUNTIME_LABEL_CUDA_ACTIVE = 'CUDA acceleration is active.'

RUNTIME_LABEL_DML_ACTIVE = 'DirectML acceleration is active.'

RUNTIME_LABEL_CPU_WITH_GPU_PACK = 'CPU mode is active. A GPU runtime pack is installed.'

RUNTIME_LABEL_CPU_NO_GPU_PACK = 'CPU mode is active. GPU runtime pack not installed.'
