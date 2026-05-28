$env:VCToolsVersion = "14.44.35207"
$env:CUDAFLAGS = "-allow-unsupported-compiler"
$env:TORCH_CUDA_ARCH_LIST = "8.9"

# Add MSVC v143 cl.exe and ninja to PATH
$env:Path = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64;C:\Users\ayber\PycharmProjects\colmap_gs_sfm\.venv\Scripts;" + $env:Path

Set-Location "C:\Users\ayber\PycharmProjects\colmap_gs_sfm"
& .\.venv\Scripts\python.exe main.py --images data/images --quality high --colmap-binary "C:\COLMAP\COLMAP.bat" --skip-dense
