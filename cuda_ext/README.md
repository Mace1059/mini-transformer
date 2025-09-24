D:
cd D:\Users\jmacy\Websites\GPT
.\.venv\Scripts\activate
cd cuda_ext
set DISTUTILS_USE_SDK=1
set MSSdk=1
rmdir /s /q build
for /d %i in (*.egg-info) do rmdir /s /q "%i"
pip install -e . --no-build-isolation --force-reinstall
