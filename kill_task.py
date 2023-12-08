import subprocess
import os
process = subprocess.Popen(['ps', '-u'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = process.communicate()

lines = out.decode().splitlines()

for l in lines:
    if 'torch' in l or 'python' in l:
        os.system(f'kill -9 {l.split()[1]}')
