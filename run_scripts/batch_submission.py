import sys
import os
import numpy as np 

output_data_path = "/p/lustre1/angelico/tile-geometry/3-18-23/1000um/"
activate_env = "source $HOME/my_personal_env/bin/activate"

zs = np.arange(-1, -1300, -5)
for z in zs:
    output_file = "{:d}".format(abs(z))
    cmd_options = "--export=ALL -p pbatch -t 18:00:00 -N 1 -J {} -o {}.out".format(output_file+"-tilegeo-1000um"+output_file, output_data_path+output_file)
    exe = "python3 $HOME/nexo-tile-geometry-calculator/run_scripts/generate_data.py 600 600 -5 {:d} {:d} {}".format(z, z-2, output_data_path+output_file)
    cmd_full = "{} && sbatch {} --wrap=\'{}\'".format(activate_env, cmd_options, exe)
    print(cmd_full)
    os.system(cmd_full)
