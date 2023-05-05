import sys
import os

data_path = "/p/lustre1/angelico/tile-geometry/3-18-23/1000um/"
outfile = "chained.p"
activate_env = "source $HOME/my_personal_env/bin/activate"

cmd_options = "--export=ALL -p pbatch -t 1:00:00 -N 1 -J {} -o {}.out".format("chain", data_path+"chain")
exe = "python3 $HOME/nexo-tile-geometry-calculator/run_scripts/chain_data.py {} {}".format(data_path, outfile)
cmd_full = "{} && sbatch {} --wrap=\'{}\'".format(activate_env, cmd_options, exe)
print(cmd_full)
os.system(cmd_full)
