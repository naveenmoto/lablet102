import os, subprocess, glob
import json
import numpy as np
week_id = "week4"
outdir = f"./submissions/{week_id}/"
num_students = 0
week_stats = {}
for name in os.listdir(week_id):
    num_students += 1
    subprocess.call(["mkdir",outdir+name])
    files = glob.glob(os.path.join(week_id+'/'+name,'**',"*.ipynb"))
    num_probs = 0
    if files:
        for f in files:
            subprocess.call(["jupyter-nbconvert",f"--output-dir={outdir+name}","--to","HTML",f])
            num_probs += 1
        week_stats[name] = num_probs

with open(f"{outdir}stats.json", "w") as stat_file:
    json.dump(week_stats, stat_file)
