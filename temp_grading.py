import os, subprocess, glob
import json
import numpy as np
week_id = "week2/kasina/"
outdir = f"./submissions/week_2/kasina/"
files = glob.glob(os.path.join(week_id+'/','**',"*.ipynb"))
for f in files:
    subprocess.call(["jupyter-nbconvert",f"--output-dir={outdir}","--to","HTML",f])

