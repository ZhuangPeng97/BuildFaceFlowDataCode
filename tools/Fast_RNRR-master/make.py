import os
import sys
os.system("rm -r build")
os.system("mkdir build")
os.chdir("build")
os.system("cmake ../")
os.system("make all")
