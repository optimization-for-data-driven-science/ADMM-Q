import threading
import os
import random
import string
import shutil
import time

def randomString(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def run(d, v, i):
	script_dir = os.path.dirname(__file__)
	rel_path = randomString() + ".bat"
	abs_file_path = os.path.join(script_dir, rel_path)
	with open(abs_file_path, "w") as f:
		f.write("title V = %d, D = %d \n" % (v, d))
		f.write("python demo.py " + str(d) + " " + str(v) + " soft_%d_%d_%d.tmp" % (d, v, i))
	os.system("start " + rel_path)	
	time.sleep(5)
	os.remove(rel_path)


def clean():
	shutil.rmtree("tmp")
	os.mkdir("tmp")

clean()
threads = []


v = 10
d = 16
for i in range(12):
	threads.append(threading.Thread(target=run,args=(d, v, i,)))

for t in threads:
	t.start()