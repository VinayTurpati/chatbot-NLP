import os
import glob
import shutil
import config

def clear_cache(clear_model = False):
	#removing files with .pkl and .h5 extensions
	for file in glob.glob("*.pkl"):
		os.remove(file)
	for file in glob.glob("*.pyc"):
		os.remove(file)

	#Delete _model.h5 file, if clear_model = True
	if clear_model:
		for file in glob.glob("*.h5"):
			os.remove(file)

	#Delete __pycache__ folder if exists
	try:
		cache = os.path.join(os.getcwd(), '__pycache__')
		shutil.rmtree(cache)
	except:
		pass

clear_cache(config.clear_model)