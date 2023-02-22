Requirements:
  - Python >= 3.5
  - PyTorch >= 1.1
  - numpy

Getting Started:
  - We provide "normal train", "binarization" and "binarization_random" functions in the main.py.
  - By default setting, use command "python main.py" will pretrain and then binarize with ADMM-Q.
  - User can modify main function to reproduce other results. For example, at Line 60, change "binarization" to "binarization_random" to test ADMM-R.