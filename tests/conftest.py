import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MPLBACKEND"] = "Agg"
