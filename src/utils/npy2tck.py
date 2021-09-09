import argparse
from dipy.io.stateful_tractogram import StatefulTractogram,Space
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.viz import window, actor, ui, colormap as cmap

def npy_2_tck(streamlines, reference_path, output_path):
    stf = StatefulTractogram(streamlines, reference_path, Space.RASMM)
    save_tractogram(stf, output_path)