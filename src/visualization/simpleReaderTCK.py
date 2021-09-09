#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:06:53 2020

@author: alonso
"""

from dipy.io.streamline import load_tractogram
from dipy.viz import window, actor, colormap as cmap

import time
start_time = time.time()

gen_path_machine = '/home/gory/cursos/Estancia verano 2021 CIMAT/DSPS Workshop/Brain Team/'

path_files      = gen_path_machine + 'data/connectome_test/'
pathToReference = gen_path_machine + 'data/connectome_test/'

subject_str = '1'
subject_folder = 's'+ subject_str+'/'

#show = False
#fNameTrac = 'tracking-probabilistic.trk'

show = True
fNameTrac = 'FMI.tck'
#fNameTrac = 'CST_L.tck'
#fNameTrac = 'IFOF_R.tck'
fNameTrac = 'IFOF_L.tck'




fNameRef  = 't1.nii.gz'



tractogram = load_tractogram(path_files+subject_folder+fNameTrac, pathToReference+subject_folder+fNameRef, bbox_valid_check=False)

# all the streamlines 
STs = tractogram.streamlines

print('number of streamlines')
print(len(STs))

print('1st streamline')
firstST = STs[0]
print(firstST)

print('number of 3D point in 1st streamline')
npuntos = len(firstST)
print(npuntos)

print('coordinates of the first 2 points from 1st streamline')

x0 = firstST[0][0]
y0 = firstST[0][1]
z0 = firstST[0][2]
print(x0,y0,z0)

x1 = firstST[1][0]
y1 = firstST[1][1]
z1 = firstST[1][2]

print(x1,y1,z1)


if show:
	# visualization
	color = cmap.line_colors(STs)

	streamlines_actor = actor.line(STs, cmap.line_colors(STs))

	# Add display objects to canvas
	scene = window.Scene()
	scene.add(streamlines_actor)

	window.show(scene)


print("--- %s seconds ---" % (time.time() - start_time))
