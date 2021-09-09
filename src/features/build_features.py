#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 15:31 2021

@author: Armando Cruz (Gory)
"""

import time
import argparse
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram,Space
from dipy.io.streamline import load_tractogram, save_tractogram
from src.utils import constants

errors_2 = np.array([ [x1, x2, x3, x4, x5, x6] for x1 in range(2) for x2 in range(2) for x3 in range(2)
                                    for x4 in range(2) for x5 in range(2) for x6 in range(2)], dtype=np.int32)
errors_3 = np.array([ [x1, x2, x3, x4, x5, x6] for x1 in range(-1,2) for x2 in range(-1,2) for x3 in range(-1,2)
                                    for x4 in range(-1,2) for x5 in range(-1,2) for x6 in range(-1,2)], dtype=np.int32)


def save_streamlines(streamlines, reference_path, output_path):
    stf = StatefulTractogram(streamlines, reference_path, Space.RASMM)
    save_tractogram(stf, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=str, help='Name of subject folder.')
    parser.add_argument('--ref', '--reference', default='fa.nii.gz', help='Reference file, Nifti or Trk file. (default: fa.nii.gz)')
    parser.add_argument('--npoints', type=int , default=20, help='Number of points in each streamline. (default: 20)')
    parser.add_argument('--epsilon', type=float , default=0.0, help='Error allowed. (default: 0.05)')
    parser.add_argument('--bruteforce', nargs='?', default=False, const=True, help='Do final bruteforce search. (default: False)')
    parser.add_argument('-tck', nargs='?', default=False, const=True, help='Save to tck. (default: False)')
    # parser.add_argument('--show_notfound', default=False, const=True help='Show best candidate to notfound streamline. (default: False)')
    args = parser.parse_args()

    TIC = time.time()
    epsilon = 1.0 if args.epsilon == 0.0 else args.epsilon

    print('##### Loading full tractogram')
    tic = time.time()
    full_tract = load_tractogram(constants.data_raw_path+args.subject+'_full_20p.tck', 
                                 constants.data_raw_path+args.subject+'/'+args.ref, 
                                 bbox_valid_check=False)
    full_streamlines = np.array(full_tract.streamlines)
    full_streamlines_endpts = np.floor(full_streamlines[:,(0,-1)]/epsilon).reshape((full_streamlines.shape[0], 2*3))
    labeled = np.zeros(len(full_streamlines), dtype=bool)
    labels = np.zeros(len(full_streamlines),dtype=np.int32)
    toc = time.time()
    full_tracto_size = len(full_streamlines)
    print('Loading full tractogram time:', toc-tic)
    print('Full tracto size:', full_tracto_size)
    print()
    
    clusters = constants.get_clusters(args.subject)
    cluster_tracto_size = 0
    not_found = []
    not_found_endpts = []
    not_found_labels = []

    for c in range(len(clusters['names'])):
        cluster_path = clusters['paths'][c]
        cluster_name = clusters['names'][c]
        cluster_label = c+1
        if cluster_label == -1:
            continue

        print('##### Loading cluster:', cluster_name, c)
        tic = time.time()
        cluster_tract = load_tractogram(cluster_path, 
                                     constants.data_raw_path+args.subject+'/'+args.ref, 
                                     bbox_valid_check=False)
        cluster_streamlines = np.array(cluster_tract.streamlines)
        cluster_tracto_size += len(cluster_streamlines)
        if len(cluster_streamlines) == 0:
            toc = time.time()
            print('Loading cluster time', toc-tic)
            print('Cluster size:', 0)
            continue
        cluster_endpts = np.floor(cluster_streamlines[:,(0,-1)]/epsilon).reshape((cluster_streamlines.shape[0], 2*3))
        cluster_endpts_inv = np.floor(cluster_streamlines[:,(-1,0)]/epsilon).reshape((cluster_streamlines.shape[0], 2*3))
        endpts = list(map(tuple, cluster_endpts)) + list(map(tuple, cluster_endpts_inv))
        cluster_set = set( endpts )
        toc = time.time()
        print('Loading cluster time', toc-tic)
        print('Cluster size:', len(cluster_streamlines))


        print('### Labeling:', cluster_name, c)
        
        print('# First')
        tic = time.time()
        labeled_set = set()
        for i, sl in enumerate(full_streamlines_endpts):
            sltuple = tuple(sl)
            if sltuple in cluster_set:
                labeled[i] = True
                labels[i] = cluster_label
                labeled_set.add(sltuple)
        toc = time.time()
        print('First labeling time:', toc-tic)
        print('Number of first labeled streamlines:', len(labeled_set))

        if len(labeled_set) == len(cluster_streamlines):
            print()
            continue


        print('# Second')
        if args.epsilon > 0:
            tic = time.time()

            endpts_error = []
            for endpts in cluster_endpts:
                endpts_error += list(map(tuple, endpts + errors_3))
            for endpts in cluster_endpts_inv:
                endpts_error += list(map(tuple, endpts + errors_3))
            cluster_set_error = set( endpts_error )

            for i, sl in enumerate(full_streamlines_endpts):
                sltuple = tuple(sl)
                if sltuple in cluster_set_error:
                    labeled[i] = True
                    labels[i] = cluster_label
                    labeled_set.add(sltuple)
            toc = time.time()
            print('Second labeling time:', toc-tic)
            print('Number of first and second labeled streamlines:', len(labeled_set))

        if len(labeled_set) == len(cluster_streamlines):
            print()
            continue

        if args.bruteforce:
            print('# Bruteforce')
            # tic = time.time()
            # nshow = 2
            # notfound_show = []
            # best_candidate_show = []
            # best_candidate_show_inv = []
            # for i, csl in enumerate(not_found_endpts[:nshow]):
            #     norms = np.linalg.norm(full_streamlines_endpts-csl.reshape(1,csl.shape[0]), axis=1)
            #     best_candidate = np.argmin( norms )
            #     notfound_show.append(not_found[i])
            #     best_candidate_show.append(full_streamlines[best_candidate])
            #     if norms[best_candidate] <= args.epsilon:
            #         labeled[best_candidate] = True
            #         print('HEUREKA')
            #         continue
            #     norms = np.linalg.norm(full_streamlines_endpts-csl.reshape(1,csl.shape[0]), axis=1)
            #     best_candidate_inv = np.argmin( norms )
            #     best_candidate_show_inv.append(full_streamlines[best_candidate_inv])
            #     if norms[best_candidate] <= args.epsilon:
            #         labeled[best_candidate] = True
            #         print('HEUREKA')
            #         continue
            # toc = time.time()
            print('Bruteforce labeling time', toc-tic)


        print('+ Unlabeled:', len(cluster_streamlines)-len(labeled_set))
        tic = time.time()
        for i in range(len(cluster_endpts)):
            csltuple = tuple(cluster_endpts[i])
            csltuple_inv = tuple(cluster_endpts_inv[i])
            if csltuple not in labeled_set and csltuple_inv not in labeled_set:
                not_found.append(cluster_streamlines[i])
                not_found_endpts.append(cluster_endpts[i])
                not_found_labels.append(cluster_label)
        toc = time.time()
        print('Unlabeled detection time', toc-tic)
        print()


    print('+++++ Save results in memory')
    tic = time.time()
    # Notfound data
    x_notfound = np.array(np.array(not_found))
    y_notfound = np.array(np.array(not_found_labels))
    save_dir = constants.data_proc_path+args.subject+'/'
    with open(save_dir+args.subject+'_notfound.npy', 'w+b') as f:
        np.save(f, x_notfound)
        np.save(f, y_notfound)
    if args.tck:
        save_streamlines(x_notfound, 
                        constants.data_raw_path+args.subject+'/'+args.ref,
                        save_dir+'tck/'+args.subject+'_notfound.tck')
    # Training data
    x_training = np.concatenate((full_streamlines, x_notfound), axis=0)
    y_training = np.concatenate((labels, y_notfound))
    with open(save_dir+args.subject+'.npy', 'w+b') as f:
        np.save(f, x_training)
        np.save(f, y_training)
    if args.tck:
        save_streamlines(x_training, 
                        constants.data_raw_path+args.subject+'/'+args.ref,
                        save_dir+'tck/'+args.subject+'.tck')
    # Labeled cluster data
    x_labeled = x_training[y_training != 0]
    y_labeled = y_training[y_training != 0]
    with open(save_dir+args.subject+'_labeled.npy', 'w+b') as f:
        np.save(f, x_labeled)
        np.save(f, y_labeled)
    if args.tck:
        save_streamlines(x_labeled, 
                        constants.data_raw_path+args.subject+'/'+args.ref,
                        save_dir+'tck/'+args.subject+'_labeled.tck')
    # Garbage data
    x_garbage = x_training[y_training == 0]
    with open(save_dir+args.subject+'_garbage.npy', 'w+b') as f:
        np.save(f, x_garbage)
    if args.tck:
        save_streamlines(x_garbage, 
                        constants.data_raw_path+args.subject+'/'+args.ref,
                        save_dir+'tck/'+args.subject+'_garbage.tck')
    toc = time.time()
    print('Save results time:', toc-tic)
    print()


    print('--------------------')
    print('    RESULTS')
    print('--------------------')
    TOC = time.time()
    print('Data generation time:', TOC-TIC)
    print('Full tracto size:', full_tracto_size)
    print('Cluster tractos size:', cluster_tracto_size)
    print('Labeled streamlines:', np.sum(labeled))
    print('Total missing labeled:', len(not_found))