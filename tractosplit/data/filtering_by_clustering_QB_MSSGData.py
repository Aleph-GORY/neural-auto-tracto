
from dipy.io.streamline import load_tractogram,save_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import ResampleFeature
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.io.stateful_tractogram import StatefulTractogram,Space

import numpy as np
import os
import sys
import glob
import random
random.seed(2019)
        

def doClusterQB(streamlines,qb_th,n_distance,min_size_cluster_loc):
        
    #QB CLUSTERING - - - - - - - 
    feature = ResampleFeature(nb_points=12)

    #if n_distance == 1:
    metric = AveragePointwiseEuclideanMetric(feature)
    #else:
        # Not implemented

    qb = QuickBundles(threshold=qb_th, metric=metric)
    clusters = qb.cluster(streamlines)

    #END QB CLUSTERING - - - - - - -         
    sumNS = 0
    nBigClusters = 0;
    for i in range(0,len(clusters)):
        #print("<", len(clusters[i]),">", end = '')
        sumNS = sumNS + len(clusters[i])
        if len(clusters[i]) >= min_size_cluster_loc:
               nBigClusters += 1
    return clusters,sumNS,nBigClusters


def saveFiles(streamlines_cleaned,streamlines_garbage,full_name_reference,fOName_cleaned,fOName_garbage):
    
    stf = StatefulTractogram(streamlines_cleaned,full_name_reference,Space.RASMM)
    save_tractogram(stf, fOName_cleaned)

    stf = StatefulTractogram(streamlines_garbage,full_name_reference,Space.RASMM)
    save_tractogram(stf, fOName_garbage)   

def cleanWMPProcess():

    print('Running batch! onn:')    
    global subject_str
    global fname
    global ratioGoal
    global epsilonGoal
    global n_distance_i
    
    subjectNames = glob.glob(resultsFolder+"/*")
    print(resultsFolder+"/*")
    strParamsFilter = '_rg_' + str(ratioGoal) #+ '_er_' + str(epsilonGoal) 

    for i in range(0,len(subjectNames)):
        if subject_str != '' and subjectNames[i][len(subjectNames[i])-6:] != subject_str:
            print("x-> " + subjectNames[i])
            continue
        
        print('\n ******************************* \n')
        print(subjectNames[i])
        full_name_reference = dataFolder+'/'+ subjectNames[i][len(subjectNames[i])-6:] +'/t1.nii.gz'
        #print(full_name_reference)
        output_subject_path = output_clean + '/' +subjectNames[i][len(subjectNames[i])-6:];
        if not os.path.exists(output_subject_path):
            os.makedirs(output_subject_path)
        
        tckfiles = glob.glob(subjectNames[i] + '/*_20p.tck')           
        str_n_distance_i = '_nd_' + str(n_distance_i);

        for k in range(0,len(tckfiles)):

            print('\n --------------------- \n')
            if len(fname)> 0 and fname != tckfiles[k][len(tckfiles[k])-len('_20p.tck')-len(fname):len(tckfiles[k])-len('_20p.tck')]:
                print(fname + "," + tckfiles[k][len(tckfiles[k])-len('_20p.tck')-len(fname):len(tckfiles[k])-len('_20p.tck')])
                continue            
        
            print('-> '+tckfiles[k])
            #fNameGeneric = tckfiles[k][0:len(tckfiles[k])-4];
            fNameGeneric = output_subject_path + '/' +tckfiles[k][0:len(tckfiles[k])-4].split('/').pop()
            
            tractogram = load_tractogram(tckfiles[k], full_name_reference, bbox_valid_check=False)
            print('-- Streamlines in the input: ',len(tractogram.streamlines))
            streamlines = tractogram.streamlines;
                           
            final_streamlines_cleaned = streamlines
            final_streamlines_garbage = []
            final_ratioClean = 1.0
            final_distanceRatioClean = 1e99;
            final_str_min_size_cluster_i = '';
            final_str_qb_th_i = ''

            if len(streamlines) == 0:
                print('0 streamlines, nothing to do');
                fname_qb_ok =   fNameGeneric + '_qb_clean.tck'
                fname_qb_nook = fNameGeneric + '_qb_garbage.tck'
                saveFiles(final_streamlines_cleaned,final_streamlines_garbage,full_name_reference,fname_qb_ok,fname_qb_nook)
                continue
            

            for qb_th_i in np.arange(0.5,30+1,0.5): #finner
                str_qb_th_i = '_dth_' + str(qb_th_i)
                
                clusters,sumNS,nBigClustersNoUsar = doClusterQB(streamlines,qb_th_i,n_distance_i,10);
                    
                for min_size_cluster_i in range(2,25+1,1): #finner
                    str_min_size_cluster_i = '_cz_' + str(min_size_cluster_i)
                                                                        
                    streamlines_cleaned = [];
                    streamlines_garbage = [];
                                
                    for i in range(0,len(clusters)):
                        if len(clusters[i])>=min_size_cluster_i:
                            for st in streamlines[clusters[i].indices]:
                                streamlines_cleaned.append(st)
                        else:
                            for st in streamlines[clusters[i].indices]:
                                streamlines_garbage.append(st)
                    ratioClean = len(streamlines_cleaned)/len(streamlines)

                    #if ratioClean>ratioGoal-epsilonGoal and ratioClean<ratioGoal+epsilonGoal:
                    if abs(ratioClean-ratioGoal) < final_distanceRatioClean:
                        final_distanceRatioClean = abs(ratioClean-ratioGoal)
                        final_streamlines_cleaned = streamlines_cleaned
                        final_streamlines_garbage = streamlines_garbage
                        final_ratioClean = ratioClean
                        final_str_qb_th_i = str_qb_th_i
                        final_str_min_size_cluster_i = str_min_size_cluster_i

                        
            str_ratioClean = '_rc_' + str(final_ratioClean)                        
            #fname_qb_ok =   fNameGeneric + '_qb_clean'   + str_n_distance_i + final_str_qb_th_i + final_str_min_size_cluster_i + strParamsFilter + str_ratioClean + '.tck'
            #fname_qb_nook = fNameGeneric + '_qb_garbage' + str_n_distance_i + final_str_qb_th_i + final_str_min_size_cluster_i + strParamsFilter + str_ratioClean + '.tck'
            fname_qb_ok =   fNameGeneric + '_qb_clean.tck'
            fname_qb_nook = fNameGeneric + '_qb_garbage.tck'
                    
            saveFiles(final_streamlines_cleaned,final_streamlines_garbage,full_name_reference,fname_qb_ok,fname_qb_nook)
                    

# ######################################################################
# MAIN script, se mandan 3 parametros 
# execute $ ipython cluster_human_tracks_QB_ARM.py distance(1,2) n_subject(1-6) Tracto(cadena, ejemplo UF_L)
# cadena vacia en n_subject y en tracto significa "procesa todos"
# ######################################################################


clear = lambda: os.system('clear') #on Windows System
clear()


#
#Alonso DSSP MSSG
dataFolder = '/home/alonso/NAS/NAS/datos/DW_Human/DSSP_workshop/structure'
resultsFolder = '/home/alonso/NAS/NAS/datos/DW_Human/DSSP_workshop/clean_semiauto'
output_clean = '/home/alonso/NAS/NAS/datos/DW_Human/DSSP_workshop/cleanQB'

global subject_str
global fname
global ratioGoal
global epsilonGoal
global n_distance_i

if len(sys.argv) > 1:
    n_distance_i = int(sys.argv[1])  #'1';
else:
    n_distance_i = 1 #average distance default

if len(sys.argv) > 2:
    subject_str = str(sys.argv[2])  #'157942';
else:
    subject_str = ''

if len(sys.argv) > 3:
    fname = str(sys.argv[3])  #'1';
else:
    fname = '' #AC


print('subject_str',subject_str)
print('fname',fname)

ratioGoal = 0.9;
#epsilonGoal =  0.05

cleanWMPProcess()

    