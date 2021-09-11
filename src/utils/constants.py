# PATHS
data_raw_path = 'data/raw/'
data_proc_path = 'data/processed/'
lstm_path = 'models/LSTM/'

# FEATURES
clusters = {
    'names' : [ 'AC', 'AF_L', 'AF_R', 'CC_MID', 'CGFP_L', 'CGFP_R', 'CGH_L', 'CGH_R', 'CGR_L', 'CGR_R',
                'CG_L', 'CG_R', 'CST_L', 'CST_R', 'FA_L', 'FA_R', 'FMA', 'FMI', 'FX_L', 'FX_R', 
                'IFOF_L', 'IFOF_R', 'ILF_L', 'ILF_R', 'MLF_L', 'MLF_R', 'OR_L', 'OR_R', 'SLF_L', 'SLF_R', 
                'TAPETUM', 'UF_L', 'UF_R', 'VOF_L', 'VOF_R' ],
    'size' : 35
}
def get_clusters(subject):
    paths = [ data_raw_path+subject+'/'+cname+'_20p.tck' for cname in clusters['names'] ]
    output = {
        'names' : clusters['names'],
        'paths' : paths
    }
    return output

# MODELS
lstm_prefix = 'lstm-ckpt'
garbage_percent = 0.5
labeled_percent = 1