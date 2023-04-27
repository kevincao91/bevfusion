import numpy as np
import mmcv
import os

dst_dir = 'data/nuscenes_cidi_byd+cidi_bkjw'
A_src_dir = 'data/nuscenes_cidi_byd'
B_src_dir = 'data/nuscenes_cidi_bkjw'
A_info_path = os.path.join(A_src_dir, 'nuscenes_infos_train.pkl')
A_dbinfo_path = os.path.join(A_src_dir, 'nuscenes_dbinfos_train.pkl')
B_info_path = os.path.join(B_src_dir, 'nuscenes_infos_train.pkl')
B_dbinfo_path = os.path.join(B_src_dir, 'nuscenes_dbinfos_train.pkl')
info_out_path = os.path.join(dst_dir, 'nuscenes_infos_train.pkl')
dbinfo_out_path = os.path.join(dst_dir, 'nuscenes_dbinfos_train.pkl')

# info
A_info = mmcv.load(A_info_path)
B_info = mmcv.load(B_info_path)
A_info['infos'] = A_info['infos']
A_info['infos'] += B_info['infos']
mmcv.dump(A_info, file=info_out_path)
print('{} saved.'.format(info_out_path))

# dbinfo
A_dbinfo = mmcv.load(A_dbinfo_path)
B_dbinfo = mmcv.load(B_dbinfo_path)
for key in A_dbinfo:
    A_dbinfo[key] += B_dbinfo[key]
mmcv.dump(A_dbinfo, file=dbinfo_out_path)
print('{} saved.'.format(dbinfo_out_path))