import numpy as np
import mmcv
import os

root_dir = 'data/nuscenes_cidi_bkjw'
# info_path = 'data/nuscenes/nuscenes_infos_val.pkl'
info_path = os.path.join(root_dir, 'nuscenes_infos_val.pkl')
out_path = os.path.join(root_dir, 'nuscenes_infos_val.json')

data = mmcv.load(info_path)

# print(len(data['infos']), '=======')
# rotation_list = []
# for info in data['infos']:
#     print(len(info['gt_boxes']), end="== ")
#     for box in info['gt_boxes']:
#         print(box)
#         rotation_list.append(box[-1])

# print('max x:', max(rotation_list),', min x:',min(rotation_list))


# max x: 1.5704445686377975 , min x: -4.712339399530078


mmcv.dump(data, file=out_path, indent=4)
print('{} saved.'.format(out_path))