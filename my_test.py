import torch

def rename_layers_in_weights(weights_path, old_name, new_name):
    # 加载原始权重文件
    state_dict = torch.load(weights_path)
    state_dict_weights = state_dict['state_dict']
    # 遍历state_dict字典，找到含有old_name的键并替换为new_name
    new_state_dict = {}
    for key, value in state_dict_weights.items():
        if old_name in key:
            new_key = key.replace(old_name, new_name)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    # 保存新的权重文件
    state_dict['state_dict'] = new_state_dict
    new_weights_path = weights_path.replace('.pth', '_renamed.pth')
    torch.save(state_dict, new_weights_path)

# 定义要更改的.pth权重文件路径、旧名称和新名称
weights_path = 'work_dirs/wider_naive_litehrnetv3_18_coco_256x192_3gpu/epoch_210.pth'
old_name = 'branch2'
new_name = 'branch'

# 更改层名称并保存为新的.pth权重文件
rename_layers_in_weights(weights_path, old_name, new_name)
