import turicreate as tc
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

data_dir = './Exercise Data/'


def find_label_for_containing_interval(intervals, index):
    containing_interval = intervals[:, 0][(intervals[:, 1] <= index) & (index <= intervals[:, 2])]
    if len(containing_interval) == 1:
        # print(containing_interval[0])
        return containing_interval[0]


def unwrap_data(sf):
    col_names = np.array(sf.column_names())
    col_ind = np.array(range(col_names.__len__()))
    unwrap_ind = col_ind[np.floor(col_ind/3) % 2 != 0]
    unwrap_col = list(col_names[unwrap_ind])
    # print(unwrap_col)
    # print(unwrap_col.__len__())
    for col_name in unwrap_col:
        # plt.plot(sf[str(col_name)])
        sf[str(col_name)] = np.unwrap(sf[str(col_name)])
        # plt.plot(sf[str(col_name)])
        plt.show()


# Load labels
labels = tc.SFrame.read_csv(data_dir + 'labels_3_exercises.csv', delimiter=',', header=True,
                            verbose=False, column_type_hints=[int, int, float, int, int])
# labels = labels.rename({'X1': 'exp_id', 'X2': 'user_id', 'X3': 'activity_id',
#                         'X4': 'start', 'X5': 'end'})
# labels
print(labels)

data_files = glob(data_dir + 'shoulder*.csv')
sel_col = ['l_arm_r','l_arm_p','l_elbow_x','l_elbow_y','l_wrist_x','l_wrist_y','r_arm_r','r_arm_p','r_elbow_x','r_elbow_y','r_wrist_x','r_wrist_y']
# Load data
data = tc.SFrame()
files = sorted(data_files)
for data_file in files:
    exp_id = int(data_file[-7])
    print('--------------', exp_id, '----------')

    # Load accel data
    sf = tc.SFrame.read_csv(data_file, delimiter=',', header=True, verbose=False, column_type_hints=float)
    col_names = sf.column_names()
    sf = sf.remove_columns(col_names[-42:])
    sf = sf.remove_columns(col_names[0:3])
    sf = sf[sel_col]
    # sf = sf.rename({'X1': 'acc_x', 'X2': 'acc_y', 'X3': 'acc_z'})
    sf['exp_id'] = exp_id

    # unwrap_data(sf)

    # Calc labels
    exp_labels = labels[labels['exp_id'] == exp_id][['activity_id', 'start', 'end']].to_numpy()
    sf = sf.add_row_number()
    sf['activity_id'] = sf['id'].apply(lambda x: find_label_for_containing_interval(exp_labels, x))
    sf = sf.remove_columns(['id'])

    data = data.append(sf)

print(sf.shape)

# target_map = {
#     1.: 'standing',
#     2.: 'shoulder_right_up',
#     3.: 'shoulder_right_down',
#     4.: 'shoulder_left_up',
#     5.: 'shoulder_left_down'
#
# }

target_map = {
    1.: 'shoulder_left',
    2.: 'shoulder_right',
    3.: 'standing',
}

# Use the same labels used in the experiment
data.save('exercise_data_unfiltered.sframe')
data = data.filter_by(list(target_map.keys()), 'activity_id')
data['activity'] = data['activity_id'].apply(lambda x: target_map[x])




plt.plot(data['l_arm_r'])
plt.plot(data['r_arm_r'])
plt.plot(data['activity_id'])
plt.show()

data = data.remove_column('activity_id')

data.save('exercise_data.sframe')
