import turicreate as tc
from glob import glob

data_dir = './Exercise Data/'

def find_label_for_containing_interval(intervals, index):
    containing_interval = intervals[:, 0][(intervals[:, 1] <= index) & (index <= intervals[:, 2])]
    if len(containing_interval) == 1:
        # print(containing_interval[0])
        return containing_interval[0]

# Load labels
labels = tc.SFrame.read_csv(data_dir + 'labels.csv', delimiter=',', header=True,
                            verbose=False)
# labels = labels.rename({'X1': 'exp_id', 'X2': 'user_id', 'X3': 'activity_id',
#                         'X4': 'start', 'X5': 'end'})
# labels
print(labels)

data_files = glob(data_dir + 'data*.csv')

# Load data
data = tc.SFrame()
files = sorted(data_files)
for data_file in files:
    exp_id = int(data_file[-5])
    print('--------------', exp_id, '----------')

    # Load accel data
    sf = tc.SFrame.read_csv(data_file, delimiter=',', header=True, verbose=False)
    # sf = sf.rename({'X1': 'acc_x', 'X2': 'acc_y', 'X3': 'acc_z'})
    sf['exp_id'] = exp_id

    # Calc labels
    exp_labels = labels[labels['exp_id'] == exp_id][['activity_id', 'start', 'end']].to_numpy()
    sf = sf.add_row_number()
    sf['activity_id'] = sf['id'].apply(lambda x: find_label_for_containing_interval(exp_labels, x))
    sf = sf.remove_columns(['id'])

    data = data.append(sf)


target_map = {
    1.: 'knee',
    2.: 'shoulder',

}

# Use the same labels used in the experiment
data = data.filter_by(list(target_map.keys()), 'activity_id')
data['activity'] = data['activity_id'].apply(lambda x: target_map[x])
data = data.remove_column('activity_id')

data.save('exercise_data.sframe')