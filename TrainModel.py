import turicreate as tc
import matplotlib.pyplot as plt
# Load sessions from preprocessed data
data = tc.SFrame('exercise_data.sframe')
dataUnfiltered = tc.SFrame('exercise_data_unfiltered.sframe')

# Train/test split by recording sessions
train, test = tc.activity_classifier.util.random_split_by_session(data,
                                                                  session_id='exp_id',
                                                                  fraction=1.)

# Create an activity classifier
model = tc.activity_classifier.create(train, session_id='exp_id', target='activity',
                                      prediction_window=125, max_iterations=10)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(train)
# target_map = {
#     'standing': 1.,
#     'shoulder_right_up': 2.,
#     'shoulder_right_down': 3.,
#     'shoulder_left_up': 4.,
#     'shoulder_left_down': 5.,
# }

target_map = {
    'shoulder_left': 1.,
    'shoulder_right': 2.,
    'standing': 3.,
}

pred = model.predict(data)
pred_id = pred.apply(lambda x: target_map[x])
target_id = data['activity'].apply(lambda x: target_map[x])
plt.plot(data['l_arm_r'])
plt.plot(data['r_arm_r'])
plt.plot(target_id-0.5, linewidth=2)
plt.plot(pred_id)
plt.show()


pred = model.classify(dataUnfiltered)
pred_id = pred['class'].apply(lambda x: target_map[x])
target_id = dataUnfiltered['activity_id']
plt.plot(dataUnfiltered['l_arm_r'])
plt.plot(dataUnfiltered['r_arm_r'])
plt.plot(target_id-0.5)
plt.plot(pred_id)
plt.show()


print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('hapt.model')

# Export for use in Core ML
model.export_coreml('ShoulderAbductionClassifier.mlmodel')