import turicreate as tc

# Load sessions from preprocessed data
data = tc.SFrame('exercise_data.sframe')

# Train/test split by recording sessions
train, test = tc.activity_classifier.util.random_split_by_session(data,
                                                                  session_id='exp_id',
                                                                  fraction=0.8)

# Create an activity classifier
model = tc.activity_classifier.create(train, session_id='exp_id', target='activity',
                                      prediction_window=350,max_iterations=10)

# Evaluate the model and save the results into a dictionary
metrics = model.evaluate(test)
print(metrics['accuracy'])

# Save the model for later use in Turi Create
model.save('hapt.model')

# Export for use in Core ML
model.export_coreml('ShoulderExerciseClassifier.mlmodel')