import os
import pandas as pd
from decision_tree import ID3


""" create Images dir for save visualizations """
current_path = os.getcwd()
results_dir = os.path.join(current_path, 'Results')
is_exist = os.path.exists(results_dir)
if not is_exist:
    os.mkdir(results_dir)

images_dir = os.path.join(results_dir, 'VisualizeDecisionTrees')
is_exist = os.path.exists(images_dir)
if not is_exist:
    os.mkdir(images_dir)

accuracy_dir = os.path.join(results_dir, 'Accuracy')
is_exist = os.path.exists(accuracy_dir)
if not is_exist:
    os.mkdir(accuracy_dir)


train_dir = os.path.join(current_path, 'DataSet/Train')
test_dir = os.path.join(current_path, 'DataSet/Test')

result_accuracy = []
for training_file in os.listdir(train_dir):
    decision_tree = ID3()
    train_file_address = f'{train_dir}\\{training_file}'
    tarin = pd.read_csv(train_file_address)
    train_file_name = training_file.split('.')[0]
    test = pd.read_csv('./DataSet/Test/monks_test.csv')
    target_class_name = tarin.columns[-1]
    record_length = len(tarin)
    attributes = list(tarin.columns)[1:-1]
    decision_tree.fit(tarin, attributes)
    dot_tree = decision_tree.export_graphviz()
    dot_tree.write_png(f'{images_dir}/{train_file_name}_decision_tree.png')
    classified_as = decision_tree.predict(test)
    true_class = test[target_class_name]
    true_class = list(true_class)
    acc = decision_tree.accuracy(classified_as, true_class)
    result_accuracy.append([train_file_name, str(acc)])

df = pd.DataFrame(result_accuracy, columns=['dataset', 'accuracy'])
df.to_csv(f"{accuracy_dir}\\test_dataset_accuracy.csv")





