from imutils import paths
import os, random, shutil

# building dataset paths
train_paths = list(paths.list_images('dataset/train'))
test_paths = list(paths.list_images('dataset/test'))
random.seed(42)
random.shuffle(train_paths)
random.shuffle(test_paths)

datasets = [('train', train_paths, 'data/train'),
            ('test', test_paths, 'data/test')]

labels = ['Closed','Open']

for (type, src_paths, target_path) in datasets:
    print(f'Building {type} set')
    if not os.path.exists(target_path):
            os.makedirs(target_path)

    for path in src_paths:
        file = path.split(os.path.sep)[-1]
        label = str(labels.index(path.split(os.path.sep)[-2]))
        label_path = os.path.sep.join([target_path, label])

        if not os.path.exists(label_path):
            os.makedirs(label_path)

        new_path = os.path.sep.join([label_path, file])
        shutil.copy2(path,new_path)
