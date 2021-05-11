import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import os

train_path = r'../datasets/aerial-cactus-identification/train/'
test_path = r'../datasets/aerial-cactus-identification/test/'
labels = pd.read_csv('../datasets/aerial-cactus-identification/train.csv')

print('Label Distribution: \n' + str(labels['has_cactus'].value_counts()))

# label = 'Has Cactus', 'No Cactus'
# plt.figure(figsize=(16, 16))
# plt.pie(labels.groupby('has_cactus').size(), labels=label, autopct='%0.1f%%', startangle=90)
# plt.show()

# fig, ax = plt.subplots(1, 5, figsize=(15, 3))
#
# for i, idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):
#     path = os.path.join(train_path, idx)
#     ax[i].imshow(img.imread(path))
#
# for i, idx in enumerate(labels[labels['has_cactus'] == 1]['id'][:5]):
#     path = os.path.join(train_path, idx)
#     ax[i].imshow(img.imread(path))
# plt.show()
