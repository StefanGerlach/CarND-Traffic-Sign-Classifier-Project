import packages.dataset_ultils as util


train_x, train_y, valid_x, valid_y, test_x, test_y = util.load_datasets('datasets/train.p', 'datasets/valid.p', 'datasets/test.p')
translations = util.load_translation_file('signnames.csv')

# util.visualize_dataset(train_x, train_y, translations)
# util.visualize_dataset(valid_x, valid_y, translations)
# util.visualize_dataset(test_x, test_y, translations)

i = 1