import rotnet_config
import mnist_classifier_config
from dataloader import DataLoader, GenericDataset

def load_data_for_rotnet():
    dataset_train = GenericDataset(
    rotnet_config.data_train_opt['dataset_name'],
    rotnet_config.data_train_opt['split']
    )

    dataset_test = GenericDataset(
        rotnet_config.data_test_opt['dataset_name'],
        rotnet_config.data_test_opt['split']
    )

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=rotnet_config.data_train_opt['batch_size'],
        unsupervised=rotnet_config.data_train_opt['unsupervised'],
        epoch_size=rotnet_config.data_train_opt['epoch_size'],
        num_workers=rotnet_config.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=rotnet_config.data_test_opt['batch_size'],
        unsupervised=rotnet_config.data_test_opt['unsupervised'],
        epoch_size=rotnet_config.data_test_opt['epoch_size'],
        num_workers=rotnet_config.num_workers,
        shuffle=False)
    # data standardization already done in DataLoader
    return dloader_train, dloader_test

def load_data_for_mnist():
    dataset_train = GenericDataset(
    mnist_classifier_config.data_train_opt['dataset_name'],
    mnist_classifier_config.data_train_opt['split'],
    num_imgs_per_cat=mnist_classifier_config.data_train_opt['num_imgs_per_cat']
    )

    dataset_test = GenericDataset(
        mnist_classifier_config.data_test_opt['dataset_name'],
        mnist_classifier_config.data_test_opt['split']
    )

    dloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=mnist_classifier_config.data_train_opt['batch_size'],
        unsupervised=mnist_classifier_config.data_train_opt['unsupervised'],
        epoch_size=mnist_classifier_config.data_train_opt['epoch_size'],
        num_workers=mnist_classifier_config.num_workers,
        shuffle=True)

    dloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=mnist_classifier_config.data_test_opt['batch_size'],
        unsupervised=mnist_classifier_config.data_test_opt['unsupervised'],
        epoch_size=mnist_classifier_config.data_test_opt['epoch_size'],
        num_workers=mnist_classifier_config.num_workers,
        shuffle=False)
    # data standardization already done in DataLoader
    return dloader_train, dloader_test

def nbr_total_pred(preds, targets):
  _, predicted = preds.max(1)
  return predicted.eq(targets).sum().item()