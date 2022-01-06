batch_size   = 32
num_workers = 4

config = {}
# set the parameters related to the training and testing set
data_train_opt = {} 
data_train_opt['batch_size'] = batch_size
data_train_opt['unsupervised'] = False
data_train_opt['epoch_size'] = 10 * 6000
data_train_opt['random_sized_crop'] = False
data_train_opt['dataset_name'] = 'mnist'
data_train_opt['split'] = 'train'
data_train_opt['num_imgs_per_cat'] = 10 #number images per category. 10 => 100 labels for training

data_test_opt = {}
data_test_opt['batch_size'] = batch_size
data_test_opt['unsupervised'] = False
data_test_opt['epoch_size'] = None
data_test_opt['random_sized_crop'] = False
data_test_opt['dataset_name'] = 'mnist'
data_test_opt['split'] = 'test'

config['data_train_opt'] = data_train_opt
config['data_test_opt']  = data_test_opt
config['max_num_epochs'] = 10

feat_net_opt = {'num_classes': 4, 'num_stages': 4, 'use_avg_on_conv3': False}

cls_net_optim_params = {'optim_type': 'sgd', 'lr': 0.03, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True}
cls_net_opt = {'num_classes':10, 'nChannels':192, 'cls_type':'NIN_ConvBlock3'}
config['out_feat_keys'] = ['conv2']

