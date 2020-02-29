import os, re, random
import numpy as np
import scipy.io as sio
import tensorflow as tf

network = 'acnn'    # choose from 'acnn', 'unet' and 'deeplab'
if network == 'unet':
    from unet import unet
elif network == 'deeplab':
    from deeplab import deeplab_v3plus
elif network == 'acnn':
    from acnn import acnn

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dataset = 'LV'  # choose from 'LV', 'RV' and 'Aorta'
lr = 0.005
cross_validation = 2
n_blocks = 64
l2regularization = True

write_prediction_result = True  # whether prediction map should be saved

# TODO specify paths for testing data
data_folder = '/data/XIAOYUN_ZHOU/Dataset/%s_2fold/' % dataset
data_test_path = data_folder + 'Test_%d/' % cross_validation
data_train_path = data_folder + 'Train_%d/' % cross_validation
data_eval_path = data_folder + 'Evaluation_%d/' % cross_validation
# TODO specify model save path
save_path = '/data/trained/%s/%s_%d_blocks%s/model_lr_%f_crossval_%s/' % (
    dataset, network, n_blocks, '_l2regularization' if l2regularization else '', lr, cross_validation)

test_image_list = sorted([name for name in os.listdir(data_test_path) if re.match('Test_Image_\\d+.mat', name)],
                         key=lambda path: int(path[path.find('Test_Image_') + 11:path.find('.mat')]))
test_label_list = sorted([name for name in os.listdir(data_test_path) if re.match('Test_Label_\\d+.mat', name)],
                         key=lambda path: int(path[path.find('Test_Label_') + 11:path.find('.mat')]))
test_num = len(test_image_list)

if dataset == 'Aorta':
    image_size = 512
else:
    image_size = 256
batch_size = 1
epoch = 1
feature_num = 16
kernel_size = 3
pool_size = 2
n_classes = 2

config_train = tf.ConfigProto()
config_train.gpu_options.allow_growth = True

images = tf.placeholder(dtype='float32', shape=[None, image_size, image_size, 1])
labels = tf.placeholder(dtype='float32', shape=[None, image_size, image_size, n_classes])
if network == 'unet':
    prediction, logits = unet(feature_in=images, labels=labels, n_blocks=n_blocks, kernel_size=kernel_size,
                              feature_num=feature_num, n_classes=n_classes, pool_size=pool_size)
elif network == 'deeplab':
    prediction, logits = deeplab_v3plus(images, labels, feature_num=feature_num, n_classes=n_classes)
else:
    prediction, logits = acnn(feature_in=images, labels=labels, n_blocks=n_blocks, kernel_size=kernel_size,
                              feature_num=feature_num, n_classes=n_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits))

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=2 * epoch)

logfile = open(save_path + 'testing_log.txt', 'w+')
ioufile = open(save_path + 'testing_result.txt', 'w+')
with tf.Session(config=config_train) as sess:
    sess.run(init_op)

    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(total_parameters)

    ckpt = tf.train.get_checkpoint_state(save_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    iou_list = []
    loss_list = []

    for i in range(test_num):
        image_path = data_test_path + test_image_list[i]
        label_path = data_test_path + test_label_list[i]

        image_np = sio.loadmat(image_path)['Image']
        image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], 1))
        label_np = sio.loadmat(label_path)['Label']
        label_np = np.reshape(label_np, (1, label_np.shape[0], label_np.shape[1], label_np.shape[2]))

        [pred, loss_value] = sess.run([prediction, loss], feed_dict={images: image_np, labels: label_np})

        # Calculate IoU
        iou_list.append(pred['IoU'])
        loss_list.append(loss_value)
        ioufile.write(str(pred['IoU']) + '\n')
        ioufile.flush()

        if not write_prediction_result:
            continue

        prediction_path = save_path + 'prediction/'
        if not os.path.exists(prediction_path):
            os.makedirs(prediction_path)
        image_idx = image_path[image_path.find('Test_Image_') + 11:image_path.find('.mat')]
        sio.savemat(prediction_path + str(image_idx) + '_prediction.mat', {'prediction': pred['probabilities']})
        sio.savemat(prediction_path + str(image_idx) + '_label.mat', {'label': label_np})

    average_iou = np.mean(iou_list)
    average_loss = np.mean(loss_list)
    msg = 'Average IoU: %f' % float(average_iou)
    print(msg)
    logfile.write(msg + '\n')

logfile.close()
ioufile.close()
