import os, re, time, random
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
restore = False     # whether to resume training from the last checkpoint

dataset = 'LV'  # choose from 'LV', 'RV', 'Aorta'

cross_validation = 2
# TODO specify dataset folders
data_folder = '/data2/XIAOYUN_ZHOU/Dataset/%s_2fold/' % dataset
data_train_path = data_folder + 'Train_%d/' % cross_validation
data_eval_path = data_folder + 'Evaluation_%d/' % cross_validation
data_test_path = data_folder + 'Test_%d/' % cross_validation
train_image_list = sorted([name for name in os.listdir(data_train_path) if re.match('Train_Image_\\d+.mat', name)])
train_label_list = sorted([name for name in os.listdir(data_train_path) if re.match('Train_Label_\\d+.mat', name)])
eval_image_list = sorted([name for name in os.listdir(data_eval_path) if re.match('Evaluation_Image_\\d+.mat', name)])
eval_label_list = sorted([name for name in os.listdir(data_eval_path) if re.match('Evaluation_Label_\\d+.mat', name)])
test_image_list = sorted([name for name in os.listdir(data_test_path) if re.match('Test_Image_\\d+.mat', name)])
test_label_list = sorted([name for name in os.listdir(data_test_path) if re.match('Test_Label_\\d+.mat', name)])
train_num = len(train_image_list)
eval_num = len(eval_image_list)
test_num = len(test_image_list)

if dataset == 'Aorta':
    image_size = 512
else:
    image_size = 256
batch_size = 1
epoch = 1
n_blocks = 64
feature_num = 16
kernel_size = 3
pool_size = 2
n_classes = 2
step_show = 10
step_eval = 100

config_train = tf.ConfigProto()
config_train.gpu_options.allow_growth = True

images = tf.placeholder(dtype='float32', shape=[None, image_size, image_size, 1])
labels = tf.placeholder(dtype='float32', shape=[None, image_size, image_size, n_classes])
# network initialization
if network == 'unet':
    prediction, logits = unet(feature_in=images, labels=labels, n_blocks=n_blocks, kernel_size=kernel_size,
                              feature_num=feature_num, n_classes=n_classes, pool_size=pool_size)
elif network == 'deeplab':
    prediction, logits = deeplab_v3plus(images, labels, feature_num=feature_num, n_classes=n_classes)
else:
    prediction, logits = acnn(feature_in=images, labels=labels, n_blocks=n_blocks, kernel_size=kernel_size,
                              feature_num=feature_num, n_classes=n_classes)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=logits)) \
       + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]) * 0.001
training_loss_summary = tf.summary.scalar('Training Loss', loss)
eval_loss_summary = tf.summary.scalar('Evaluation Loss', loss)
eval_iou_summary = tf.summary.scalar('Evaluation IoU', prediction['IoU'])

for lr in [0.1, 0.05, 0.01, 0.005]:

    boundaries = [2000, 4000]
    values = [lr, lr / 5, lr / 25]
    # TODO specify model save path
    save_path = '/data/trained/%s/%s_%d_blocks_l2regularization/model_lr_%f_crossval_%s/' % (
        dataset, network, n_blocks, lr, cross_validation)
    if not restore:
        os.system('rm -rf %s' % save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # record training information in a log file
    logfile = open(save_path + 'training_log.txt', 'w+')
    logfile.write('\n********************* LR = %f *********************\n' % lr)

    # training setup
    global_step = tf.Variable(0, trainable=False)
    lr_t = tf.train.piecewise_constant(x=global_step, boundaries=boundaries, values=values)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr_t, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    training_lr_summary = tf.summary.scalar('Learning Rate', lr_t)
    # tensorboard setup
    merged_summary_training = tf.summary.merge([training_loss_summary, training_lr_summary], name='training info')
    merged_summary_eval = tf.summary.merge([eval_loss_summary, eval_iou_summary], name='eval info')

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=2 * epoch)
    with tf.Session(config=config_train) as sess:
        tf.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=save_path, name='Model')
        training_summary_writer = tf.summary.FileWriter(save_path + 'train/', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(save_path + 'eval/', sess.graph)
        sess.run(init_op)
        total_loss = 0

        # number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('total number of parameters: %d' % total_parameters)

        if restore:
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

        start_time = time.time()
        for step in range(train_num * epoch):
            image_id_list = random.sample(range(train_num), batch_size)
            image_batch = []
            label_batch = []
            # read a batch of images
            for image_id in image_id_list:
                image_np = sio.loadmat(data_train_path + train_image_list[image_id])['Image']
                image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], 1))
                image_batch.append(image_np)
                label_np = sio.loadmat(data_train_path + train_label_list[image_id])['Label']
                label_np = np.reshape(label_np,
                                      (1, label_np.shape[0], label_np.shape[1], label_np.shape[2]))
                label_batch.append(label_np)
            images_np = np.concatenate(image_batch, axis=0)
            labels_np = np.concatenate(label_batch, axis=0)

            # network training
            _, loss_value, training_summary, global_step_show, pred, Lr_show = sess.run(
                [train_op, loss, merged_summary_training, global_step, prediction, lr_t],
                feed_dict={images: images_np, labels: labels_np})

            # display every $(step_show) step
            if (step + 1) % step_show == 0:
                total_loss += loss_value
                total_loss = total_loss / step_show
                msg = 'Step: %d, Learning rate: %f, Loss: %f, Running time: %f' % (
                    global_step_show, Lr_show, total_loss, time.time() - start_time)
                print(msg)
                logfile.write(msg + '\n')

                total_loss = 0

                training_summary_writer.add_summary(training_summary, global_step=global_step_show)
                training_summary_writer.flush()
                start_time = time.time()
            else:
                total_loss += loss_value

            # Evaluation after every 1000 steps
            if ((step + 1) % step_eval) == 0:
                saver.save(sess, save_path + 'Model', global_step=global_step)
                print('-----------------Evaluation----------------------')
                logfile.write('------------------Evaluation---------------------\n')

                iou_list = []
                loss_list = []
                eval_image_id_list = random.sample(range(eval_num), eval_num)
                for eval_id in eval_image_id_list:
                    image_path = data_eval_path + eval_image_list[eval_id]
                    label_path = data_eval_path + eval_label_list[eval_id]

                    image_np = sio.loadmat(image_path)['Image']
                    image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], 1))
                    label_np = sio.loadmat(label_path)['Label']
                    label_np = np.reshape(label_np, (1, label_np.shape[0], label_np.shape[1], label_np.shape[2]))

                    [pred, loss_value, eval_summary] = sess.run([prediction, loss, merged_summary_eval],
                                                                feed_dict={images: image_np, labels: label_np})

                    # Calculate IoU
                    iou_list.append(pred['IoU'])
                    loss_list.append(loss_value)

                average_iou = np.mean(iou_list)
                average_loss = np.mean(loss_list)

                eval_summary_writer.add_summary(eval_summary, global_step=global_step_show)
                eval_summary_writer.flush()
                msg = 'epoch %d, Average evaluation loss: %f, Average evaluation IoU: %f' % (
                    step // train_num + 1, float(average_loss), float(average_iou))
                print(msg)
                logfile.write(msg + '\n')
                print('-------------------------------------------------')
                logfile.write('-------------------------------------------------\n\n')
                logfile.flush()
                start_time = time.time()

        # test in the end
        ckpt = tf.train.get_checkpoint_state(save_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(test_num):
            image_path = data_test_path + test_image_list[i]
            label_path = data_test_path + test_label_list[i]

            image_np = sio.loadmat(image_path)['Image']
            image_np = np.reshape(image_np, (1, image_np.shape[0], image_np.shape[1], 1))
            label_np = sio.loadmat(label_path)['Label']
            label_np = np.reshape(label_np, (1, label_np.shape[0], label_np.shape[1], label_np.shape[2]))

            [pred, ] = sess.run([prediction], feed_dict={images: image_np, labels: label_np})

            # save inference results
            prediction_path = save_path + 'prediction/'
            if not os.path.exists(prediction_path):
                os.makedirs(prediction_path)
            sio.savemat(prediction_path + str(i) + '_prediction.mat', {'prediction': pred['probabilities']})
            sio.savemat(prediction_path + str(i) + '_label.mat', {'label': label_np})
    logfile.close()
