import os, sys
import numpy as np
import tensorflow as tf
import pdb

os.chdir('../../') # カレントディレクトリを一階層上へ
print(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(__file__))) #パスの追加
import synchronization.functions.utility as my_utility

def dict_to_matrix(dict_data):
    """ jsonから読み込んだdictからmatrixを復元する (jsonの形式はreadmeを参照) """
    input_images = []
    output_labels = []
    for i in range(len(dict_data)):
        data = dict_data[str(i)]
        for d in data:
            image = np.array(d['Input'])
            label = np.array(d['Output'])
            input_images.append(image)
            output_labels.append(label)
    return input_images, output_labels

def flatten_images(images, labels):
    """ 2次元のデータを1次元化する """
    tmp_images = []
    tmp_labels = []
    for image, label in zip(images, labels):
        tmp_array = np.ravel(image)
        if (len(tmp_array) == 1080):
            tmp_images.append(tmp_array)
            tmp_labels.append(label)
    return tmp_images, tmp_labels

def shuffle_data(images, labels):
    """ データをシャッフルする
        images: list    入力画像のリスト
        labels: list    出力画像のリスト """
    np.random.seed(20210101) # 乱数シードを固定
    idx = np.arange(0 , len(labels)) # [0, 1, ..., len(labels)] までのndarrayを作成
    np.random.shuffle(idx)
    images_shuffle = [images[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return images_shuffle, labels_shuffle

def split_data(images, labels, prop=0.8):
    """ データを訓練データとテストデータに分割する """
    idx = int(len(labels) * prop)
    train_set_x, train_set_t = images[:idx], labels[:idx]
    test_set_x, test_set_t = images[idx:], labels[idx:]
    return train_set_x, train_set_t, test_set_x, test_set_t

if __name__ == "__main__":
    recordjsonfile = './synchronization/detection_model/data.json'
    dict_data = my_utility.load_json(recordjsonfile)
    images, labels = dict_to_matrix(dict_data)
    images, labels = shuffle_data(images, labels)
    images, labels = flatten_images(images, labels)
    train_X, train_y, test_X, test_y = split_data(images, labels)
    
    # --- 1段目のコンボリューション層とプーリング層を定義する ---
    num_filters1 = 1
    x = tf.placeholder(tf.float32, [None, 1080])
    x_image = tf.reshape(x, [-1, 72, 15, 1])

    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, num_filters1], stddev=0.1)) # 縦3, 横3，入力レイヤー数1, 出力レイヤー数1のフィルター
    h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 3, 3, 1], padding='SAME') # 縦3, 横3ごとにスライド
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
    h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)

    # --- 2段目のコンボリューション層とプーリング層を定義する ---
    num_filters2 = 5
    W_conv2 = tf.Variable(tf.truncated_normal([3, 1, num_filters1, num_filters2], stddev=0.1)) # 縦3, 横1，入力レイヤー数1, 出力レイヤー数1のフィルター
    h_conv2 = tf.nn.conv2d(h_conv1_cutoff, W_conv2, strides=[1, 1, 5, 1], padding='SAME') # 縦1, 横5ごとにスライド
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
    h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2) # b_conv2にもreluを適用することでピクセルのカットオフ値もパラメータ最適化の対象とする
    h_pool2 =tf.nn.max_pool(h_conv2_cutoff, ksize=[1,3,5,1], strides=[1,3,5,1], padding='SAME')

    # --- 2段目のプーリング層からの出力に対して全結合層を通す ---
    h_pool2_flat = tf.reshape(h_pool2, [-1, 8*1*num_filters2])

    num_units1 = 8*1*num_filters2
    num_units2 = 8

    w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
    b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
    hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

    keep_prob = tf.placeholder(tf.float32) # dropoutの割合を設定
    hidden2_drop = tf.nn.dropout(hidden2, keep_prob) # dropoutを設定

    w0 = tf.Variable(tf.zeros([num_units2, 1]))
    b0 = tf.Variable(tf.zeros([1]))
    p = tf.nn.sigmoid(tf.matmul(hidden2_drop, w0) + b0)

    # --- 損失関数を定義 ---
    t = tf.placeholder(tf.float32, [None, 1])
    loss = -tf.reduce_sum(t * tf.log(p) + (1 - t) * tf.log(1 - p))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    correct_prediction = tf.equal(tf.sign(p-0.5), tf.sign(t-0.5))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    # saver = tf.train.Saver()
    num_split = 4
    num_batch = 0
    for i in range(1, 2001):
        # ミニバッチ勾配降下法
        if (num_batch == num_split):
            num_batch = 0
        idx1 = int(len(test_y) * (num_batch / num_split))
        idx2 = int(len(test_y) * ((num_batch + 1) / num_split))
        batch_xs, batch_ts = train_X[idx1:idx2], train_y[idx1:idx2]
        sess.run(train_step, feed_dict={x:batch_xs, t:batch_ts, keep_prob:0.5})
        # テストデータでの評価を行う
        if i % 200 == 0:
            loss_val, acc_val = sess.run([loss, accuracy], feed_dict={x:test_X, t:test_y, keep_prob:1.0})
            print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
            pdb.set_trace()
            # saver.save(sess, 'cnn_session', global_step=i)
        num_batch += 1
        