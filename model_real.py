import numpy as np
import tensorflow as tf
import os
import glob
import cv2
idx_in_epoch = 0
temp=[]

def read_test_image() :
    input_path=os.getcwd()+'/aligned_img/test_Gang'
    img_list=os.listdir(input_path)
    test_img=[]
    test_label=[]

    for img in img_list :
        img_path=os.path.join(input_path, img)
        img_arr=cv2.imread(img_path)

        test_img.append(img_arr)
        test_label.append(0)

    test_label=np.array(test_label).reshape(-1, 1)
    return np.array(test_img), test_label

def read_image() :
    path=os.getcwd()+'/aligned_img'
    class_list=os.listdir(path)
    train_img=[]
    train_label=[]
    for index in range(len(class_list)-2) :
        input_path=os.path.join(path, class_list[index])
        img_list=os.listdir(input_path)

        for img in img_list :
            img_path=os.path.join(input_path, img)
            img_arr = cv2.imread(img_path, 1)
            train_img.append(img_arr)
            train_label.append(index)

    train_label=np.array(train_label).reshape(-1, 1)            # train_img와 shape을 맞춰주기위해 열벡터로 변환한다

    return np.array(train_img), train_label


def next_batch(batch_size, x, y):                                   #x, y에 대해 batch_size만큼 batch해서 반환한다
    global idx_in_epoch                                             #현재 가리키고 있는 idx위치
    global temp                                                     #epoch한번 돌때마다 0~data총개수 까지의 idx번호를 shuffle해서 갖고있는 리스트
    start = idx_in_epoch
    idx_in_epoch += batch_size
    if idx_in_epoch > x.shape[0] or start==0:
        temp = np.arange(0, x.shape[0])
        np.random.shuffle(temp)
        start = 0
        idx_in_epoch = batch_size

    end = idx_in_epoch
    xp=temp[start:end]
    yp=temp[start:end]
    return x[xp], y[yp]                                             #numpy에서는 x[st:end] 이런식으로 슬라이싱으로 접근할수도 있다

class Model:
    def __init__(self, sess, name):
        self.sess=sess
        self.name=name
        self.build_net()

    def build_net(self):
        with tf.variable_scope(self.name) :
            self.X = tf.placeholder(tf.float32, [None, 128, 128, 3])
            self.X_img=tf.reshape(self.X, [-1, 128, 128, 3])

            self.Y=tf.placeholder(tf.float32, [None, 2])
            self.training=tf.placeholder(tf.bool)

            #convolution 1
            conv1=tf.layers.conv2d(inputs=self.X_img, filters=16, kernel_size=[8, 8], padding='SAME',
                                   activation=tf.nn.relu, strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            #pooling 1
            pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[12, 12], padding='SAME', strides=2)

            #convolution 2
            conv2=tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[5, 5], padding='SAME',
                                   activation=tf.nn.relu, strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            #pooling 2
            pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[12, 12], padding='SAME', strides=2)

            #convolution 3
            conv3=tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[3, 3], padding='SAME',
                                   activation=tf.nn.relu, strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())
            #pooling 3
            pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[5, 5], padding='SAME', strides=2)

            #convolution 4
            self.conv4=tf.layers.conv2d(inputs=pool3, filters=64, kernel_size=[3, 3], padding='SAME',
                                   activation=tf.nn.relu, strides=(1, 1), kernel_initializer=tf.contrib.layers.xavier_initializer())

############################################   CONVOLUTION LAYER   ###########################################################################
            self.gap = tf.reduce_mean(self.conv4, axis=[1, 2])

            self.logit = tf.layers.dense(inputs=self.gap, units=2, kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.Y, logits=self.logit))
        self.optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)

        self.prediction=tf.argmax(self.logit, axis=1)
        self.correct=tf.argmax(self.Y, axis=1)

        self.accuracy=tf.reduce_mean(tf.cast(tf.equal(self.prediction, self.correct), tf.float32))

    def visualize_class_activation_map(self):
        training=False
        input_path = os.getcwd() + '/aligned_img/test_Gang'
        count=0
        for img_path in glob.glob(os.path.join(input_path, '*.jpg')):
            img = cv2.imread(img_path, 1)

            i = img[np.newaxis]
            weights = tf.get_default_graph().get_tensor_by_name(os.path.split(self.logit.name)[0] + '/kernel:0')
            c = self.sess.run(self.conv4, feed_dict={self.X: i, self.training : training})
            w = weights.eval(session=self.sess)
            conv_outputs = tf.squeeze(c).eval(session=self.sess)  # (32, 32, 64)
            # Create the class activation map.
            cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[:2])
            for j in range(len(w)):
                cam += w[j][0] * conv_outputs[:, :, j]
            cam = np.maximum(cam, 0)
            cam /= np.max(cam)
            cam = cv2.resize(cam, (128, 128))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap[np.where(cam< 0.1)] = 0
            new_img = heatmap * 0.8 + img
            cv2.imwrite(os.getcwd() + '/cam_img/cam_img{}.jpg'.format(count), new_img)
            count+=1

    def train(self, x_data, y_data, training=False):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.X : x_data, self.Y : y_data, self.training : training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.X : x_test, self.Y : y_test, self.training : training})

    def predict(self, x_test, training=False):
        return self.sess.run(self.logit, feed_dict={self.X : x_test, self.training : training})


x_train, y_train=read_image()
x_test, y_test=read_test_image()
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 2),axis=1)                      #y_train, y_test값은 onehot 인코딩 안돼있는 상태이므로 one_hot시켜준다
sess=tf.Session()

model=Model(sess, 'model')
sess.run(tf.global_variables_initializer())
print('학습시작!!')
batch_size=50

for epoch in range(10) :
    avg_cost=0
    total_batch=int(len(x_train))//batch_size
    for i in range(total_batch) :
        batch_x, batch_y=next_batch(batch_size, x_train, y_train_one_hot.eval(session=sess))        #eval()로 onehot 인코딩 실행한 후의 값들을 넘겨준다
        c, _=model.train(batch_x, batch_y, training=False)
        avg_cost+=c/total_batch

    print('EPOCH : ', epoch+1, 'cost : ', avg_cost)

print('학습 끝')

model.visualize_class_activation_map()
print('Accuracy:', model.get_accuracy(x_test, y_test_one_hot.eval(session=sess)))
p=model.predict(x_test)
result=np.argmax(p, axis=1)
for re in result :
    if re==0 :
        print('Good!! Answer : Ganghee    Predict : Ganghee')
    else :
        print('Wrong! Answer : Ganghee    Predict : Parent')

