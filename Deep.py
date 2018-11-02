import sys
print(sys.version)
import tensorflow as tf
import numpy as np
import cv2
mnist = tf.keras.datasets.mnist


(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
cv2.imshow("%s" %y_test[0], x_test[0])
cv2.waitKey(0)
if 1==2:
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #model.predict(x_test)
    model.fit(x_train, y_train, epochs=5)
    #print(sum(model.predict(np.ndarray.flatten(x_test[i]))==y_test[i] for i in range(len(x_test)))/len(x_test))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print("accuracy: " + str(accuracy))
    #model.predict(x_test, y_test)
