
import tensorflow as tf
import numpy as np
import cv2
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

#show a sample
cv2.imshow("%s" %y_test[0], x_test[0])
cv2.waitKey(0)

model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(), #Take in an image, and vectorize it
      tf.keras.layers.Dense(512, activation=tf.nn.relu), #add a densely connected layer with exactly 512 neurons
      tf.keras.layers.Dropout(0.2), #Add a "dropout" layer, to combat overfitting
      tf.keras.layers.Dense(10, activation=tf.nn.softmax) #Create an output layer
])

model.compile(optimizer='adam', 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)

loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

print("accuracy: " + str(accuracy))
