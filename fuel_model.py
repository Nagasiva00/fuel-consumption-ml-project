import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# main window
main = tk.Tk()
main.title("Average Fuel Consumption Prediction")
main.geometry("700x500")

filename = ""
model = None
testdata = None
predictdata = None


text = tk.Text(main,height=15,width=80)
text.pack()


# Upload Dataset
def upload():
    global filename
    filename = filedialog.askopenfilename()
    text.delete('1.0',tk.END)
    text.insert(tk.END,filename+" loaded\n")


# Train Model
def runANN():
    global model,testdata,predictdata

    data = pd.read_csv(filename)

    X = data.values[:,0:7]
    y = data.values[:,7]

    y = y.reshape(-1,1)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    train_x,test_x,train_y,test_y = train_test_split(X,y,test_size=0.2)

    model = Sequential()

    model.add(Dense(200,input_shape=(7,),activation='relu'))
    model.add(Dense(200,activation='relu'))
    # model.add(Dense(19,activation='softmax'))
    model.add(Dense(y.shape[1],activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x,train_y,epochs=50,batch_size=5)

    results = model.evaluate(test_x,test_y)

    text.insert(tk.END,"\nModel Accuracy : "+str(results[1]*100)+"\n")

    testdata = test_x


# Predict
def predictFuel():
    global model,testdata,predictdata

    predictdata = model.predict(testdata)

    text.insert(tk.END,"\nPrediction Completed\n")


# Graph
def graph():

    x=[]
    y=[]

    for i in range(len(predictdata)):
        x.append(i)
        y.append(np.argmax(predictdata[i]))

    plt.plot(x,y)
    plt.xlabel("Vehicle ID")
    plt.ylabel("Fuel Consumption")
    plt.title("Average Fuel Consumption Graph")
    plt.show()


# Buttons
uploadBtn = tk.Button(main,text="Upload Dataset",command=upload)
uploadBtn.pack()

annBtn = tk.Button(main,text="Run ANN Algorithm",command=runANN)
annBtn.pack()

predictBtn = tk.Button(main,text="Predict Fuel Consumption",command=predictFuel)
predictBtn.pack()

graphBtn = tk.Button(main,text="Fuel Consumption Graph",command=graph)
graphBtn.pack()


main.mainloop()