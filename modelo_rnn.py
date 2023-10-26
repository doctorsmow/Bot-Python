import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class ModeloML:
    def __init__(self):
        self.model = Sequential()
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data = self.data.set_index("Date")  # Asumiendo que los datos tienen una columna "Date"
        self.data_values = self.data.values
        self.data_values = self.data_values.astype('float32')

    def train_model(self):
        self.data_scaled = self.scaler.fit_transform(self.data_values)

        train_size = int(len(self.data_scaled) * 0.7)
        train, test = self.data_scaled[0:train_size,:], self.data_scaled[train_size:len(self.data_scaled),:]

        # reshape into X=t and Y=t+1
        trainX, trainY = self.create_dataset(train, 1)
        testX, testY = self.create_dataset(test, 1)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # Ajustar la estructura del modelo según las necesidades del problema
        self.model.add(LSTM(100, input_shape=(trainX.shape[1], trainX.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

        self.model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

        # Hacer predicciones
        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)

        # Invertir las predicciones
        trainPredict = self.scaler.inverse_transform(trainPredict)
        trainY = self.scaler.inverse_transform([trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testY = self.scaler.inverse_transform([testY])

        # Calcular el error de la raíz cuadrada media
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    def predict(self, X):
        X = self.scaler.transform(X)
        X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        prediction = self.model.predict(X)
        return prediction
