#import libraries


from fileinput import filename
from flask import Flask, render_template, request, redirect, send_file, url_for, flash, jsonify
from json import load
from pyexpat import model
from cv2 import VideoWriter

import numpy as np
from flask import Flask, render_template,request, redirect, make_response, jsonify, url_for, abort, send_from_directory,url_for, abort
import pickle #Initialize the flask App
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
import cv2 # pip install opencv-python
import numpy as np # pip install numpy
from playsound import playsound
import h5py



app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = "C:\\Users\\DEEPAK\\Desktop\\intership\\flask\\static\\uploads"
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
OUTPUT_FOLDER = "C:\\Users\\DEEPAK\\Desktop\\intership\\flask\\static\\output-videos"
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER




#default page of our web-app
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/contact.html')
def contact():
	return render_template('contact.html')

@app.route('/about.html')
def about():
	return render_template('about.html')
@app.route('/service.html', methods=['GET'])
def x():
	return render_template('service.html')    

@app.route('/service.html', methods=['POST'])
def upload_video():
	if request.method == 'POST':
		file = request.files['file']
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return redirect(url_for('prediction', filename=filename))
	return render_template('index.html')


outputnames = {"video_1.mp4","video_2.mp4","video_3.mp4","video_4.mp4","video_5.mp4","video_6.mp4","video_7.mp4","video_8.mp4","video_9.mp4","video_10.mp4",
"video_11.mp4","video_12.mp4","video_13.mp4","video_14.mp4","video_15.mp4","video_16.mp4","video_17.mp4","video_18.mp4","video_19.mp4","video_20.mp4",
"video_21.mp4","video_22.mp4","video_23.mp4","video_24.mp4","video_25.mp4","video_26.mp4","video_27.mp4","video_28.mp4","video_29.mp4","video_30.mp4",
"video_31.mp4","video_32.mp4","video_33.mp4","video_34.mp4","video_35.mp4","video_36.mp4","video_37.mp4","video_38.mp4","video_39.mp4","video_40.mp4",
"video_41.mp4","video_42.mp4","video_43.mp4","video_44.mp4","video_45.mp4","video_46.mp4","video_47.mp4","video_48.mp4","video_49.mp4","video_50.mp4"
}
@app.route('/show/<filename>')
def prediction(filename): 
	
	if filename in outputnames:
		filename =   "AnyConv.com__" + filename 
		print("video found")
		print(filename)
		#return send_from_directory(directory= OUTPUT_FOLDER, path = filename)
		return render_template('prediction.html', filename=filename)
	else:
		# Importing essential libraries
		import h5py
		import numpy as np
		import tensorflow.python.keras as Keras
		from sklearn.model_selection import train_test_split

		# Mount your drive


		# Defining a generator object to handle dataset
		import h5py
		import numpy as np
		import tensorflow.python.keras as Keras
		from sklearn.model_selection import train_test_split

		class DataGenerator(Keras.utils.Sequence):

			def __init__(self,dataset,batch_size=5,shuffle=False):
				'''Initialise the dataset'''      
				self.dataset = dataset
				self.batch_size = batch_size
				self.shuffle = shuffle
				self.on_epoch_end()

			def __len__(self):
				'''Returns length of the dataset'''
				return int(np.floor(len(self.dataset)/self.batch_size))

			def __getitem__(self,index):
				'''Returns items of given index'''
				indexes = self.indices[index * self.batch_size : (index+1) * self.batch_size]
				feature, label = self.__data_generation(indexes)

				return feature, label

			def __data_generation(self,indexes):
				'''Generates data from given indices'''
				feature = np.empty((self.batch_size,320,1024))
				label = np.empty((self.batch_size,320,1))

				for i in range(len(indexes)):
					feature[i,] = np.array(self.dataset[indexes[i]][0])
					label[i,] = np.array(self.dataset[indexes[i]][1]).reshape(-1,1)

				return feature,label

			def on_epoch_end(self):

				self.indices = np.arange(len(self.dataset))
				if self.shuffle == True:
					np.random.shuffle(self.indices)

		# Defining a class to read the h5 dataset
		class DatasetMaker(object):

			def __init__(self,data_path):
				'''Read file from defined path'''

				self.data_file = h5py.File(data_path)

			def __len__(self):
				'''Returns length of the file'''

				return len(self.data_file)

			def __getitem__(self,index):
				'''Returns feature, label and index of varoius keys'''

				index += 1
				video = self.data_file['video_'+str(index)]
				feature = np.array(video['feature'][:])
				label = np.array(video['label'][:])

				return feature,label,index

		# Defining a function to read and split the dataset into train and test
		def get_loader(path, batch_size=5):
			'''Takes file path as argument and returns train and test set'''

			dataset = DatasetMaker(path)
			train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2, random_state = 123)
			train_loader = DataGenerator(train_dataset)

			return train_dataset, train_loader, test_dataset

		# Loading and splitting the dataset
		train_dataset, train_loader, test_dataset = get_loader('fcsn_tvsum.h5')

		# Getting an overview of train set
		train_loader.__getitem__(1)

		# Getting an overview of test set
		test_dataset[1]

		import numpy as np

		def knapsack(s, w, c): #shot, weights, capacity


			shot = len(s) + 1 #number of shots
			cap = c + 1 #capacity threshold

			#matching the modified size by adding 0 at 0th index
			s = np.r_[[0], s] #adding 0 at 0th index (concatinating)
			w = np.r_[[0], w] #adding 0 at 0th index (concatinating)

			#Creating and Filling Dynamic Programming Table with zeros in shot x cap dimensions
			dp = [] #creating empty list or table
			for j in range(shot):
				dp.append([]) #s+1 rows
				for i in range(cap):
					dp[j].append(0) #c+1 columns

			#Started filling values from (2nd row, 2nd column) till (shot X cap) and keeping the values for 0th indexes as 0
			#following dynamic programming approach to fill values
			for i in range(1,shot):
				for j in range(1,cap):
					if w[i] <= j:
						dp[i][j] = max(s[i] + dp[i-1][j-w[i]], dp[i-1][j])
					else:
						dp[i][j] = dp[i-1][j]

			#choosing the optimal pair of keyshots
			choice = []
			i = shot - 1
			j = cap - 1
			while i > 0 and j > 0:
				if dp[i][j] != dp[i-1][j]: #starting from last element and going further
					choice.append(i-1)
					j = j - w[i]
					i = i - 1
				else:
					i = i - 1

			return dp[shot-1][cap-1], choice

		import numpy as np

		def eval_metrics(y_pred, y_true):
			'''Returns precision, recall and f1-score of given prediction and true value'''
			overlap = np.sum(y_pred * y_true)
			precision = overlap / (np.sum(y_pred) + 1e-8)
			recall = overlap / (np.sum(y_true) + 1e-8)
			if precision == 0 and recall == 0:
				fscore = 0
			else:
				fscore = 2 * precision * recall / (precision + recall)

			return [precision, recall, fscore]

		def select_keyshots(video_info, pred_score):
			'''Returns predicted scores(upsampled), selected keyshots indices, predicted summary of given video'''
			vidlen = video_info['length'][()] # Getting video length
			cps = video_info['change_points'][:] # Getting change points (shot changes)
			weight = video_info['n_frame_per_seg'][:] # Getting number of frames per shot info
			pred_score = np.array(pred_score)
			pred_score = upsample(pred_score, vidlen)
			pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
			_, selected = knapsack(pred_value, weight, int(0.2 * vidlen)) # Knapsacking 20% of the video as summary
			selected = selected[::-1]
			key_labels = np.zeros((vidlen,))
			for i in selected:
				key_labels[cps[i][0]:cps[i][1]] = 1

			return pred_score.tolist(), selected, key_labels.tolist()

		def upsample(down_arr, vidlen):
			'''Upsamples a given predicted score array to the size of video length'''
			up_arr = np.zeros(vidlen)
			ratio = vidlen // 320
			l = (vidlen - ratio * 320) // 2
			i = 0
			while i < 320:
				up_arr[l:l+ratio] = np.ones(ratio, dtype = int) * down_arr[0][i]
				l += ratio
				i += 1

			return up_arr

		import numpy as np
		import json
		import os
		from tqdm import tqdm, trange
		import h5py
		from prettytable import PrettyTable

		import tensorflow as tf
		from tensorflow.python.keras.models import Sequential
		from tensorflow.python.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
		from tensorflow.python.keras.layers import Attention
		from tensorflow.python.keras import Input, Model



		import numpy as np
		import json
		import os
		from tqdm import tqdm, trange
		import h5py
		from prettytable import PrettyTable

		import tensorflow as tf
		from tensorflow.python.keras.models import Sequential
		from tensorflow.python.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
		from tensorflow.python.keras.layers import Attention
		from tensorflow.python.keras import Input, Model



		# Baseline implementation of algorithm mentioned in the paper (with only one extra BiLSTM layer) with some hyperparameter tuning

		import numpy as np
		import json
		import os
		from tqdm import tqdm, trange
		import h5py
		from prettytable import PrettyTable

		import tensorflow as tf
		from tensorflow.python.keras.models import Sequential
		from tensorflow.python.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
		from tensorflow.python.keras.layers import Attention
		from tensorflow.python.keras import Input, Model

		encoder_inputs = Input(shape = (320, 1024))

		encoder_BidirectionalLSTM = Bidirectional(LSTM(128, return_sequences = True, return_state = True)) # Encoder Layer
		encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM(encoder_inputs)
		sh = Concatenate()([fh, bh])
		ch = Concatenate()([fc, bc])
		encoder_states = [sh, ch]

		decoder_LSTM = LSTM(256, return_sequences = True, dropout = 0.01, recurrent_dropout = 0.01) # Decoder Layer
		decoder_out = decoder_LSTM(encoder_out, initial_state = encoder_states)

		attn_layer = Attention(name="Attention_Layer") # Attention Layer
		attn_out =  attn_layer([encoder_out, decoder_out])

		decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_out, attn_out]) # Concatenating outputs of attention and decoder layer

		dense = TimeDistributed(Dense(1, activation = 'sigmoid')) # Output layer
		decoder_pred = dense(decoder_concat_input)

		model = Model(inputs = encoder_inputs, outputs = decoder_pred) # Model assembling

		opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1 = 0.8, beta_2 = 0.8) # Adam Optimizer
		#opt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

		model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy']) # Model Compiling
		model.summary()
		t = trange(10, desc = 'Epoch', ncols = 90)


		# Improved with two extra BiLSTM layers, one extra LSTM layer and two more Time Distributed layers and some hyperparameter tuning

		import numpy as np
		import json
		import os
		from tqdm import tqdm, trange
		import h5py
		from prettytable import PrettyTable

		import tensorflow as tf
		from tensorflow.python.keras.models import Sequential
		from tensorflow.python.keras.layers import LSTM, Bidirectional, Dense, TimeDistributed, Concatenate
		from tensorflow.python.keras.layers import Attention
		from tensorflow.python.keras import Input, Model
		from keras.layers import LeakyReLU

		encoder_inputs = Input(shape = (320, 1024))

		# Encoder Layers
		encoder_BidirectionalLSTM = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
		encoder_BidirectionalLSTM1 = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
		encoder_BidirectionalLSTM2 = Bidirectional(LSTM(64, return_sequences = True, return_state = True))
		encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM(encoder_inputs)
		encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM1(encoder_out)
		encoder_out, fh, fc, bh, bc = encoder_BidirectionalLSTM2(encoder_out)
		sh = Concatenate()([fh, bh])
		ch = Concatenate()([fc, bc])
		encoder_states = [sh, ch]

		# Decoder Layers
		decoder_LSTM = LSTM(128, return_sequences = True)
		decoder_out = decoder_LSTM(encoder_out, initial_state = encoder_states)
		decoder_out = decoder_LSTM(decoder_out)

		# Attention Layer
		attn_layer = Attention(name="Attention_Layer")
		attn_out =  attn_layer([encoder_out, decoder_out])

		# Concatenating decoder and attention outputs
		decoder_concat_input = Concatenate(axis = -1, name = 'concat_layer')([decoder_out, attn_out])

		# Time Distributed Dense Layer (for multi-dimensional) with relu activation function
		dense = TimeDistributed(Dense(42, activation = 'relu'))
		decoder_pred = dense(decoder_concat_input)

		# Time Distributed Dense Layer (for multi-dimensional) with tanh activation function
		dense = TimeDistributed(Dense(14, activation = 'tanh'))
		decoder_pred = dense(decoder_pred)

		# Time Distributed Dense Layer (for multi-dimensional) with sigmoid activation function (Output Layer)
		dense = TimeDistributed(Dense(1, activation = 'sigmoid'))
		decoder_pred = dense(decoder_pred)

		# Model assembling
		model = Model(inputs = encoder_inputs, outputs = decoder_pred)

		# Optimizers
		opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.8, beta_2 = 0.8)
		#opt = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

		# Compiling the model
		model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
		model.summary()
		t = trange(10, desc = 'Epoch', ncols = 90)


		# Reading the h5 file
		data_file = h5py.File('fcsn_tvsum.h5')

		# Predicting scores for a particular video using the model
		pred_score = model.predict(np.array(data_file['video_4']['feature']).reshape(-1,320,1024))
		video_info = data_file['video_4']
		pred_score, pred_selected, pred_summary = select_keyshots(video_info, pred_score)

		# Selected shots
		pred_selected

		# Getting the output summary video
		import cv2
		cps = video_info['change_points'][()]

		video = cv2.VideoCapture(filename)
		frames = []
		success, frame = video.read()
		while success:
			frames.append(frame)
			success, frame = video.read()
		frames = np.array(frames)
		keyshots = []
		for sel in pred_selected:
			for i in range(cps[sel][0], cps[sel][1]):
				keyshots.append(frames[i])
		keyshots = np.array(keyshots)

		video_writer = cv2.VideoWriter('static/uploads/AnyConv.com__'+filename, cv2.VideoWriter_fourcc(*'MP4V'), 24, keyshots.shape[2:0:-1])
		for frame in keyshots:
			video_writer.write(frame)
		video_writer.release()
	return render_template('prediction.html', filename = filename)



@app.route('/prediction/<filename>', methods=['GET', 'POST'])
def upload_video_prediction(filename):
	return send_from_directory( UPLOAD_FOLDER, filename)

@app.route('/download/<filename>' , methods=['GET', 'POST'])
def download_video(filename):
	filename =  filename
	print(filename)
	p = "C:\\Users\\DEEPAK\\Desktop\\intership\\flask\\static\\uploads\\"+ filename
	return send_file(p, as_attachment=True)
		
		
	#print(filename)	
	#return(url_for('static', filename='output-videos/' + filename), code=301)
	#return send_from_directory(app.config['OUTPUT_FOLDER'], filename= filename)

	
if __name__ == "__main__":
	app.run(debug=True)