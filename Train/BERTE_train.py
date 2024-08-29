import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import Conv2D,MaxPooling2D,Input
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_bert import AdamWarmup, calc_train_steps
from keras import backend as K


parser = argparse.ArgumentParser(description='Train a CNN model to classify DNA sequences using BERT embeddings and k-mer counts.')
parser.add_argument('rank', help='Rank name for output file naming.')
parser.add_argument('epoch', type=int, help='Number of epochs for training the model.')
parser.add_argument('batchsize', type=int, help='Batch size for training the model.')
args = parser.parse_args()

rank = args.rank
epoch = args.epoch
batchsize = args.batchsize
rank_name = str(rank)
epoch=int(epoch)
batchsize = int(batchsize)

# Function to calculate shape of a nested list
def get_list_shape(lst):
    shape = []
    while isinstance(lst, list):
        shape.append(len(lst))
        lst = lst[0] if lst else None
    return shape

# Function to normalize frequencies in a list of lists
def normalize_frequencies(count_pkl):
    normalized_frequencies = []
    for sublist in count_pkl:
        total_frequency = sum(sublist)
        normalized_sublist = [frequency / total_frequency for frequency in sublist]
        normalized_frequencies.append(normalized_sublist)
    return normalized_frequencies


# Start time to track duration and name files
start_time = datetime.datetime.now()
print("start_time:",start_time)

input_store_report_dir = './report/'


'''---------------------------'''
'''1. Load labels and features'''
'''---------------------------'''


# Load labels associated with sequences
sequence_superfamily = 'demo_SINE_510bp_superfamliy.pkl'

# Load BERT embedding features for different k-mer lengths
cls_4mer = 'demo_SINE_506bp_bothend_kmer_fragments.jsonl_cls_embedding_features.pkl'
cls_5mer = 'demo_SINE_508bp_bothend_kmer_fragments.jsonl_cls_embedding_features.pkl'
cls_6mer = 'demo_SINE_510bp_bothend_kmer_fragments.jsonl_cls_embedding_features.pkl'

cls_4mer_pkl = pickle.load(open(cls_4mer,'rb'))
cls_5mer_pkl = pickle.load(open(cls_5mer,'rb'))
cls_6mer_pkl = pickle.load(open(cls_6mer,'rb'))

# Print shapes of loaded BERT embedding features
print("cls_4mer_pkl:",get_list_shape(cls_4mer_pkl))
print("cls_5mer_pkl:",get_list_shape(cls_5mer_pkl))
print("cls_6mer_pkl:",get_list_shape(cls_6mer_pkl))

all_superfamily = pickle.load(open(sequence_superfamily,'rb'))
print("Shape of sequence_superfamily_pkl:",get_list_shape(all_superfamily))

# Load full length k-mer count data
count_4mer = 'demo_SINE_506bp_full_length_kmer_counts_list.pkl'
count_5mer = 'demo_SINE_508bp_full_length_kmer_counts_list.pkl'
count_6mer = 'demo_SINE_510bp_full_length_kmer_counts_list.pkl'

count_4mer_pkl = pickle.load(open(count_4mer,'rb'))
count_5mer_pkl = pickle.load(open(count_5mer,'rb'))
count_6mer_pkl = pickle.load(open(count_6mer,'rb'))

print("count_4mer_pkl:",get_list_shape(count_4mer_pkl))
print("count_5mer_pkl:",get_list_shape(count_5mer_pkl))
print("count_6mer_pkl:",get_list_shape(count_6mer_pkl))

# Normalize k-mer frequencies
kmer_4_freq = normalize_frequencies(count_4mer_pkl)
kmer_5_freq = normalize_frequencies(count_5mer_pkl)
kmer_6_freq = normalize_frequencies(count_6mer_pkl)

all_4mer_feature_combine = []
all_5mer_feature_combine = []
all_6mer_feature_combine = []

# Combine features for training
all_4mer_feature_combine = []
all_5mer_feature_combine = []
all_6mer_feature_combine = []

for i in range(0,len(all_superfamily)):
    all_4mer_feature_combine.append(cls_4mer_pkl[i] + count_4mer_pkl[i])
    all_5mer_feature_combine.append(cls_5mer_pkl[i] + count_5mer_pkl[i]) 
    all_6mer_feature_combine.append(cls_6mer_pkl[i] + count_6mer_pkl[i]) 


'''------------------------------------'''
'''2. Convert labels, splitting dataset'''
'''------------------------------------'''


def conv_labels(labels,input_data_nm):
    converted = []
    for label in labels:
        Class2_list = ['CACTA','MULE','PIF','TcMar','hAT','Helitron']
        LINE_list = ['CR1','I','Jockey','L1','R2','RTE','Rex1']
        SINE_list = ['SINE1/7SL','SINE2/tRNA','SINE3/5S','ID']
        LTR_list = ['Bel-Pao','Copia','Gypsy','ERV1','ERV2','ERV3','ERV4']
        nonLTR_list = ['DIRS','PLE','CR1','I','Jockey','L1','R2','RTE','Rex1','SINE1/7SL','SINE2/tRNA','SINE3/5S','ID']
        TIR_list = ['CACTA','MULE','PIF','hAT','TcMar']

        if input_data_nm == 'all':
            if str(label) not in Class2_list:
                converted.append(0)
            elif str(label) in Class2_list:
                converted.append(1)
        
        if input_data_nm == 'ClassI':
            if label in LTR_list:
                converted.append(0)
            elif label in nonLTR_list:
                converted.append(1)
        if input_data_nm == 'ClassII':
            if label in TIR_list:
                converted.append(0)
            elif label == 'Helitron':
                converted.append(1)   

        if input_data_nm =='LTR':
            if label == 'Bel-Pao':
                converted.append(0)
            elif label == 'Copia':
                converted.append(1)
            elif label == 'Gypsy':
                converted.append(2)
            elif 'ERV' in label:
                converted.append(3)

        if input_data_nm =='nonLTR':
            if label == 'DIRS':
                converted.append(0)
            elif label == 'PLE':
                converted.append(1)
            elif label in LINE_list:
                converted.append(2)
            elif label in SINE_list:
                converted.append(3)

        if input_data_nm =='LINE':
            if label == 'CR1':
                converted.append(0)
            elif label == 'I':
                converted.append(1)
            elif label == 'Jockey':
                converted.append(2)
            elif label == 'L1':
                converted.append(3)
            elif label == 'R2':
                converted.append(4)
            elif label == 'RTE':
                converted.append(5)
            elif label == 'Rex1':
                converted.append(6)

        if input_data_nm =='SINE':
            if label == 'SINE1/7SL':
                converted.append(0)
            elif label == 'SINE2/tRNA':
                converted.append(1)
            elif label == 'SINE3/5S':
                converted.append(2) 
            elif label == 'ID':
                converted.append(3) 

        if input_data_nm == 'TIR':
            if label == 'CACTA':
                converted.append(0)
            elif label == 'MULE':
                converted.append(1)
            elif label == 'PIF':
                converted.append(2)
            elif label == 'TcMar':
                converted.append(3)
            elif label == 'hAT':
                converted.append(4)

    return converted

# Specify the rank to be trained
input_data_nm = rank_name

class_num = 0 

if input_data_nm == 'all':
    class_num = 2
if input_data_nm == 'ClassI':
    class_num = 2
if input_data_nm == 'ClassII':
    class_num = 2
if input_data_nm == 'LTR':
    class_num = 4
if input_data_nm == 'nonLTR':
    class_num = 4
if input_data_nm == 'LINE':
    class_num = 7
if input_data_nm == 'SINE':
    class_num = 4
if input_data_nm == 'TIR':
    class_num = 5

# Split data into training and test sets
y1 = conv_labels(all_superfamily,input_data_nm)
Y1 = np.asarray(y1)
X1 = np.asarray(all_4mer_feature_combine)
X2 = np.asarray(all_5mer_feature_combine)
X3 = np.asarray(all_6mer_feature_combine)

X_4mer_train, X_4mer_test, Y_train, Y_test,= train_test_split(
    X1, Y1, test_size=0.1, random_state=42,stratify=Y1)
X_5mer_train, X_5mer_test, Y_train, Y_test,= train_test_split(
    X2, Y1, test_size=0.1, random_state=42,stratify=Y1)
X_6mer_train, X_6mer_test, Y_train, Y_test,= train_test_split(
    X3, Y1, test_size=0.1, random_state=42,stratify=Y1)

print("Y_test:",np.unique(Y_test))
print("Y_train:",np.unique(Y_train))

##########################
# 2. Preprocess input data
print("X_4mer_train.shape[0]:",X_4mer_train.shape[0])
print("X_4mer_train.shape[1]:",X_4mer_train.shape[1])

X_4mer_train = X_4mer_train.astype('float64')
X_4mer_test = X_4mer_test.astype('float64')
X_5mer_train = X_5mer_train.astype('float64')
X_5mer_test = X_5mer_test.astype('float64')
X_6mer_train = X_6mer_train.astype('float64')
X_6mer_test = X_6mer_test.astype('float64')

X_4mer_train = X_4mer_train.reshape(X_4mer_train.shape[0], 1, 4**4+256, 1)  
X_5mer_train = X_5mer_train.reshape(X_5mer_train.shape[0], 1, 4**5+256, 1)  
X_6mer_train = X_6mer_train.reshape(X_6mer_train.shape[0], 1, 4**6+256, 1)  

X_4mer_test = X_4mer_test.reshape(X_4mer_test.shape[0], 1, 4**4+256, 1)     
X_5mer_test = X_5mer_test.reshape(X_5mer_test.shape[0], 1, 4**5+256, 1)     
X_6mer_test = X_6mer_test.reshape(X_6mer_test.shape[0], 1, 4**6+256, 1)   

# Convert labels to one-hot encoding
Y_train_one_hot = np_utils.to_categorical(Y_train, int(class_num))  
Y_test_one_hot = np_utils.to_categorical(Y_test, int(class_num)) 

input_4mer_shape=(1, 4**4+256, 1)
input_5mer_shape=(1, 4**5+256, 1)
input_6mer_shape=(1, 4**6+256, 1)


'''--------------------------------------------'''
'''3. Define the classifying model architecture
      and parameters'''
'''--------------------------------------------'''


input_a = Input(shape=input_4mer_shape)
model1 = Sequential()

model1.add(Conv2D(100, (1, 3),activation='relu', input_shape=input_4mer_shape))
model1.add(MaxPooling2D(pool_size=(1, 2)))
model1.add(Dropout(0.5))
model1.add(Flatten())

input_b = Input(shape=input_5mer_shape)
model2 = Sequential()
model2.add(Conv2D(100, (1, 5), activation='relu',input_shape=input_5mer_shape))       
model2.add(MaxPooling2D(pool_size=(1, 2)))
model2.add(Dropout(0.5))
model2.add(Flatten())

input_c = Input(shape=input_6mer_shape)
model3 = Sequential()
model3.add(Conv2D(100, (1, 7), activation='relu',input_shape=input_6mer_shape))       
model3.add(MaxPooling2D(pool_size=(1, 2)))
model3.add(Dropout(0.5))
model3.add(Flatten())

print("model1(input_shape):",K.int_shape(model1(input_a)))
print("model2(input_shape):",K.int_shape(model2(input_b)))
print("model3(input_shape):",K.int_shape(model3(input_c)))

concat = concatenate([model1(input_a), model2(input_b), model3(input_c)], axis=1, name="concat_layer")
print("model_concat(input_shape):",K.int_shape(concat))
output = Dense(int(class_num), activation='softmax')(concat)
model = Model(inputs=[input_a, input_b,input_c], outputs=[output])
model.summary()

# Setting callbacks for the training
monitor_mode = 'val_accuracy'
callback_lists = []
callback_lists.append(EarlyStopping(monitor=monitor_mode, patience=6, min_delta=0.001, verbose=1,restore_best_weights=True))
callback_lists.append(ModelCheckpoint(filepath=input_store_report_dir + '/' + str(start_time) + input_data_nm + '_best_val_loss_model.h5',save_best_only=True, monitor='val_loss', mode='min'))
callback_lists.append(ModelCheckpoint(filepath=input_store_report_dir + '/' + str(start_time) + input_data_nm + '_best_val_accuracy_model.h5',save_best_only=True, monitor='val_accuracy', mode='max'))


# Setting wamrups
total_steps, warmup_steps = calc_train_steps(
    num_example=X_4mer_train.shape[0],
    batch_size=batchsize,
    epochs=epoch,
    warmup_proportion=0.1,
)

optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-4, min_lr=1e-7)
print("warmup_steps",warmup_steps)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])


'''-----------------'''
'''4. Model training'''
'''-----------------'''


# Fit model on training data
model.fit([X_4mer_train,X_5mer_train,X_6mer_train], Y_train_one_hot, validation_split=0.1,
            batch_size=batchsize, epochs=epoch, verbose=1, shuffle=False,callbacks=callback_lists) 
                                             
# Evaluate model on test data
X_test_final = [X_4mer_test,X_5mer_test,X_6mer_test]   #并行cnn时多个输入
score = model.evaluate(X_test_final, Y_test_one_hot, verbose=1)
print ("\nscore = " + str(score))

# Save the model
model.save(input_store_report_dir + '/' + str(start_time) + input_data_nm + '_model.h5')

# Generate a classification report
##predict label comparing with true data
predicted_classes = model.predict(X_test_final)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
Y_test = np.asarray(Y_test)
target_names = ["Class {}".format(i) for i in range(int(class_num))]

# Save the report
with open (input_store_report_dir + '/' + str(start_time) + input_data_nm + '_class_report.txt','w+') as opt:
    opt.write(classification_report(Y_test, predicted_classes, target_names=target_names) + '\n')
    print(classification_report(Y_test, predicted_classes, target_names=target_names) + '\n')
print("model.predict(X_test):\n",predicted_classes)
pickle.dump(predicted_classes, open(str(input_data_nm) + str(start_time) +'test_predictions.pkl', 'wb'))
