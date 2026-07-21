import os
import random
import gc, numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import compute_class_weight
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MultiHeadAttention, concatenate, Conv1D, GlobalAveragePooling1D, GlobalAveragePooling2D, LayerNormalization, Reshape, Add, Activation, MaxPooling2D
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

def make_img(t_img):
    img = pd.read_pickle(t_img)
    img_l = []
    for i in range(len(img)):
        img_l.append(img.values[i][0])
    
    return np.array(img_l)


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
   
               
def resnet_block(x, filters, stride=1):
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x

def create_model_img():
    img_input = Input(shape=(72, 72, 3))
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)
    
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 128, stride=2)
    x = resnet_block(x, 128)
    x = resnet_block(x, 256, stride=2)
    x = resnet_block(x, 256)
    x = resnet_block(x, 512, stride=2)
    x = resnet_block(x, 512)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    
    return Model(img_input, x)

def create_model_snp():
    snp_input = Input(shape=(15965,))
    x = Reshape((15965, 1))(snp_input)
    x = Conv1D(filters=64, kernel_size=32, strides=32, activation='relu')(x)
    
    for _ in range(4):
        attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = Add()([x, attn_output])
        x = LayerNormalization()(x)
        
        ffn_output = Dense(2048, activation='relu')(x)
        ffn_output = Dense(512)(ffn_output)
        ffn_output = Dropout(0.1)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization()(x)
    
    x = GlobalAveragePooling1D()(x)
    x = Dense(512, activation='relu')(x)
    
    return Model(snp_input, x)


def plot_classification_report(y_tru, y_prd, mode, learning_rate, batch_size,epochs, figsize=(7, 7), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = ["Control", "Moderate", "Alzheimer's" ] 
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    sns.heatmap(rep,
                annot=True, 
                cbar=False, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax, cmap = "Blues")
    
    plt.savefig('report_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'_' + str(epochs)+'.png')
    


def calc_confusion_matrix(result, test_label,mode, learning_rate, batch_size, epochs):
    test_label = to_categorical(test_label,3)

    true_label = test_label

    predicted_label= np.argmax(result, axis =1)
    
    n_classes = 3
    precision = dict()
    recall = dict()
    thres = dict()
    for i in range(n_classes):
        precision[i], recall[i], thres[i] = precision_recall_curve(test_label[:, i],
                                                            result[:, i])


    print ("Classification Report :") 
    print (classification_report(true_label, predicted_label))
    cr = classification_report(true_label, predicted_label, output_dict=True)
    return cr, precision, recall, thres



def cross_modal_attention(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 8,key_dim=64)(x, y)
    a2 = MultiHeadAttention(num_heads = 8,key_dim=64)(y, x)
    a1 = a1[:,0,:]
    a2 = a2[:,0,:]
    return concatenate([a1, a2])

def cross_modal_attention_split(x, y):
    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    a1 = MultiHeadAttention(num_heads = 8,key_dim=64)(x, y)
    a2 = MultiHeadAttention(num_heads = 8,key_dim=64)(y, x)
    return a1[:,0,:], a2[:,0,:]


def self_attention(x):
    x = tf.expand_dims(x, axis=1)
    attention = MultiHeadAttention(num_heads = 8, key_dim=64)(x, x)
    attention = attention[:,0,:]
    return attention
    

def multi_modal_model(mode, train_snp, train_img):
    
    in_snp = Input(shape=(train_snp.shape[1]))
    
    in_img = Input(shape=(train_img.shape[1], train_img.shape[2], train_img.shape[3]))
    
    dense_snp = create_model_snp()(in_snp) 
    dense_img = create_model_img()(in_img) 
    
 
        
    ########### Attention Layer ############
        
    ## Cross Modal Bi-directional Attention ##

    if mode == 'MM_BA':
            
        av_att = cross_modal_attention(dense_snp, dense_img)
                
        merged = concatenate([av_att, dense_img, dense_snp])
                 
   
        
        
    ## Self Attention ##
    elif mode == 'MM_SA':
            
        vv_att = self_attention(dense_img)
        aa_att = self_attention(dense_snp)
            
        merged = concatenate([aa_att, vv_att, dense_img, dense_snp])
        
    ## Self Attention and Cross Modal Bi-directional Attention##
    elif mode == 'MM_SA_BA':
            
        cross_img, cross_snp = cross_modal_attention_split(dense_img, dense_snp)
        
        vv_att = self_attention(cross_img)
        aa_att = self_attention(cross_snp)
            
        merged = concatenate([vv_att, aa_att, dense_img, dense_snp])
            
        
    ## No Attention ##    
    elif mode == 'None':
            
        merged = concatenate([dense_img, dense_snp])
                
    else:
        print ("Mode must be one of 'MM_SA', 'MM_BA', 'MM_SA_BA' or 'None'.")
        return
                
        
    ########### Output Layer ############
        
    merged = Dense(512, activation='relu')(merged)
    merged = Dropout(0.1)(merged)
    merged = Dense(256, activation='relu')(merged)
    output = Dense(2, activation='softmax', name='main_output')(merged)
    
    dense_img_aux = Dense(512, activation='relu')(dense_img)
    dense_img_aux = Dense(256, activation='relu')(dense_img_aux)
    output_img = Dense(2, activation='softmax', name='mri_output')(dense_img_aux)
    
    dense_snp_aux = Dense(512, activation='relu')(dense_snp)
    dense_snp_aux = Dense(256, activation='relu')(dense_snp_aux)
    output_snp = Dense(2, activation='softmax', name='snp_output')(dense_snp_aux)
    
    model = Model([in_snp, in_img], [output, output_img, output_snp])        
        
    return model


class CurriculumDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, train_snp, train_img, train_label, sample_weights, batch_size, model, total_epochs):
        self.train_snp = train_snp
        self.train_img = train_img
        self.train_label = train_label
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.model = model
        self.total_epochs = total_epochs
        
        self.indices = np.arange(len(train_label))
        self.current_epoch = 0
        self.update_curriculum()
        
    def update_curriculum(self):
        pace = min(1.0, 0.3 + 0.7 * (self.current_epoch / max(1, self.total_epochs * 0.5)))
        subset_size = int(len(self.train_label) * pace)
        
        if self.current_epoch == 0:
            np.random.shuffle(self.indices)
            self.active_indices = self.indices[:subset_size]
            print(f"\n--- Curriculum Learning: Epoch 1, Randomly selected {subset_size} samples ({pace*100:.1f}%) ---")
        else:
            print(f"\n--- Curriculum Learning: Evaluating losses for Epoch {self.current_epoch + 1}... ---")
            preds = self.model.predict([self.train_snp, self.train_img], batch_size=32, verbose=0)
            main_preds = preds[0]
            
            scce = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            losses = scce(self.train_label, main_preds).numpy()
            
            sorted_indices = np.argsort(losses)
            self.active_indices = sorted_indices[:subset_size]
            np.random.shuffle(self.active_indices)
            print(f"--- Curriculum Learning: Sorted and selected {subset_size} easiest samples ({pace*100:.1f}%) ---")
            
    def __len__(self):
        return int(np.ceil(len(self.active_indices) / self.batch_size))
        
    def __getitem__(self, index):
        batch_idx = self.active_indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        b_snp = self.train_snp[batch_idx]
        b_img = self.train_img[batch_idx]
        b_label = self.train_label[batch_idx]
        b_weight = self.sample_weights[batch_idx]
        
        return ([b_snp, b_img], 
                {'main_output': b_label, 'mri_output': b_label, 'snp_output': b_label},
                {'main_output': b_weight, 'mri_output': b_weight, 'snp_output': b_weight})
                
    def on_epoch_end(self):
        self.current_epoch += 1
        if self.current_epoch < self.total_epochs:
            self.update_curriculum()

def train(mode, batch_size, epochs, learning_rate, seed):
    
 
    
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
    train_snp = pd.read_csv(os.path.join(data_dir, "X_train_snp.csv")).drop("Unnamed: 0", axis=1).values
    test_snp = pd.read_csv(os.path.join(data_dir, "X_test_snp.csv")).drop("Unnamed: 0", axis=1).values

    
    train_img= make_img(os.path.join(data_dir, "X_train_img.pkl"))
    test_img= make_img(os.path.join(data_dir, "X_test_img.pkl"))

    
    train_label= pd.read_csv(os.path.join(data_dir, "y_train.csv")).drop("Unnamed: 0", axis=1).values.astype("int").flatten()
    test_label= pd.read_csv(os.path.join(data_dir, "y_test.csv")).drop("Unnamed: 0", axis=1).values.astype("int").flatten()

    reset_random_seeds(seed)
    class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(train_label),y = train_label)
    d_class_weights = dict(enumerate(class_weights))
    sample_weights = np.array([d_class_weights[y] for y in train_label])
    
    # compile model #
    model = multi_modal_model(mode, train_snp, train_img)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss={'main_output': 'sparse_categorical_crossentropy', 
                        'mri_output': 'sparse_categorical_crossentropy', 
                        'snp_output': 'sparse_categorical_crossentropy'},
                  loss_weights={'main_output': 1.0, 'mri_output': 0.3, 'snp_output': 0.3},
                  metrics=['sparse_categorical_accuracy'])
    

    # Manual validation split since we are using a Generator
    val_split_idx = int(len(train_snp) * 0.9)
    
    val_snp = train_snp[val_split_idx:]
    val_img = train_img[val_split_idx:]
    val_label = train_label[val_split_idx:]
    
    train_snp = train_snp[:val_split_idx]
    train_img = train_img[:val_split_idx]
    train_label = train_label[:val_split_idx]
    sample_weights = sample_weights[:val_split_idx]
    
    # prepare curriculum generator
    curriculum_gen = CurriculumDataGenerator(train_snp, train_img, train_label, sample_weights, batch_size, model, epochs)

    # summarize results
    history = model.fit(curriculum_gen,
                        epochs=epochs,
                        validation_data=([val_snp, val_img], 
                                        {'main_output': val_label, 'mri_output': val_label, 'snp_output': val_label}),
                        verbose=1)
                        
                

    score = model.evaluate([test_snp, test_img], 
                           {'main_output': test_label, 'mri_output': test_label, 'snp_output': test_label})
    
    test_predictions = model.predict([test_snp, test_img])
    main_predictions = test_predictions[0]
    cr, precision_d, recall_d, thres = calc_confusion_matrix(main_predictions, test_label, mode, learning_rate, batch_size, epochs)
    
    
    """
    plt.clf()
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('accuracy_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig('loss_' + str(mode) + '_' + str(learning_rate) +'_' + str(batch_size)+'.png')
    plt.clf()
    """
    
 
    
    # release gpu memory #
    K.clear_session()
    del model, history
    gc.collect()
        
        
    print ('Mode: ', mode)
    print ('Batch size:  ', batch_size)
    print ('Learning rate: ', learning_rate)
    print ('Epochs:  ', epochs)
    print ('Test Accuracy:', '{0:.4f}'.format(acc))
    print ('-'*55)
    
    return acc, batch_size, learning_rate, epochs, seed
    
    
if __name__=="__main__":
    
    m_a = {}
    seeds = random.sample(range(1, 200), 5)
    for s in seeds:
        acc, bs_, lr_, e_ , seed= train('MM_SA_BA', 32, 50, 0.001, s)
        m_a[acc] = ('MM_SA_BA', acc, bs_, lr_, e_, seed)
    print(m_a)
    print ('-'*55)
    max_acc = max(m_a, key=float)
    print("Highest accuracy of: " + str(max_acc) + " with parameters: " + str(m_a[max_acc]))
    
