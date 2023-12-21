# -------------------Imports--------------------------#
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.activations import sigmoid, relu
from keras.layers import Dense
from files import Files
from utils import Functions
import keras

class Train:
    def __init__(self, stuttering_folder: str = "dataset/train/stutter", no_stuttering_folder: str = "dataset/train/noStutter"):
        """
        Initializes the Train class.
        
        Args:
            stuttering_folder (str): Path to the folder containing stuttering audio files.
            no_stuttering_folder (str): Path to the folder containing non-stuttering audio files.
        
        Outputs:
            None
        
        Important notes:
            This method initializes the Train class with the specified folders and triggers the initial training of the model.
        """
        self.stuttering_folder = stuttering_folder
        self.no_stuttering_folder = no_stuttering_folder
        self.initial_training_model()
    
    def initial_training_model(self):
        """
        Performs the initial training of the model using audio files from specified folders.
        
        Args:
            None
        
        Outputs:
            None
        
        Important notes:
            This method prints the list of audio files processed and triggers the binary classifier training.
        """
        audio_no_stutter = Files(self.no_stuttering_folder).files_audio()
        audio_stutter = Files(self.stuttering_folder).files_audio()
        list_concatenated_vector_get = []
        labels = []
        # Process non-stuttering audio files
        for i in range(0*int(len(audio_no_stutter)/7), 1*int(len(audio_no_stutter)/7)):
            concatenated_vector_get = Functions(audio_no_stutter[i]).get_concatenated_vector_get()
            list_concatenated_vector_get.append(concatenated_vector_get)
            labels.append(0)
        # Process stuttering audio files
        for i in range(0*int(len(audio_stutter)/7), 1*int(len(audio_stutter)/7)):
            concatenated_vector_get = Functions(audio_stutter[i]).get_concatenated_vector_get()
            list_concatenated_vector_get.append(concatenated_vector_get)
            labels.append(1)
        # Trigger binary classifier training
        self.binary_classifier_training(list_concatenated_vector_get, labels)

    def binary_classifier_training(self, ambidig_vectors, labels):
        """
        Trains a binary classifier model using the specified vectors and labels.
        
        Args:
            ambidig_vectors (list): List of concatenated vectors for training.
            labels (list): List of corresponding labels (0 or 1).
        
        Outputs:
            None
        
        Important notes:
            This method builds, compiles, and trains a binary classifier model. The trained model is saved as 'classification_model.h5'.
        """
        concatenated_vectors = np.concatenate(ambidig_vectors, axis=0)
        concatenated_labels = np.repeat(labels, [v.shape[0] for v in ambidig_vectors], axis=0)
        
        # Build a neural network model
        model = Sequential()
        model.add(Dense(64, activation=relu, input_dim=769))
        model.add(Dense(64, activation=relu))
        model.add(Dense(64, activation=relu))
        model.add(Dense(1, activation=sigmoid))

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train the model
        model.fit(concatenated_vectors, concatenated_labels, epochs=10, batch_size=32)
        
        # Save the model
        model.save('classification_model.h5')
        print('Model weights saved successfully.')
