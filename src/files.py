import glob
import numpy as np
import os
from tensorflow import keras

class Files:

    def __init__(self, root_folder: str = ""):
        """
        Initializes the Files class with an optional root_folder parameter.

        Parameters:
        - root_folder (str): Root folder path.

        Returns:
        None
        """
        self.root_folder = root_folder

    def files_audio(self):
        """
        Retrieves a list of audio file paths (wav) from the specified root_folder.

        Returns:
        list: List of audio file paths.
        """
        list_audio = []
        subfolders = glob.glob(os.path.join(self.root_folder, "**/"), recursive=True)
        for subfolder in subfolders:
            wav_files = glob.glob(os.path.join(subfolder, "*.wav"))
            for file_path in wav_files:
                list_audio.append(file_path)
                print(file_path)
        return list_audio

    def predict_with_model(self, input_vectors):
        """
        Loads a saved Keras model and predicts binary labels for input vectors.

        Parameters:
        - input_vectors (numpy.ndarray): Input vectors for prediction.

        Returns:
        numpy.ndarray: Binary predictions.
        """
        print(self.root_folder)
        model = keras.models.load_model(self.root_folder)
        prediction = model.predict(input_vectors)
        binary_predictions = np.round(prediction).flatten()
        return binary_predictions