import numpy as np
from keras import models
from files import Files
from utils import Functions

class Test:
    def __init__(self, stuttering_folder: str = "dataset/test/stutter",
                 no_stuttering_folder: str = "dataset/test/noStutter", classified: str = "classification_model.h5"):
        """
        Initializes the Test class.
        
        Args:
            stuttering_folder (str): Path to the folder containing stuttering audio files for testing.
            no_stuttering_folder (str): Path to the folder containing non-stuttering audio files for testing.
            classified (str): Path to the pre-trained classification model file.
        
        Outputs:
            None
        
        Important notes:
            This method initializes the Test class with the specified folders and the pre-trained classification model.
            It then triggers the testing of the model on the provided test data.
        """
        self.stuttering_folder = stuttering_folder
        self.no_stuttering_folder = no_stuttering_folder
        self.classified = classified
        self.test_model()

    def test_model(self):
        """
        Tests the pre-trained model on the specified test data and prints evaluation metrics.
        
        Args:
            None
        
        Outputs:
            None
        
        Important notes:
            This method loads the pre-trained model, processes test audio data, and calculates precision, recall, F1-score, and accuracy.
            The evaluation metrics are then printed.
        """
        model = models.load_model(self.classified)
        audio_no_stutter = Files(self.no_stuttering_folder).files_audio()
        audio_stutter = Files(self.stuttering_folder).files_audio()
        test_vectors = []
        test_labels = []
        test_vectors_to = []
        # Process non-stuttering audio files
        for i in range(len(audio_no_stutter)):
            concatenated_vector_get = Functions(audio_no_stutter[i]).get_concatenated_vector_get()
            prediction = model.predict(concatenated_vector_get)
            binary_predictions = np.round(prediction).flatten()
            test_vectors_to.append(binary_predictions[0])
            test_vectors.append(concatenated_vector_get)
            test_labels.append(0)
        # Process stuttering audio files
        for i in range(len(audio_stutter)):
            concatenated_vector_get = Functions(audio_stutter[i]).get_concatenated_vector_get()
            prediction = model.predict(concatenated_vector_get)
            binary_predictions = np.round(prediction).flatten()
            test_vectors.append(concatenated_vector_get)
            test_vectors_to.append(binary_predictions[0])
            test_labels.append(1)

        precision = self.calculate_precision(test_labels, test_vectors_to)
        recall = self.calculate_recall(test_labels, test_vectors_to)
        f1_score = self.calculate_f1_score(test_labels, test_vectors_to)
        accuracy = self.calculate_accuracy(test_labels, test_vectors_to)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-score:", f1_score)
        print("Accuracy:", accuracy)

    def calculate_precision(self, labels, predicted_labels):
        """
        Calculates precision based on true positives and predicted positives.
        
        Args:
            labels (list): True labels (0 or 1).
            predicted_labels (list): Predicted labels (0 or 1).
        
        Outputs:
            float: Precision score.
        
        Important notes:
            Precision is the ratio of true positives to the sum of true positives and false positives.
        """
        true_positives = 0
        predicted_positives = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == 1:
                predicted_positives += 1
                if labels[i] == 1:
                    true_positives += 1

        precision = true_positives / predicted_positives if predicted_positives != 0 else 0
        return precision

    def calculate_recall(self, labels, predicted_labels):
        """
        Calculates recall based on true positives and actual positives.
        
        Args:
            labels (list): True labels (0 or 1).
            predicted_labels (list): Predicted labels (0 or 1).
        
        Outputs:
            float: Recall score.
        
        Important notes:
            Recall is the ratio of true positives to the sum of true positives and false negatives.
        """
        true_positives = 0
        actual_positives = 0
        for i in range(len(predicted_labels)):
            if labels[i] == 1:
                actual_positives += 1
                if predicted_labels[i] == 1:
                    true_positives += 1
        recall = true_positives / actual_positives if actual_positives != 0 else 0
        return recall

    def calculate_f1_score(self, labels, predicted_labels):
        """
        Calculates F1-score based on precision and recall.
        
        Args:
            labels (list): True labels (0 or 1).
            predicted_labels (list): Predicted labels (0 or 1).
        
        Outputs:
            float: F1-score.
        
        Important notes:
            F1-score is the harmonic mean of precision and recall, providing a balance between the two metrics.
        """
        precision = self.calculate_precision(labels, predicted_labels)
        recall = self.calculate_recall(labels, predicted_labels)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score

    def calculate_accuracy(self, labels, predicted_labels):
        """
        Calculates accuracy based on correct predictions.
        
        Args:
            labels (list): True labels (0 or 1).
            predicted_labels (list): Predicted labels (0 or 1).
        
        Outputs:
            float: Accuracy score.
        
        Important notes:
            Accuracy is the ratio of correct predictions to the total number of predictions.
        """
        correct_predictions = 0
        total_predictions = len(labels)
        for i in range(len(predicted_labels)):
            if labels[i] == predicted_labels[i]:
                correct_predictions += 1
        accuracy = correct_predictions / total_predictions
        return accuracy
