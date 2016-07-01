"""

This is the abstract base class that all models should inherit from

It has a few main components that all models need to have

constants should be added as well

"""

import abc


class ModelBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def set_model_architecture(self):
        """
        Create the model architecture (in tensorflow)
        Returns:
            Tensorflow Model
        """
        return

    @abc.abstractmethod
    def save_architecture(self, filename):
        """
        Saves the Tensorflow model to disk
        """
        return

    @abc.abstractmethod
    def load_architecture(self, filename):
        """
        Loads a model architecture from disk
        Returns:
            Tensorflow Model
        """

    @abc.abstractmethod
    def train(self):
        """
        Train the model and returns the trained model
        Returns:
            Trained Tensorflow Model
        """
        return

    @abc.abstractmethod
    def save_weights(self, filename):
        """
        Save the model weights to disk
        """

    @abc.abstractmethod
    def load_weights(self, filename):
        """
        Load weights from disk
        """

    @abc.abstractmethod
    def loss(self, logits, labels):
        """
        Calculates the loss for
        """
        return

    @abc.abstractmethod
    def inference(self, images):
        """
        Takes images and returns inferences\
        """
        return
