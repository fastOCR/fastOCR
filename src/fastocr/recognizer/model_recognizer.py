import torch.nn as nn
from .modeling_utils.transformation import TPS_SpatialTransformerNetwork
from .modeling_utils.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from .modeling_utils.sequence_modeling import BidirectionalLSTM
from .modeling_utils.prediction import Attention


class RecognizerModel(nn.Module):

    def __init__(self, config):
        super(RecognizerModel, self).__init__()
        self.config = config
        self.stages = {'Trans': self.config.transformation, 'Feat': self.config.feature_extraction,
                       'Seq': self.config.sequence_modeling, 'Pred': self.config.prediction}


        """ Transformation """
        if self.config.transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=self.config.num_fiducial, I_size=(self.config.imgH, self.config.imgW), I_r_size=(self.config.imgH, self.config.imgW), I_channel_num=self.config.input_channel)

        """ FeatureExtraction """
        if self.config.feature_extraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(self.config.input_channel, self.config.output_channel)
        elif self.config.feature_extraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(self.config.input_channel, self.config.output_channel)
        elif self.config.feature_extraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(self.config.input_channel, self.config.output_channel)
        self.FeatureExtraction_output = self.config.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        if self.config.sequence_modeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.config.hidden_size, self.config.hidden_size),
                BidirectionalLSTM(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size))
            self.SequenceModeling_output = self.config.hidden_size
        else:
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if self.config.prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, self.config.num_classes)
        elif self.config.prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, self.config.hidden_size, self.config.num_classes)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.config.batch_max_length)

        return prediction
