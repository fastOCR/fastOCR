class RecognizerConfig():
    """
    Configuration class for the recognizer class. Used to instanciate recognizer models according to the specific arguments.

    Args :
        transformation (:obj:`str`, `optional`, defaults to 'TPS')
            Transformation stage module. Normalizes the input image using a Spatial Transformer Network to ease downstream stages.
            'TPS' or 'None'.
        prediction (:obj:`str`, `optional`, defaults to 'Attn')
            Prediction stage module. Estimates the output character sequence from the identified features of an image.
            'CTC' or 'Attn'
        sequence_modeling (:obj:`str`, `optional`, defaults to 'BiLSTM')
            Sequence modeling stage module. Captures the contextual information within a sequence of characters for the next stage  to  predict  each  character  more  robustly.
            'BiLSTM' or 'None'
        feature_extraction (:obj:`str`, `optional`, defaults to 'ResNet')
            Feature Extraction stage module. Maps  the  input  image  to a representation that focuses on the attributes relevant for character recognition.
            'VGG' or 'RCNN' or 'ResNet
        imgH (:obj:`int`, `optional`, defaults to 32)
            Height of input image.
        imgW (:obj:`int`, `optional`, defaults to 100)
            Width of input image.
        input_channel (:obj:`int`, `optional`, defaults to 1)
            Number of channels of image input.
        batch_max_length (:obj:`int`, `optional`, defaults to 25)
            Maximum output text length
        hidden_size (:obj:`int`, `optional`, defaults to 256)
            Size of the LSTM hidden state
        output_channel (:obj:`int`, `optional`, defaults to 512)
            Number of output channels of the feature extractor
        num_fiducial (:obj:`int`, `optional`, defaults to 20)
            Number of fiducial points of TPS-STN
        rgb (:obj:`bool`, `optional`, defaults to False)
            Whether to use rgb input
        sensitive (:obj:`bool`, `optional`, defaults to False)
            Whether to use sensitive character mode
        PAD (:obj:`bool`, `optional`, defaults to False)
            Whether to keep ratio then pad for image resize
        batch_size (:obj:`int`, `optional`, defaults to 1)
            Input batch size
        charcaters (:obj:`str`, `optional`, defaults to '0123456789abcdefghijklmnopqrstuvwxyz')
            Characters to recognize
    """
    def __init__(
            self,
            transformation="TPS",
            prediction="Attn",
            sequence_modeling="BiLSTM",
            feature_extraction="ResNet",
            imgH=32,
            imgW=100,
            input_channel=1,
            batch_max_length=25,
            hidden_size=256,
            output_channel=512,
            num_fiducial=20,
            rgb=False,
            sensitive=False,
            PAD=False,
            batch_size=1,
            characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    ):
        self.transformation = transformation
        self.prediction = prediction
        self.sequence_modeling = sequence_modeling
        self.feature_extraction = feature_extraction
        self.imgH = imgH
        self.imgW = imgW
        self.input_channel = input_channel
        self.batch_max_length = batch_max_length
        self.hidden_size = hidden_size
        self.output_channel = output_channel
        self.num_fiducial = num_fiducial
        self.rgb = rgb
        self.sensitive = sensitive
        self.PAD = PAD
        self.batch_size = batch_size
        self.characters = characters
        if "CTC" in self.prediction:
            self.num_classes = len(characters) + 1
        else:
            self.num_classes = len(characters) + 2