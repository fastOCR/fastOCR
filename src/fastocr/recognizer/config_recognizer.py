class RecognizerConfig():

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