import torch
import cv2
import numpy as np
import torch.nn.functional as F
from .encoding_utils import CTCLabelConverter, AttnLabelConverter
from .model_recognizer import RecognizerModel
from .pretrained_utils import PreTrainedModel

class Recognizer(PreTrainedModel):
    def __init__(self, config, weights_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weights_path = weights_path
        self.model = self._init_model()
        if 'CTC' in self.config.prediction:
            self.converter = CTCLabelConverter(self.config.characters)
        else:
            self.converter = AttnLabelConverter(self.config.characters)


    def _init_model(self):
        model = RecognizerModel(self.config)
        model = torch.nn.DataParallel(model).to(self.device)
        model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
        model.eval()
        return model

    def _preprocess_images(self, imgs):
        imgs = [cv2.resize(img, (self.config.imgW, self.config.imgH)) for img in imgs]
        imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        imgs = np.array(imgs)/255.0
        imgs = imgs[:,np.newaxis,:,:]
        imgs = torch.FloatTensor(imgs).to(self.device)
        imgs.sub_(0.5).div_(0.5)
        return imgs

    def predict(self, imgs):
        imgs = self._preprocess_images(imgs)
        batch_size = imgs.shape[0]
        length_for_pred = torch.IntTensor([self.config.batch_max_length] * batch_size).to(self.device)
        text_for_pred = torch.LongTensor(batch_size, self.config.batch_max_length + 1).fill_(0).to(self.device)
        if 'CTC' in self.config.prediction:
            preds = self.model(imgs, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, preds_size)

        else:
            preds = self.model(imgs, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = self.converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        output = []
        for pred, pred_max_prob in zip(preds_str, preds_max_prob):
            if 'Attn' in self.config.prediction:
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
            confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            output.append({"text": pred, "score": confidence_score})

        return output

