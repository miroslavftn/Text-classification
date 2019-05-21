import os
from models.bert_model.model import BertModel
from models.bert_model.processors import SST2Processor
from configs import DATA_DIR, SST2_DIR


nbepochs = 10.0
max_sequence_length = 128
batch_size = 4

models_dir = os.path.join(DATA_DIR, './bert_models/')


def train_bert(processor: SST2Processor, data_path: str):
    """
    Train BERT model on SST2 data
    :param processor:
    :param data_path:
    :return:
    """
    model = BertModel(data_dir=data_path,
                      processor=processor,
                      epochs=nbepochs,
                      max_seq_length=max_sequence_length,
                      batch_size=batch_size,
                      output_dir=models_dir,
                      device='cuda')
    model.train()


def predict_bert(text: str) -> str:
    """
    Predict for a single input text using singleton instance
    :param text:
    :return:
    """
    model = BertE2EBinLabels()
    return model.predict(text_a=text)


class BertE2EBinLabels:
    """
    Singleton class for Bert instance
    """
    instance = None

    def __new__(cls, *args, **kwargs):
        if not BertE2EBinLabels.instance:
            processor = SST2Processor()  # only for labels
            BertE2EBinLabels.instance = BertModel(data_dir='',
                                                  processor=processor,
                                                  epochs=nbepochs,
                                                  max_seq_length=max_sequence_length,
                                                  do_train=False,
                                                  batch_size=batch_size,
                                                  output_dir=models_dir,
                                                  device='cpu')
        return BertE2EBinLabels.instance


if __name__ == '__main__':
    processor = SST2Processor(SST2_DIR)
    train_bert(processor=processor, data_path=SST2_DIR)
    text_0 = 'uneasy mishmash of styles and genres. '
    text_1 = 'now trimmed by about 20 minutes , this lavish three-year-old production has enough grandeur and scale to satisfy as grown-up escapism .'
    print(predict_bert(text_0))
    print(predict_bert(text_1))
