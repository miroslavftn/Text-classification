# coding=utf-8
import os
from typing import List, Dict

import numpy as np
import torch
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.nn import CrossEntropyLoss
from torch.nn.functional import cross_entropy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange, tqdm

from models.bert_model.processors import DataProcessor
from models.bert_model.utils import convert_examples_to_features, convert_example_to_feature, logger, create_dirs
from models.bert_model.utils import warmup_linear, accuracy
from models.bert_model.trainer import Trainer

SEED = 42

def set_seeds(seed, cuda=True):
    """ Set Numpy and PyTorch seeds.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    logger.info("==> 🌱 Set NumPy and PyTorch seeds.")


class BertModel(object):
    def __init__(self, data_dir: str, processor: DataProcessor, bert_model: str = 'bert-base-uncased',
                 output_dir: str = './models/',
                 max_seq_length: int = 64, do_train: bool = True, do_eval: bool = True, lowercase: bool = True,
                 batch_size: int = 4, learning_rate: float = 2e-5, epochs: float = 3, warmup_proportion: float = 0.1,
                 device: str = 'cuda'):
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.do_train = do_train
        self.do_eval = do_eval
        self.lowercase = lowercase
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.warmup_proportion = warmup_proportion
        self.processor = processor
        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.valid_loader = None
        self.device = torch.device(device)
        logger.info('==> 💻 Device: {} n_gpu: {}'.format(self.device, 1))
        set_seeds(SEED)

        self.tokenizer = self.init_tokenizer()

        if self.do_train:
            self.model = self.init_model()
            self.train_examples = self.processor.get_train_examples()
            self.num_train_steps = int(len(self.train_examples) / self.batch_size) * self.epochs
            self.optimizer = self.init_optimizer()
            self.train_loader = self.get_train_data()
        if self.do_eval:
            # load pre-trained model
            if not self.do_train: self.model = self.init_model(load=True)
            self.eval_examples = self.processor.get_dev_examples()
            self.valid_loader = self.get_eval_data()

        create_dirs(self.output_dir)
        if self.do_train:
            self.trainer = Trainer(model=self.model,
                                   optimizer=self.optimizer,
                                   epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   num_train_steps=self.num_train_steps,
                                   device=self.device,
                                   train_loader=self.train_loader,
                                   val_loader=self.valid_loader)

    def init_tokenizer(self) -> BertTokenizer:
        """
        Load pre-trained BERT tokenizer
        :return:
        """
        logger.info("\n==> 🚀 Initializing tokenizer:")
        return BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.lowercase)

    def init_model(self, load: bool = False) -> BertForSequenceClassification:
        """
        Initialize BertForSequenceClassification model
        :param load: If true, load custom pre-trained model
        :return:
        """
        if load:
            logger.info("\n==> 🚀 Loading model:")
            output_model_file = os.path.join(self.output_dir, WEIGHTS_NAME)
            output_config_file = os.path.join(self.output_dir, CONFIG_NAME)
            config = BertConfig(output_config_file)
            model = BertForSequenceClassification(config, num_labels=self.num_labels)
            model.load_state_dict(torch.load(output_model_file))

        else:
            logger.info("\n==> 🚀 Initializing model:")
            model = BertForSequenceClassification.from_pretrained(self.bert_model,
                                                                  cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                                                  num_labels=self.num_labels)

        model.to(self.device)
        return model

    def init_optimizer(self) -> BertAdam:
        """
        Initialize BertAdam optimizer.
        :return:
        """
        logger.info("\n==> 🚀 Initializing optimizer:")
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = self.num_train_steps

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.learning_rate,
                             warmup=self.warmup_proportion,
                             t_total=t_total)
        return optimizer

    def get_train_data(self) -> DataLoader:
        """
        Load training data, prepare inputs for the model. Do random sampling and convert to DataLoader object
        :return:
        """
        logger.info("==> 🍣 Loading train data:")
        train_features = convert_examples_to_features(
            self.train_examples, self.label_list, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        logger.info("\n==> 🚿 Transforming training data:")
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        return train_dataloader

    def get_eval_data(self) -> DataLoader:
        """
        Load and transform eval data
        :return: eval data loader
        """
        logger.info("==> 🍣 Loading eval data:")

        eval_features = convert_examples_to_features(
            self.eval_examples, self.label_list, self.max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        logger.info("\n==> 🚿 Transforming:")
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.batch_size)
        return eval_dataloader

    def transform_text(self, text_a: str, text_b: str) -> (List, List, List):
        """
        Transform input text into required features for a prediction
        :param text_a: input text
        :param text_b: input text
        :return: Tuple with (ids, mask, segments)
        """
        logger.info("==> 🍣 Transforming text...")
        feature = convert_example_to_feature(text_a, text_b, self.max_seq_length, self.tokenizer)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)

        return input_ids, input_mask, segment_ids

    def save_model(self):
        """
        Save Pytorch model and configs
        :return:
        """
        # Save a trained model and the associated configuration
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        output_model_file = os.path.join(self.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("==> 📂 Saved: {0}".format(output_model_file))
        output_config_file = os.path.join(self.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())
            logger.info("==> 📂 Saved: {0}".format(output_config_file))

    def save_eval_results(self, result: Dict):
        """
        Save eval results
        :param result:
        :return:
        """
        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("==> 📂 Saved: {0}".format(output_eval_file))

    def train(self):
        self.trainer.run_train_loop()
        self.model = self.trainer.model
        self.save_model()

    def predict(self, text_a: str, text_b: str = None) -> str:
        """
        Make a prediction for a single input text
        :param text_a:
        :param text_b:
        :return: predicted label
        """

        input_ids, input_mask, segment_ids = self.transform_text(text_a=text_a, text_b=text_b)
        logger.info("n==> 🛍️ Running evaluation")
        logger.info("  Batch size = %d", self.batch_size)

        # change mode to eval
        self.model.eval()
        # reshape tensors
        input_ids = input_ids.to(self.device).view(1, self.max_seq_length)
        input_mask = input_mask.to(self.device).view(1, self.max_seq_length)
        segment_ids = segment_ids.to(self.device).view(1, self.max_seq_length)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        return self.label_list[np.argmax(logits, axis=1)[0]]
