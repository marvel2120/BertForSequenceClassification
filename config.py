class Config:
    def __init__(self):
        super().__init__()
        self.train_path = "data/train.txt"
        self.test_path = "data/test.txt"
        self.valid_path = "data/valid.txt"
        self.model_save_path = "model/model.pt"
        self.train_loss_path = "loss_record/train_loss_path.txt"
        self.test_loss_path = "loss_record/test_loss_path.txt"
        self.max_sen_len = 75
        self.epoch = 3
        self.batch_size = 32
