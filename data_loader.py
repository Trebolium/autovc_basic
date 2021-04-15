from torch.utils import data
import torch
import numpy as np
import os, pdb, pickle, random
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    # this object will contain both melspecs and speaker embeddings taken from the train.pkl
    def __init__(self, config):
        """Initialize and preprocess the Utterances dataset."""
        self.spmel_dir = config.spmel_dir
        self.len_crop = config.len_crop
        self.step = 10
        self.file_name = config.file_name
        self.one_hot = config.one_hot
        self.config = config
        # metaname = os.path.join(self.spmel_dir, "all_meta_data.pkl")
        meta_all_data = pickle.load(open('./all_meta_data.pkl', "rb"))
        # split into training data
        num_training_speakers=config.train_size
        random.seed(1)
        training_indices =  random.sample(range(0, len(meta_all_data)), num_training_speakers)
        training_set = []

        meta_training_speaker_all_uttrs = []
        # make list of training speakers
        for idx in training_indices:
            meta_training_speaker_all_uttrs.append(meta_all_data[idx])
        # get training files
        for speaker_info in meta_training_speaker_all_uttrs:
            speaker_id_emb = speaker_info[:2]
            speaker_uttrs = speaker_info[2:]
            num_files = len(speaker_uttrs) # first 2 entries are speaker ID and speaker_emb)
            training_file_num = round(num_files*0.9)
            training_file_indices = random.sample(range(0, num_files), training_file_num)

            training_file_names = []
            for index in training_file_indices:
                fileName = speaker_uttrs[index]
                training_file_names.append(fileName)
            training_set.append(speaker_id_emb+training_file_names)
            # training_file_names_array = np.asarray(training_file_names)
            # training_file_indices_array = np.asarray(training_file_indices)
            # test_file_indices = np.setdiff1d(np.arange(num_files_in_subdir), training_file_indices_array)
        meta = training_set
        # pdb.set_trace()
        with open(self.config.data_dir +'/model_saves/' +self.file_name +'/training_meta_data.pkl', 'wb') as train_pack:
            pickle.dump(training_set, train_pack)
        # pdb.set_trace()

        training_info = pickle.load(open(self.config.data_dir +'/model_saves/' +self.file_name +'/training_meta_data.pkl', 'rb'))
        num_speakers_seq = np.arange(len(training_info))
        self.one_hot_array = np.eye(len(training_info))[num_speakers_seq]
        self.spkr_id_list = [spkr[0] for spkr in training_info]

        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        # uses a different process thread for every self.steps of the meta content
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        # pdb.set_trace()    
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
    # this function is called within the class init (after self.data_loader its the arguments) 
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            # pdb.set_trace()
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join(self.spmel_dir, tmp))
            dataset[idx_offset+k] = uttrs
                   
    """__getitem__ selects a speaker and chooses a random subset of data (in this case
    an utterance) and randomly crops that data. It also selects the corresponding speaker
    embedding and loads that up. It will now also get corresponding pitch contour for such a file""" 
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        # list_uttrs is literally a list of utterance from a single speaker
        list_uttrs = dataset[index]
        # pdb.set_trace()
        emb_org = list_uttrs[1]
        speaker_name = list_uttrs[0]
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        uttr_info = list_uttrs[a]
        
        spmel_tmp = uttr_info
        #spmel_tmp = uttr_info[0]
        #pitch_tmp = uttr_info[1]
        if spmel_tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - spmel_tmp.shape[0]
            uttr = np.pad(spmel_tmp, ((0,len_pad),(0,0)), 'constant')
        #    pitch = np.pad(pitch_tmp, ((0,len_pad),(0,0)), 'constant')
        elif spmel_tmp.shape[0] > self.len_crop:
            left = np.random.randint(spmel_tmp.shape[0]-self.len_crop)
            uttr = spmel_tmp[left:left+self.len_crop, :]
        #    pitch = pitch_tmp[left:left+self.len_crop, :]
        else:
            uttr = spmel_tmp
        #    pitch = pitch_tmp    

        # find out where speaker is in the order of the training list for one-hot
        for i, spkr_id in enumerate(self.spkr_id_list):
            if speaker_name == spkr_id:
                spkr_label = i
                break
        one_hot_spkr_label = self.one_hot_array[spkr_label]
        if self.one_hot==False:
            return uttr, emb_org, speaker_name # pitch
        else:
            return uttr, one_hot_spkr_label, speaker_name

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    
    

def get_loader(config, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Utterances(config)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader






