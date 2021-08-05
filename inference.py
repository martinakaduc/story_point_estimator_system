import _pickle as cPickle
import numpy as np
import gzip
import utils
import tensorflow as tf
from subprocess import Popen, PIPE
from sklearn.neighbors import NearestNeighbors
import csv
import os 

tf.compat.v1.disable_eager_execution()

MAX_LEN = 100
project_name = "kanoo"

datasetDict = {
    # 'mesos': 'apache',
    # 'usergrid': 'apache',
    # 'titanium': 'appcelerator',
    'kanoo': 'tpssoft',
    'opus': 'tpssoft',
    'vk': 'tpssoft'
}

class DeepSE():
    def __init__(self, project_name, max_len=MAX_LEN):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth=True
        
        self.sess = tf.compat.v1.Session(config=config)
        self.tokenizer_cmd = ['perl', 'tokenizer.perl', '-l', 'en', '-q', '-']
        self.org_name = datasetDict[project_name]

        self.dictionary = self.read_dictionary(self.org_name)

        self.emb_weight = self.read_emb_weight(self.org_name)
        self.vocab_size, self.emb_dim = self.emb_weight.shape
        self.max_len = max_len

        self.regres_layer = None
        self.title_emb = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len, self.emb_dim], name='title_inp')
        self.descr_emb = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len, self.emb_dim], name='descr_inp')
        self.title_mask = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len], name='title_mask')
        self.descr_mask = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len], name='descr_mask')
        self.historical_data = []
        self.historical_embedding = None

        self.init_model(project_name)
        self.get_embedding_db(project_name)

    def preprocess_text(self, title, description):
        title = self.tokenize(title)
        description = self.tokenize(description)

        seqs = [[None] * len(title), [None] * len(description)]
        for i, sentences in enumerate([title, description]):
            for idx, ss in enumerate(sentences):
                words = ss.strip().lower().split()
                seqs[i][idx] = [self.dictionary[w] if w in self.dictionary else 0 for w in words]

        return seqs[0], seqs[1]

    def tokenize(self, sentences):
        # print ('Tokenizing..')
        text = "\n".join(sentences)
        tokenizer = Popen(self.tokenizer_cmd, stdin=PIPE, stdout=PIPE)
        tok_text, _ = tokenizer.communicate(bytes(text.encode("utf-8")))
        toks = tok_text.decode("utf-8").split('\n')[:-1]
        # print ('Done')

        return toks

    def read_emb_weight(self, org_name):
        with open("emb_weights/" + org_name + ".pkl", "rb") as wf:
            weights = cPickle.load(wf)
            return weights

    def read_dictionary(self, org_name):
        f_dict = gzip.open("dicts/" + org_name + ".dict.pkl.gz", 'rb')
        return cPickle.load(f_dict)

    def to_features(self, list_seqs):
        weight = np.zeros((self.vocab_size + 1, self.emb_dim)).astype(np.float32)
        weight[1:] = self.emb_weight

        list_feats = []
        for seqs in list_seqs:
            n_samples, seq_len = seqs.shape
            feat = weight[seqs.flatten()].reshape([n_samples, seq_len, self.emb_dim])
            list_feats.append(feat)
        return list_feats

    def create_mask(self, seqs):
        new_seqs = []
        for idx, s in enumerate(seqs):
            new_s = [w for w in s if w < self.vocab_size]
            if len(new_s) == 0: new_s = [0]
            new_seqs.append(new_s)

        seqs = new_seqs

        lengths = [min(self.max_len, len(s)) for s in seqs]
        n_samples = len(lengths)

        x = np.zeros((n_samples, self.max_len)).astype(np.int64)
        mask = np.zeros((n_samples, self.max_len)).astype(np.float32)

        for i, s in enumerate(seqs):
            l = lengths[i]
            mask[i, :l] = 1
            x[i, :l] = s[:l]
            x[i, :l] += 1

        return x, mask

    def init_model(self, project_name):
        f = tf.io.gfile.GFile("bestModelsPb/" + project_name + ".pb", 'rb')
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        f.close()

        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, {"title_inp": self.title_emb,
                                        "descr_inp": self.descr_emb,
                                        "title_mask": self.title_mask,
                                        "descr_mask": self.descr_mask})

        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # print('Check out the input placeholders:')
        # nodes = [n.name + ' => ' +  n.op for n in self.sess.graph_def.node if n.op in ('Placeholder')]
        # for node in nodes:
        #     print(node)

        out_tensor_name = self.sess.graph.as_graph_def().node[-1].name + ":0"
        self.regres_layer = self.sess.graph.get_tensor_by_name(out_tensor_name)

    def _embed_input(self, title, desc):
        prep_t, prep_d = self.preprocess_text(title, desc)
        title_vec, title_mask = self.create_mask(prep_t)
        descr_vec, descr_mask = self.create_mask(prep_d)

        title_emb, descr_emb = self.to_features([title_vec, descr_vec])
        return title_emb, title_mask, descr_emb, descr_mask

    def inference(self, title, desc, return_history=False):
        title_emb, title_mask, descr_emb, descr_mask = self._embed_input(title, desc)

        results = self.sess.run(self.regres_layer, {self.title_emb: title_emb,
                                                    self.descr_emb: descr_emb,
                                                    self.title_mask: title_mask,
                                                    self.descr_mask: descr_mask})
        return_sp = []
        for sp in results:
            return_sp.append(utils.nearest_fib(sp[0]))

        if return_history:
            num_features = title_emb.shape[1] * title_emb.shape[2]

            title_emb = np.reshape(title_emb, (-1, num_features))
            descr_emb = np.reshape(descr_emb, (-1, num_features))

            total_emb = np.concatenate((title_emb, descr_emb), axis=-1)

            distances, indexes = self.historical_embedding.kneighbors(total_emb)
            histories = self._get_histories_by_idx(indexes, return_sp)

            return return_sp, histories

        return return_sp

    def _get_histories_by_idx(self, indexes, sps):
        results = []
        for neighbor_idxs, sp in zip(indexes, sps):
            histories_one_sample = []
            for idx in neighbor_idxs:
                if int(self.historical_data[idx][-1]) == sp:
                    histories_one_sample.append(self.historical_data[idx])
            results.append(histories_one_sample)

        return results

    def get_embedding_db(self, project_name):
        if not os.path.exists("database/embedding/" + project_name + ".pkl"):
            self._embed_db(project_name)

        with open("database/embedding/" + project_name + ".pkl", "rb") as ef:
            self.historical_embedding = cPickle.load(ef)

        with open("database/" + project_name + ".csv", 'r', encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0: continue
                self.historical_data.append(row)

    def _embed_db(self, project_name):
        list_title = []
        list_descr = []

        with open("database/" + project_name + ".csv", 'r', encoding="utf-8") as f:
            csv_reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i == 0: continue
                list_title.append(row[1])
                list_descr.append(row[2])

        title_emb, _, descr_emb, _  = self._embed_input(list_title, list_descr)
        num_features = title_emb.shape[1] * title_emb.shape[2]

        title_emb = np.reshape(title_emb, (-1, num_features))
        descr_emb = np.reshape(descr_emb, (-1, num_features))

        total_emb = np.concatenate((title_emb, descr_emb), axis=-1)
        NN = NearestNeighbors(n_neighbors=16, metric="euclidean", n_jobs=-1)
        NN.fit(total_emb)

        f = open("database/embedding/" + project_name + ".pkl", "wb")
        cPickle.dump(NN, f, -1)
        f.close()

if __name__ == '__main__':
    title = "Test different versions of perf"
    descr = "Test across different kernel versions (at least 2.6.XX and 3.X) and across different distributions. Test input flags and parsing output."
    gt_sp = 3

    deep_se_model = DeepSE(project_name, max_len=MAX_LEN)
    prediction, history = deep_se_model.inference([title], [descr], return_history=True)

    print("Prediction:", prediction)
    print("Ground Truth:", gt_sp)
    print("History: ", history)
