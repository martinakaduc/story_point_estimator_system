import _pickle as cPickle
import numpy as np
import gzip
import utils
import tensorflow as tf
from subprocess import Popen, PIPE

tf.compat.v1.disable_eager_execution()

MAX_LEN = 100
emb_weight_file = "lstm2v_tpssoft_dim50.pkl"
dict_file = "tpssoft.dict.pkl.gz"
model_file = "kanoo_lstm_highway_dim50_reginphid_prefixed_lm_poolmean.pb"

datasetDict = {
    'mesos': 'apache',
    'usergrid': 'apache',
    'appceleratorstudio': 'appcelerator',
    'aptanastudio': 'appcelerator',
    'titanium': 'appcelerator',
    'duracloud': 'duraspace',
    'bamboo': 'jira',
    'clover': 'jira',
    'jirasoftware': 'jira',
    'crucible': 'jira',
    'moodle': 'moodle',
    'datamanagement': 'lsstcorp',
    'mule': 'mulesoft',
    'mulestudio': 'mulesoft',
    'springxd': 'spring',
    'talenddataquality': 'talendforge',
    'talendesb': 'talendforge',
    'kanoo': 'tpssoft'
}

class DeepSE():
    def __init__(self, emb_weight_file, dict_file, model_file, max_len=100):
        self.sess = tf.compat.v1.Session()
        self.tokenizer_cmd = ['perl', 'tokenizer.perl', '-l', 'en', '-q', '-']

        self.dictionary = self.read_dictionary(dict_file)

        self.emb_weight = self.read_emb_weight(emb_weight_file)
        self.vocab_size, self.emb_dim = self.emb_weight.shape
        self.max_len = max_len

        self.regres_layer = None
        self.title_emb = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len, self.emb_dim], name='title_inp')
        self.descr_emb = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len, self.emb_dim], name='descr_inp')
        self.title_mask = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len], name='title_mask')
        self.descr_mask = tf.compat.v1.placeholder(np.float32, shape = [None, self.max_len], name='descr_mask')
        self.init_model(model_file)

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

    def read_emb_weight(self, weight_file):
        with open("emb_weights/" + weight_file, "rb") as wf:
            weights = cPickle.load(wf)
            return weights

    def read_dictionary(self, dict_file):
        f_dict = gzip.open("dicts/" + dict_file, 'rb')
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

    def init_model(self, model_file):
        f = tf.io.gfile.GFile("./bestModelsPb/" + model_file, 'rb')
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

    def inference(self, title, desc):
        prep_t, prep_d = self.preprocess_text(title, desc)
        title_vec, title_mask = self.create_mask(prep_t)
        descr_vec, descr_mask = self.create_mask(prep_d)

        title_emb, descr_emb = self.to_features([title_vec, descr_vec])

        results = self.sess.run(self.regres_layer, {self.title_emb: title_emb,
                                                    self.descr_emb: descr_emb,
                                                    self.title_mask: title_mask,
                                                    self.descr_mask: descr_mask})
        return_sp = []
        for sp in results:
            return_sp.append(utils.nearest_fib(sp[0]))

        return return_sp

if __name__ == '__main__':
    title = "Test different versions of perf"
    descr = "Test across different kernel versions (at least 2.6.XX and 3.X) and across different distributions. Test input flags and parsing output."
    gt_sp = 3

    deep_se_model = DeepSE(emb_weight_file, dict_file, model_file, max_len=MAX_LEN)
    prediction = deep_se_model.inference([title], [descr])

    print("Prediction:", prediction)
    print("Ground Truth:", gt_sp)
