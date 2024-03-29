import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiDAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)

        self.char_conv = nn.Conv1d(args.char_dim, args.char_channel_size, args.char_channel_width)

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # highway network
        # assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        #assert self.args.word_dim == 2 * self.args.hidden_size
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(2 * args.hidden_size, 2 * args.hidden_size),
                                  nn.ReLU(inplace=True)))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(2 * args.hidden_size, 2 * args.hidden_size),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # 4. Attention Flow Layer
        self.att_weight_cq = Linear(args.hidden_size * 6, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 8,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.modeling_LSTM2 = LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight = Linear(args.hidden_size * 10, 1, dropout=args.dropout)
        self.p2_weight = Linear(args.hidden_size * 10, 1, dropout=args.dropout)

        self.output_LSTM = LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout,inplace=True)

    def forward(self, c_char, q_char, c_word, q_word, c_lens, q_lens):
        # TODO: More memory-efficient architecture
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len,char_dim, word_len)
            x = x.view(-1, self.args.char_dim, x.size(2))
            # (batch * seq_len, char_channel_size,conv_len)
            x = self.char_conv(x)
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.args.char_channel_size)

            return x

        def highway_network(x):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            # c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            # cq_tiled = c_tiled * q_tiled
            # cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            # cq = []
            c_ex = c.unsqueeze(2).expand(-1,-1,q_len,-1)
            q_ex = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            s = self.att_weight_cq(torch.cat((c_ex,q_ex,c_ex*q_ex),dim=-1)).squeeze()


            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight(torch.cat((g,m),dim=-1))).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight(torch.cat((g,m2),dim=-1))).squeeze()

            return p1, p2

        # 1. Character Embedding Layer

        c_maxlen = c_word.size()[1]
        c_char = char_emb_layer(c_char)
        q_char = char_emb_layer(q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(c_word)
        q_word = self.word_emb(q_word)

        # Highway network
        c_cat = torch.cat([F.pad(c_char,(0,0,0,c_maxlen-c_char.size()[-2])), c_word], dim=-1)
        q_cat = torch.cat([q_char, q_word], dim=-1)
        c = highway_network(c_cat)
        q = highway_network(q_cat)

        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]

        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)

        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]

        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        
        # (batch, c_len), (batch, c_len)
        p1_padded = F.pad(p1,(0,c_maxlen - p1.size()[-1]))
        p2_padded = F.pad(p2,(0,c_maxlen - p2.size()[-1]))
        return p1_padded, p2_padded
