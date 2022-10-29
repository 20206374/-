import numpy
import torch
import torch.nn as nn

from selfattention import SelfAttention


class Interstellar(nn.Module):
    def __init__(self, options):
        super(Interstellar, self).__init__()
        self._options = options
        hidden_size = self.hidden_size = options.hidden_size

        self.sub_embed = nn.Embedding(options._ent_num, hidden_size)
        self.rel_embed = nn.Embedding(options._rel_num, hidden_size)
        self.obj_embed = nn.Embedding(options._ent_num, hidden_size)

        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)

        self.drop = nn.Dropout(options.drop)
        # self.gate = nn.Linear(2*hidden_size, hidden_size)
        self.gate1 = nn.Linear(2 * hidden_size, hidden_size)
        self.gate2 = nn.Linear(2 * hidden_size, hidden_size)
        self.gate3 = nn.Linear(2 * hidden_size, hidden_size)
        self.gate4 = nn.Linear(2 * hidden_size, hidden_size)

        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W4 = nn.Linear(hidden_size, hidden_size)
        self.W5 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W6 = nn.Linear(hidden_size, hidden_size)

        self.idd = lambda x:x

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            if param.dim()>1:
                nn.init.xavier_uniform_(param.data)

     #测试
    def para_print(self):
        print("Model state_dict")
        for param_tensor in self.state_dict():
            print(param_tensor,"\t",self.state_dict()[param_tensor].size())


    def _loss(self, seqs, struct):
        res_outputs, obj_embed = self.get_res_outputs(seqs, struct)
        positive = self.get_pair(res_outputs, obj_embed)
        negative = self.get_tail(res_outputs)
        max_neg = torch.max(negative, 1, keepdim=True)[0]

        loss = -positive + max_neg + torch.log(torch.sum(torch.exp(negative - max_neg), 1))
        return torch.sum(loss)

    def get_pair(self, x1, x2):
        return torch.sum(x1*x2, dim=1)

    def get_tail(self, x):
        return torch.mm(x, self.obj_embed.weight.transpose(1,0))

    def forward(self, seqs, struct=[2,10,2,2,0,0,0,0,0,0,0]):
        length = seqs.size(1)

        # with tf.name_scope('sub_emb'):
        sub = seqs[:, :-1:2]
        sub_emb_f = self.sub_embed(sub)
        #with tf.name_scope('relational_emb'):
        rel = seqs[:, 1::2]
        rel_emb_f = self.rel_embed(rel)
        # with tf.name_scope('obj_emb'):
        obj = seqs[:, 2::2]
        obj_emb_f = self.obj_embed(obj)


        forw_seq = []
        for i in range(1):
            f_seq = []
            f_seq.append(sub_emb_f[:, i, :])
            f_seq.append(rel_emb_f[:, i, :])
            forw_seq.append(f_seq)

        # h0
        # with tf.name_scope('h1'):
        #   h_f = sub_emb_f[:, 0, :]

        outputs_f = []  # 存放输出的嵌入
        for i in range(1):
            out_f, h_f = self.get_connect(forw_seq[i], h_f, struct)
            outputs_f.append(out_f)

        outputs_f = self.bn3(self.drop(torch.cat(outputs_f, dim=0)))

        target_f = obj_emb_f.permute(1, 0, 2).contiguous().view(-1, self.hidden_size)
        return outputs_f, target_f

    def get_res_outputs(self, seqs, struct):
        length = seqs.size(1)
        sub = seqs[:, :-1:2]
        rel = seqs[:, 1::2]
        obj = seqs[:, 2::2]

        sub_emb_f = self.sub_embed(sub)
        rel_emb_f = self.rel_embed(rel)
        obj_emb_f = self.obj_embed(obj)

        forw_seq = []
        for i in range(length//2):
            f_seq = []
            f_seq.append(sub_emb_f[:,i,:])
            f_seq.append(rel_emb_f[:,i,:])
            forw_seq.append(f_seq)

        # h0
        h_f = sub_emb_f[:,0,:]

        outputs_f = []#存放输出的嵌入
        for i in range(length//2):
            out_f, h_f = self.get_connect(forw_seq[i], h_f, struct)
            outputs_f.append(out_f)

        outputs_f = self.bn3(self.drop(torch.cat(outputs_f, dim=0)))

        target_f = obj_emb_f.permute(1,0,2).contiguous().view(-1, self.hidden_size)
        return outputs_f, target_f
    
    def get_connect(self, ent_rel, h_tm1, struct):#计算输出
        st = self.bn1(ent_rel[0])
        rt = self.bn2(ent_rel[1])
        zeros = torch.zeros(st.size()).cuda()
        ops = ['add', 'mult', 'complx', 'gate']

        W1 = [self.idd, self.W1][struct[5]]
        W2 = [self.idd, self.W2][struct[6]]

        # with tf.name_scope('op1'):
        if struct[0] == 0:
            op1 = self.ops(st, h_tm1, 'add', W1, W2)
        elif struct[0] == 1:
            op1 = self.ops(st, h_tm1, 'mult', W1, W2)
        elif struct[0] == 2:
            op1 = self.ops(st, h_tm1, 'complx', W1, W2)
        elif struct[0] == 3:
            op1 = self.ops(st, h_tm1, 'gate', W1, W2)

        if struct[3] == 1:
            op1 = torch.tanh(op1)
        elif struct[3] == 2:
            op1 = torch.sigmoid(op1)


        W3 = [self.idd, self.W3][struct[7]]
        W4 = [self.idd, self.W4][struct[8]]

        op2_1 = self.ops(st, rt, ops[struct[1] % 4], W3, W4)
        op2_2 = self.ops(h_tm1, rt, ops[struct[1] % 4], W3, W4)
        op2_3 = self.ops(op1, rt, ops[struct[1] % 4], W3, W4)
        op2_4 = self.ops(zeros, rt, ops[struct[1] % 4], W3, W4)
        op2_total = [op2_1, op2_2, op2_3, op2_4]

        op2_total = torch.stack(op2_total)
        # print(op2_total.size())
        device = torch.device("cuda")
        op2_total = op2_total.cuda()

        op2_out = SelfAttention(num_attention_heads=1, input_size=2, hidden_size=self.hidden_size,
                                hidden_dropout_prob=0.5)
        op2_out = op2_out.forward(op2_total)
        op2_outs = op2_out.data.cpu().numpy()
        choice = numpy.argmax(op2_outs)
        choice = choice % 4
        # op2_outs = op2_outs.mean(1).mean(1)
        # choice = numpy.argmax(op2_outs)
        # print("choice")
        # print(choice)
        # 第二位数字struct[1]除以4，商为2表示操作符Or的输入含有操作Os运算后的输出结果，余数为2代表Or的combinator
        op2_in = [st, h_tm1, op1, zeros][choice]
        op2 = self.ops(op2_in, rt, ops[struct[1]%4], W3, W4)# 对Or的操作

        # struct[4]是第五位数字，代表Or的激活函数
        if struct[4] == 1:
            op2 = torch.tanh(op2)
        elif struct[4] == 2:
            op2 = torch.sigmoid(op2)

        #对Ov的操作
        W5 = [self.idd, self.W5][struct[9]]
        W6 = [self.idd, self.W6][struct[10]]

        op3_1 = self.ops(st, rt, ops[struct[1] % 4], W5, W6)
        op3_2 = self.ops(h_tm1, rt, ops[struct[1] % 4], W5, W6)
        op3_3 = self.ops(op1, rt, ops[struct[1] % 4], W5, W6)
        op3_4 = self.ops(zeros, rt, ops[struct[1] % 4], W5, W6)
        op3_total = [op3_1, op3_2, op3_3, op3_4]

        op3_total = torch.stack(op3_total)
        # print(op3_total.size())
        device = torch.device("cuda")
        op3_total = op3_total.cuda()

        op3_out = SelfAttention(num_attention_heads=1, input_size=2, hidden_size=self.hidden_size,
                                hidden_dropout_prob=0.5)
        op3_out = op3_out.forward(op3_total)
        op3_outs = op3_out.data.cpu().numpy()
        choice = numpy.argmax(op3_outs)
        choice = choice % 4
        # op3_outs = op3_outs.mean(1).mean(1)
        # choice = numpy.argmax(op3_outs)

        op3_in = [st, h_tm1, op1, zeros][choice]
        op3 = self.ops(op3_in, op2, ops[struct[2]%4], W5, W6)

        ht = op2
        return op3.view(-1, self.hidden_size), ht.view(-1, self.hidden_size)


    def ops(self, x, h, name, W1, W2):
        x = W1(x)
        h = W2(h)
        if name == 'add':
            out = x+h
        elif name == 'mult':
            out = x*h
        elif name == 'complx':
            x1, x2 = torch.chunk(x, 2, dim=1)
            h1, h2 = torch.chunk(h, 2, dim=1)
            o1 = x1*h1 - x2*h2
            o2 = x1*h2 + x2*h1
            out = torch.cat([o1, o2], dim=1)
        elif name == 'gate':
            R_t = torch.sigmoid(self.gate1(torch.cat([x, h], dim=1)))
            Z_t = torch.sigmoid(self.gate2(torch.cat([x, h], dim=1)))
            H_t_h = torch.tanh(self.gate3(torch.cat([x, (torch.mul(x, h))], dim=1)))
            out = torch.mul(Z_t, h) + torch.mul((1 - Z_t), H_t_h)
            # gate = torch.sigmoid(self.gate(torch.cat([x, h], dim=1)))
            # out = gate * h + (1-gate)*x
        return out.squeeze()

    def test(self):
        for param in self.parameters():
            if param.dim()>1:
                pass
