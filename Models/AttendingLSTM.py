import math
import torch
import torch.nn as nn

class AttendingLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, input_features: int):
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        
        #i_t
        # self.U_i = nn.Parameter(torch.Tensor(input_features, hidden_sz))
        # self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
        # #f_t
        # self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
        # #c_t
        # self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
        # #o_t
        # self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
        # self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
        self.W_query = torch.nn.Parameter(torch.Tensor(input_features, input_features))
        self.W_key = torch.nn.Parameter(torch.Tensor(input_features, input_features))
        self.W_value = torch.nn.Parameter(torch.Tensor(input_features, input_features))
        # query = embedded_sentence @ W_query
        # key = embedded_sentence @ W_key
        # value = embedded_sentence @ W_value
        self.d_k= float(input_features)
        self.binary_step = ApproxBinaryStepFunction()
        self.output_softmax = torch.nn.Linear(hidden_sz, 1)

        
        self.lstm = torch.nn.LSTM(input_size=input_sz, hidden_size=hidden_sz,batch_first=False,num_layers=3)
        self.init_weights()

        #x is the following order of data per day open, high, low, close, vol, quote asset vol, num of trades, taker buy base vol, taker buy quote vol
        #self attention

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def attention_calculations(self,x):
        Q = x @ self.W_query
        K = x @ self.W_key
        V = x @ self.W_value
        attention_scores = Q @ K.T / math.sqrt(self.d_k)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_vector = attention_weights @ V
        return attention_vector
    
    # aggragate to scalar and the value is a mean aggragate
    def aggragate_attention(self, attention_vector):
        # Mean aggregation
        # aggregate rows
        mean_value = torch.mean(attention_vector,dim = 0)
        #aggregate column
        mean_value = torch.mean(mean_value)
        return mean_value.reshape(1,1)


    #x is the time steps and v is the daily values
    def forward(self, x, v, init_states=None):
        
        """
        assumes x.shape represents (batch_size, sequence_size, input_size)
        """
        bs, _, seq_sz = v.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        # a = []
        # for t in range(seq_sz):
        #     v_t = v[:, :, t]
        #     v_t = torch.nn.functional.normalize(v_t)
        #     v_t = self.attention_calculations(v_t)
        #     a.append(v_t)
            
        # a = torch.stack(a)
        
        # print(a)
        
        h_t = self.lstm(v.resize(seq_sz,bs,_))[0][0]
        # for t in range(seq_sz):
        #     x_t = x[:, :, t]
        #     v_t = v[:, :, t]
        #     x_t = torch.nn.functional.normalize(x_t)
        #     v_t = torch.nn.functional.normalize(v_t)
        #     v_t = self.attention_calculations(v_t)
        #     #a_t = self.aggragate_attention(v_t)
        #     i_t = torch.tanh(v_t @ self.U_i + h_t @ self.V_i + self.b_i)
        #     f_t = torch.tanh(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
        #     g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
        #     o_t = torch.tanh(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
        #     c_t = f_t * c_t + i_t * g_t
        #     h_t = o_t * torch.tanh(c_t)
            
        #     hidden_seq.append(h_t.unsqueeze(0))
        
        #reshape hidden_seq p/ retornar
        #hidden_seq = torch.cat(hidden_seq, dim=0)
        #hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        prediction = self.output_softmax(h_t)
        prediction = self.binary_step.forward(prediction)
        return prediction#, hidden_seq, (h_t, c_t)


class ApproxBinaryStepFunction(nn.Module):
    def __init__(self, gain=100):
        super(ApproxBinaryStepFunction, self).__init__()
        self.gain = gain

    def forward(self, x):
        # Apply the scaled sigmoid function to approximate the binary step
        return torch.sigmoid(self.gain * x)
# import math
# import torch
# import torch.nn as nn

# class AttendingLSTM(nn.Module):
#     def __init__(self, input_sz: int, hidden_sz: int, input_features: int):
#         super().__init__()
#         self.input_size = input_sz
#         self.hidden_size = hidden_sz
        
#         #i_t
#         self.U_i = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
#         self.V_i = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
#         self.b_i = nn.Parameter(torch.Tensor(hidden_sz))
        
#         #f_t
#         self.U_f = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
#         self.V_f = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
#         self.b_f = nn.Parameter(torch.Tensor(hidden_sz))
        
#         #c_t
#         self.U_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
#         self.V_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
#         self.b_c = nn.Parameter(torch.Tensor(hidden_sz))
        
#         #o_t
#         self.U_o = nn.Parameter(torch.Tensor(input_sz, hidden_sz))
#         self.V_o = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
#         self.b_o = nn.Parameter(torch.Tensor(hidden_sz))
        
#         self.W_query = torch.nn.Parameter(torch.Tensor(input_features, 1))
#         self.W_key = torch.nn.Parameter(torch.Tensor(input_features, 1))
#         self.W_value = torch.nn.Parameter(torch.Tensor(input_features, 1))
#         # query = embedded_sentence @ W_query
#         # key = embedded_sentence @ W_key
#         # value = embedded_sentence @ W_value
#         self.d_k=1.0

#         self.init_weights()

#         #x is the following order of data per day open, high, low, close, vol, quote asset vol, num of trades, taker buy base vol, taker buy quote vol
#         #self attention

#     def init_weights(self):
#         stdv = 1.0 / math.sqrt(self.hidden_size)
#         for weight in self.parameters():
#             weight.data.uniform_(-stdv, stdv)

#     def attention_calculations(self,x):
#         Q = x @ self.W_query
#         K = x @ self.W_key
#         V = x @ self.W_value
#         attention_scores = Q @ K.T / math.sqrt(self.d_k)
#         attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
#         attention_vector = attention_weights @ V
#         return attention_vector
    
#     # aggragate to scalar and the value is a mean aggragate
#     def aggragate_attention(self, attention_vector):
#         # Mean aggregation
#         mean_value = torch.tensor.mean().reshape(1, 1)
#         return mean_value.squeeze()


#     #x is the time steps and v is the daily values
#     def forward(self, x, v, init_states=None):
        
#         """
#         assumes x.shape represents (batch_size, sequence_size, input_size)
#         """
#         bs, seq_sz, _ = x.size()
#         hidden_seq = []
        
#         if init_states is None:
#             h_t, c_t = (
#                 torch.zeros(bs, self.hidden_size).to(x.device),
#                 torch.zeros(bs, self.hidden_size).to(x.device),
#             )
#         else:
#             h_t, c_t = init_states
            
#         for t in range(seq_sz):
#             x_t = x[:, :, t]
#             v_t = v[:, :, t]
#             v_t = self.attention_calculations(v_t)
#             a_t = v_t #self.aggragate_attention(v_t)
#             i_t = torch.sigmoid(a_t @ self.U_i + h_t @ self.V_i + self.b_i)
#             f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
#             g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
#             o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
#             c_t = f_t * c_t + i_t * g_t
#             h_t = o_t * torch.tanh(c_t)
            
#             hidden_seq.append(h_t.unsqueeze(0))
        
#         #reshape hidden_seq p/ retornar
#         hidden_seq = torch.cat(hidden_seq, dim=0)
#         hidden_seq = hidden_seq.transpose(0, 1).contiguous()
#         return hidden_seq, (h_t, c_t),v_t
    