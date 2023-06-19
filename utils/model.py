# default imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import for TFN
from torch.autograd import Variable
from torch.nn.parameter import Parameter

#################
# call function #
#################
def get_model(flag, input_shapes, hidden_units, output_units):
    if flag == 'simple':
        model = ModalityFusionModel_simple(input_shapes, hidden_units, output_units)
    elif flag == 'attention':
        model = ModalityFusionModule_attention(input_shapes, hidden_units, output_units)
    else:
        NotImplementedError
    return model

############################
# simple mlp from baseline #
############################
class ModalityFusionModel_simple(nn.Module):
    def __init__(self, input_shape, hidden_units, output_units):
        super(ModalityFusionModel_simple, self).__init__()
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.fc_layer = nn.Linear(input_shape, hidden_units)
        self.fusion_layer = nn.Linear(hidden_units, hidden_units)
        self.output_layer = nn.Linear(hidden_units, output_units)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        fused_output = self.relu(self.fc_layer(inputs))
        fused_output = self.relu(self.fusion_layer(fused_output))
        output = self.output_layer(fused_output)

        return self.relu(output) 


########################
# attention based demo #
########################
class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = torch.softmax(scores, dim=-1)
        weighted_value = torch.matmul(attention_weights, value)

        return weighted_value

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

class ModalityFusionModule_attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModalityFusionModule_attention, self).__init__()
        self.self_attention = SelfAttention(input_dim, hidden_dim)
        self.mlp = MLP(input_dim, hidden_dim, output_dim)

    def forward(self, input_features):
        attended_features = self.self_attention(input_features)
        fused_features = torch.mean(attended_features, dim=1)
        output = self.mlp(fused_features)

        return output


##########################
# TFN: tensor fusion net #
##########################
'''
Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
modified for ACM-MM 2023
'''
class TFN(nn.Module):

    def __init__(self, input_dims, hidden_dims, dropouts, post_fusion_dim):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            #text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.text_hidden = hidden_dims[2]
        #self.text_out= text_out
        self.post_fusion_dim = post_fusion_dim

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.text_prob = dropouts[2]
        self.post_fusion_prob = dropouts[3]

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear((self.text_hidden + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, audio_h, video_h, text_h):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)
        _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))
        
        # next we do kronecker product between fusion_tensor and _text_h. This is even trickier
        # we have to reshape the fusion tensor during the computation
        # in the end we don't keep the 3-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(-1, (self.audio_hidden + 1) * (self.video_hidden + 1), 1)
        fusion_tensor = torch.bmm(fusion_tensor, _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = F.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y_2 = F.relu(self.post_fusion_layer_2(post_fusion_y_1))
        post_fusion_y_3 = F.sigmoid(self.post_fusion_layer_3(post_fusion_y_2))
        output = post_fusion_y_3 * self.output_range + self.output_shift

        return output

if __name__ == '__main__':
    # Example usage
    input_shapes = 83  # Example input shapes for 3 modalities
    hidden_units = 64
    output_units = 1

    # Get model
    #model = ModalityFusionModule_attention(input_shapes, hidden_units, output_units)
    model = TFN(
        input_dims=[None,None,None],
        hidden_dims=[1023,314,367], #video,audio,kinect
        dropouts=[0.01,0.01,0.01,0.01],
        post_fusion_dim=256
    )

    # Example forward pass
    input_data = torch.randn(161798, 83)  # Your concatenated input data

    audio_h,video_h,text_h = torch.randn(161798, 1024),torch.randn(161798, 314),torch.randn(161798, 367)

    output = model(audio_h,video_h,text_h)

    print(output)
