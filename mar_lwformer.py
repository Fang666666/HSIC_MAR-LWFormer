import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# 等于 PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Pooling(nn.Module):

    def __init__(self, dim, dropout=0.1, pool_size=5,  **kwargs):
        super().__init__()



        self.pool1 = nn.AvgPool2d(
            kernel_size=pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        self.pool2 = nn.AvgPool2d(
            kernel_size=pool_size+2, stride=1, padding=7 // 2, count_include_pad=False)
        self.pool3 = nn.AvgPool2d(
            kernel_size=pool_size+4, stride=1, padding=9 // 2, count_include_pad=False)
        self.nn1 = nn.Linear(dim, dim)
        # self.do1 = nn.Dropout(dropout)



    def forward(self, x, mask=None):

        y1 = self.pool1(x) - x
        y2 = self.pool2(x) - x
        y3 = self.pool3(x) - x

        out = y1 + y2 + y3

        out = self.nn1(out)
        # out = self.do1(out)

        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Pooling(dim))),
                # Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for pooling, mlp in self.layers:
            x = pooling(x)
            # x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.crossAttention = LayerNormalize(dim, CrossAttention(dim, heads=heads, dropout=dropout))


    def forward(self, x, mask=None):
        input_origin_emap_lbp = torch.chunk(x, chunks=3, dim=1)
        x_origin = input_origin_emap_lbp[0]
        x_emap = input_origin_emap_lbp[1]
        x_lbp = input_origin_emap_lbp[2]

        cross_att_o_e = torch.cat((x_origin, x_emap), dim=0)
        cross_att_o_l = torch.cat((x_origin, x_lbp), dim=0)

        cross_att_o_e = self.crossAttention(cross_att_o_e, mask=mask)

        cross_att_o_l = self.crossAttention(cross_att_o_l, mask=mask)

        cross_att_e_l = torch.add(cross_att_o_e, cross_att_o_l)

        return cross_att_e_l


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_q = nn.Linear(dim, dim, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads

        x12 = torch.chunk(x, chunks=2, dim=0)
        x1 = x12[0]
        x2 = x12[1]

        q = self.to_q(x1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        k = self.to_k(x2)
        v = self.to_v(x2)
        qkv = []
        qkv.append(q)
        qkv.append(k)
        qkv.append(v)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out








BATCH_SIZE_TRAIN = 64
NUM_CLASS = 16



class ExplicitFeatureAttention(nn.Module):
    def __init__(self, kernel_size=7, num_classes=NUM_CLASS, num_tokens=6, dim=64, depth=1, heads=4, mlp_dim=64, emb_dropout=0.4):
        super(ExplicitFeatureAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv2d = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.BN = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Tokenization
        self.L = num_tokens
        self.cT = dim
        self.token_wA = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, self.L, 64),
                                     requires_grad=True)  # Tokenization parameters
        # torch.nn.init.xavier_uniform_(self.token_wA)
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(BATCH_SIZE_TRAIN, 64, self.cT),
                                     requires_grad=True)  # Tokenization parameters
        # torch.nn.init.xavier_uniform_(self.token_wV)
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        # self.pos_embedding = nn.Parameter(torch.empty(1, 26, dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)  # initialized based on the paper

        # self.patch_conv= nn.Conv2d(64,dim, self.patch_size, stride = self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))  # initialized based on the paper
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x):
        att = x
        avg_out = torch.mean(att, dim=1, keepdim=True)
        max_out, _ = torch.max(att, dim=1, keepdim=True)

        avg_out = self.conv2d(avg_out)
        max_out = self.conv2d(max_out)
        att = torch.add(avg_out, max_out)

        att = self.conv2d(att)
        x = x * self.sigmoid(att)

        x = rearrange(x, 'b c h w -> b (h w) c')
        wa = rearrange(self.token_wA, 'b h w -> b w h')
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')
        A = A.softmax(dim=-1)

        T = torch.einsum('bij,bjk->bik', A, x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        return x

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.basicconv = BasicConv2d(in_planes=2, out_planes=1, kernel_size=kernel_size, padding=(kernel_size-1) // 2,
                                     dilation=1, groups=1, relu=False, bn=True)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.basicconv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class CSE_Attention(nn.Module):
    def __init__(self):
        super(CSE_Attention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0,
        dilation=1, groups=1, relu=True, bn=True,bias=False):
        super(BasicConv3d, self).__init__()
        self.out_channels = out_planes
        self.basicconv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-5, momentum=0.01) \
            if bn else None

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.basicconv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, relu=True, bn=True,bias=False):
        super(BasicConv2d, self).__init__()
        self.out_channels = out_planes
        self.basicconv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) \
            if bn else None

        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.basicconv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Mar_LWFormer(nn.Module):
    def __init__(self, in_channels=1, num_classes=NUM_CLASS, num_tokens=6, dim=64, depth=1, heads=4, mlp_dim=64,
                 dropout=0.4):
        super(Mar_LWFormer, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=39, kernel_size=(3, 3, 3), stride=(1, 1, 1)),
            nn.BatchNorm3d(39),
            nn.ReLU(),
        )
        self.conv3d_features = BasicConv3d(in_planes=in_channels, out_planes=39, kernel_size=(3, 3, 3), stride=(1, 1, 1), relu=True, bn=True)
        self.conv3d_features_1 = BasicConv3d(in_planes=in_channels, out_planes=14, kernel_size=(3, 3, 3), stride=(1, 1, 1), relu=True, bn=True)
        self.conv3d_features_2 = BasicConv3d(in_planes=in_channels, out_planes=14, kernel_size=(5, 3, 3), stride=(1, 1, 1), relu=True, bn=True)
        self.conv3d_features_3 = BasicConv3d(in_planes=in_channels, out_planes=14, kernel_size=(7, 3, 3), stride=(1, 1, 1), relu=True, bn=True)
        self.conv3d_features_cat = BasicConv3d(in_planes=14, out_planes=14, kernel_size=(1, 1, 1), stride=(1, 1, 1), relu=True, bn=True)
        self.conv3d_features_cat2 = BasicConv3d(in_planes=39, out_planes=8, kernel_size=(1, 1, 1), stride=(1, 1, 1), relu=True, bn=True)

        ######################################################

        self.conv2d_features_1_emap = BasicConv2d(in_planes=1, out_planes=8, kernel_size=(3, 3), relu=True, bn=True)
        self.conv2d_features_2_emap = BasicConv2d(in_planes=8, out_planes=64, kernel_size=(3, 3), relu=True, bn=True)

        ######################################################

        self.conv2d_features_1_lbp = BasicConv2d(in_planes=1, out_planes=8, kernel_size=(3, 3), relu=True, bn=True)
        self.conv2d_features_2_lbp = BasicConv2d(in_planes=8, out_planes=64, kernel_size=(3, 3), relu=True, bn=True)
        self.conv2d_features = BasicConv2d(in_planes=224, out_planes=64, kernel_size=(3, 3), relu=True, bn=True)


        self.CFAF = CrossAttentionBlock(dim, heads=heads, dropout=dropout)

        self.CSE_A = CSE_Attention()

        self.EFA_Tokenizer = ExplicitFeatureAttention(kernel_size=3, num_tokens=6, dim=64, depth=1, heads=4, mlp_dim=64, emb_dropout=0.4)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None, patch_size=13):


        input_origin_emap_lbp = torch.chunk(x, chunks=3, dim=1)
        x_origin = input_origin_emap_lbp[0]
        x_emap = input_origin_emap_lbp[1]
        x_lbp = input_origin_emap_lbp[2]
        x_emap = x_emap[:, :, 1, :, :]
        x_lbp = x_lbp[:, :, 1, :, :]

        x1 = self.conv3d_features(x_origin)
        x1 = torch.reshape(x1, (64, -1, patch_size-2, patch_size-2))      ###################################### patch=>size  9=>7, 11=>9, 13=>11
        branch_1 = self.conv3d_features_1(x_origin)
        branch_2 = self.conv3d_features_2(x_origin)
        branch_3 = self.conv3d_features_3(x_origin)

        cat1 = torch.cat((branch_1, branch_2, branch_3), dim=2)
        cat2 = self.conv3d_features_cat(cat1)
        cat2 = torch.reshape(cat2, (64, -1, patch_size-2, patch_size-2))    ###################################### patch=>size  9=>7, 11=>9, 13=>11


        cat = torch.add(x1, cat2)
        # cat = self.tripletAttention(cat)
        cat = torch.reshape(cat, (64, -1, 28, patch_size-2, patch_size-2))  ###################################### patch=>size  9=>7, 11=>9, 13=>11
        cat = self.conv3d_features_cat2(cat)
        x_origin = cat



        ###############################################################################################
        x_emap0 = self.conv2d_features_1_emap(x_emap)
        cat_emap = self.conv2d_features_2_emap(x_emap0)

        ###############################################################################################

        x_lbp0 = self.conv2d_features_1_lbp(x_lbp)
        cat_lbp = self.conv2d_features_2_lbp(x_lbp0)


        x_origin = x_origin.view(x_origin.size()[0], x_origin.size()[1] * x_origin.size()[2], x_origin.size()[3], x_origin.size()[4])
        x = self.conv2d_features(x_origin)

        x_origin = rearrange(x, 'b c h w -> b (h w) c')
        cat_emap = rearrange(cat_emap, 'b c h w -> b (h w) c')
        cat_lbp = rearrange(cat_lbp, 'b c h w -> b (h w) c')

        cross_att_x = torch.cat((x_origin, cat_emap, cat_lbp), dim=1)

        cross_att = self.CFAF(cross_att_x)
        cross_att = rearrange(cross_att, 'b (h w) c -> b c h w', h=patch_size-4, w=patch_size-4)  ###################################### patch=>size  9=>5, 11=>7, 13=>9
        x = torch.add(x, cross_att)


        x = self.CSE_A(x)
        x = self.EFA_Tokenizer(x)
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])

        x = self.nn1(x)

        return x


if __name__ == '__main__':
    model = Mar_LWFormer()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())
