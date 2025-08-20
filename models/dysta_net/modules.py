import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# class CustomActivation(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.w0 = nn.Parameter(torch.tensor(0.01))
#     self.w1 = nn.Parameter(torch.tensor(1.))
#     self.w2 = nn.Parameter(torch.tensor(1.))
#     self.w3 = nn.Parameter(torch.tensor(1.))
#     self.b = nn.Parameter(torch.tensor(0.))
#   def forward(self,x):
#     res = torch.where(x > 0,torch.log1p(torch.abs(self.w3*(x**3)+self.w2*(x**2)+self.w1*x+self.b)),(self.w0**2)*x)
#     return res

# class aulu(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.beta = nn.Parameter(torch.tensor(0.5),requires_grad=True)
#         self.alpha = nn.Parameter(torch.tensor(0.5),requires_grad=True)
#     def forward(self,x):
#         # rs = torch.where(x >= 0,x*nn.Sigmoid()(x*self.alpha),x*nn.Sigmoid()(x*(self.beta)))
#         rs = x*nn.Sigmoid()(x)
#         return rs


class Conv_func_1(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,padding='same',stride=1,dilation=1,norm=True,activation=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.norm=norm
        self.activation=activation
        if self.norm:
            self.norm_func=nn.GroupNorm(out_channel,out_channel,affine=False)
        # self.w = nn.Parameter(torch.rand(kernel*kernel*in_channel,out_channel),requires_grad=True)
        # nn.init.kaiming_normal_(self.w, mode='fan_out', nonlinearity='relu')
        # self.conv=nn.Conv2d(in_channel,out_channel,kernel,padding=padding,stride=stride,dilation=dilation)
        self.router = nn.Sequential(
            nn.Linear(3,128,bias=False),
            nn.RMSNorm(128,elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(128,1,bias=False),nn.Sigmoid())
        self.experts =  nn.ModuleList([nn.Sequential(nn.Linear(self.in_channel,self.out_channel,bias=False),nn.SiLU()) for _ in range(self.kernel*self.kernel)])
    def forward(self,x):
        b,c,h,w = x.shape
        if self.padding=='same' and isinstance(self.padding,str):
            pad_w=((x.shape[-1]-1)*self.stride+self.dilation*(self.kernel-1)+1-x.shape[-1])/2
            pad_h=((x.shape[-2]-1)*self.stride+self.dilation*(self.kernel-1)+1-x.shape[-2])/2
            pad_=(int(math.ceil(pad_h)),int(math.floor(pad_w)),int(math.ceil(pad_h)),int(math.floor(pad_w)))
        else:
            pad_=(self.padding,self.padding,self.padding,self.padding)


        x=torch.nn.functional.pad(x,pad_,value=0)
        # print(padded_x.shape)
        b,c,h_new,w_new = x.shape
        # print(padded_x.shape)
        if self.padding=='same':
            x = F.unfold(x,kernel_size=self.kernel,stride =self.stride,dilation=self.dilation).permute(0,2,1).reshape(b,h,w,c,self.kernel*self.kernel).permute(0,1,2,4,3)
        
        else:
           
            x = F.unfold(x,kernel_size=self.kernel,stride =self.stride,dilation=self.dilation).permute(0,2,1).reshape(b,int(((h_new-self.dilation*(self.kernel-1)-1)/self.stride)+1),int(((w_new-self.dilation*(self.kernel-1)-1)/self.stride)+1),c,self.kernel*self.kernel).permute(0,1,2,4,3)
        b,h,w,_,_ = x.shape
        router=self.router(torch.cat((x.max(-1).values.unsqueeze(-1),x.mean(-1).unsqueeze(-1),x.std(-1).unsqueeze(-1)),-1)).flatten(-2)
        # router>router.mean(-1).unsqueeze(-1)
        output = torch.zeros(b,h,w,self.out_channel).to(x.device)
        # if self.training:
        #     noise = torch.rand_like(router)
        #     router+=noise.to(router.device)
        
        # logits,indices = router.topk(4,-1)     
        # print(logits)
        # print(indices) 
        # inf_matrix = torch.where(router>router.mean(-1).unsqueeze(-1),router,-torch.inf)

        # router = F.sigmoid(router)
        # print(fill_zero_gate)
        
        # prob = nn.Softmax(-1)(self.router(fill_zero_gate.flatten(-2)))
        
        # # print(self.experts[0](prob[...,0].unsqueeze(-1)*tmp_x[...,0,:]))
        
        for i in range(self.kernel*self.kernel):
            output += self.experts[i](x[...,i,:])*router[...,i].unsqueeze(-1)
        output= output.permute(0,3,1,2).contiguous()
        if self.norm:
            output=self.norm_func(output)
        if self.activation:
            output=nn.ReLU()(output)
        return output

class Residual_net_1(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            Conv_func(in_channel,out_channel,kernel = kernel),
            Conv_func_1(out_channel,out_channel,kernel = kernel))
        self.change_feature = Conv_func_1(in_channel,out_channel,3,padding='same',activation=True)
    def forward(self,x):
        return self.change_feature(x)+self.feature(x)

class Conv_func(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,padding='same',bias=False),
            nn.GroupNorm(out_channel,out_channel,affine=False),
            nn.ReLU()
            )
    def forward(self,x):
        return self.feature(x)

class Residual_net(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            Conv_func(in_channel,out_channel,kernel = kernel),
            Conv_func(out_channel,out_channel,kernel = kernel)
            )
        self.change_feature = nn.Conv2d(in_channel,out_channel,1,bias=False)
    def forward(self,x):
        return self.change_feature(x)+self.feature(x)


class Unpooling_func(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.out = nn.ConvTranspose2d(in_channel,out_channel,kernel_size= scale_factor,stride = scale_factor,bias=False)
    def forward(self,x):
        return self.out(x)
    

class down_sampling(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.out =  nn.Sequential(
            Residual_net(in_channel,out_channel,3),
            # Residual_net(out_channel,out_channel,3)

            )
        self.down = Conv_func_1(out_channel,out_channel,kernel=2,padding=0,stride=2,activation=False)
        # self.down = nn.Conv2d(out_channel,out_channel,kernel_size=2,stride=2,bias=False)
    def forward(self,x):
        out = self.out(x)
        return out,self.down(out)
    

class Up_sampling(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.up = Unpooling_func(in_channel,out_channel,scale_factor=scale_factor)

        self.out =  Residual_net(out_channel*2,out_channel,3)
    def forward(self,x,x_encode):
        up = self.up(x)
        return self.out(torch.cat((up,x_encode),1))
    
class model_exchange_feature(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.rs1 = Residual_net(in_channel,in_channel,3)
        self.rs2 = Residual_net(in_channel,in_channel,3)

    def forward(self,x1,x2):
        x1_1,x1_2 = x1.chunk(2,1)
        x2_1,x2_2 = x2.chunk(2,1)
        new_x1 = torch.cat((x1_1,x2_2),1)
        new_x2 = torch.cat((x1_2,x2_1),1)
        return self.rs1(new_x1),self.rs2(new_x2)