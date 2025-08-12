import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# class Conv_func(nn.Module):
#     def __init__(self,in_channel,out_channel,kernel=3):
#         super().__init__()
#         self.feature = nn.Sequential(
#             nn.Conv2d(in_channel,out_channel,kernel_size=kernel,padding='same',bias=False),
#             nn.GroupNorm(out_channel,out_channel,affine=False),
#             nn.ReLU(),
#             )
#     def forward(self,x):
#         return self.feature(x)
class Conv_func(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,padding='same',stride=1,dilation=1):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.dilation = dilation 
        # self.w = nn.Parameter(torch.rand(kernel*kernel*in_channel,out_channel),requires_grad=True)
        # nn.init.kaiming_normal_(self.w, mode='fan_out', nonlinearity='relu')

        self.conv=nn.Conv2d(in_channel,out_channel,kernel,padding=padding,stride=stride,dilation=dilation)
        self.router = nn.Linear(self.kernel*self.kernel*self.in_channel,self.kernel*self.kernel,bias=False)
        self.experts =  nn.ModuleList([nn.Linear(self.in_channel,self.out_channel,bias=False) for _ in range(self.kernel*self.kernel)])
        self.norm=nn.GroupNorm(out_channel,out_channel,affine=False)

    def forward(self,x):
        b,c,h,w = x.shape
        if self.padding=='same' and isinstance(self.padding,str):
            pad_w=((x.shape[-1]-1)*self.stride+self.dilation*(self.kernel-1)+1-x.shape[-1])/2
            pad_h=((x.shape[-2]-1)*self.stride+self.dilation*(self.kernel-1)+1-x.shape[-2])/2
            pad_=(int(math.ceil(pad_h)),int(math.floor(pad_w)),int(math.ceil(pad_h)),int(math.floor(pad_w)))
        else:
            pad_=(self.padding,self.padding,self.padding,self.padding)


        padded_x=torch.nn.functional.pad(x,pad_,value=0)
        # print(padded_x.shape)
        b,c,h_new,w_new = padded_x.shape
        # print(padded_x.shape)
        if self.padding=='same':
            tmp_x = F.unfold(padded_x,kernel_size=self.kernel,stride =self.stride,dilation=self.dilation).permute(0,2,1).reshape(b,h,w,c,self.kernel*self.kernel).permute(0,1,2,4,3)
        
        else:
           
            tmp_x = F.unfold(padded_x,kernel_size=self.kernel,stride =self.stride,dilation=self.dilation).permute(0,2,1).reshape(b,int(((h_new-self.dilation*(self.kernel-1)-1)/self.stride)+1),int(((w_new-self.dilation*(self.kernel-1)-1)/self.stride)+1),c,self.kernel*self.kernel).permute(0,1,2,4,3)
        b,h,w,_,_ = tmp_x.shape
        router=self.router(tmp_x.flatten(-2))
        # router>router.mean(-1).unsqueeze(-1)
        output = torch.zeros(b,h,w,self.out_channel).to(x.device)
        if self.training:
            noise = torch.rand_like(router)
            router+=noise.to(router.device)
        
        logits,indices = router.topk(4,-1)     
        # print(logits)
        # print(indices) 
        inf_matrix = torch.where(router>router.mean(-1).unsqueeze(-1),router,-torch.inf)

        fill_zero_gate = F.softmax(inf_matrix/torch.sqrt(torch.tensor(inf_matrix.shape[-1]).float()),-1)
        # print(fill_zero_gate)
        
        # prob = nn.Softmax(-1)(self.router(fill_zero_gate.flatten(-2)))
        
        # # print(self.experts[0](prob[...,0].unsqueeze(-1)*tmp_x[...,0,:]))
        
        for i in range(self.kernel*self.kernel):
            output += self.experts[i](tmp_x[...,i,:])*fill_zero_gate[...,i].unsqueeze(-1)
        return self.norm(nn.ReLU()(output.permute(0,3,1,2).contiguous()+self.conv(x)))
class Residual_net(nn.Module):
    def __init__(self,in_channel,out_channel,kernel=3):
        super().__init__()
        self.feature = nn.Sequential(
            Conv_func(in_channel,out_channel,kernel = kernel),
            Conv_func(out_channel,out_channel,kernel = kernel))
        self.change_feature = nn.Conv2d(in_channel,out_channel,3,padding='same',bias=False)
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
        self.out =  Residual_net(in_channel,out_channel,3)
        self.down = nn.MaxPool2d(2)
    def forward(self,x):
        out = self.out(x)
        return out,self.down(out)
    

class Up_sampling(nn.Module):
    def __init__(self,in_channel,out_channel,scale_factor = 2):
        super().__init__()
        self.up = Unpooling_func(in_channel,out_channel,scale_factor=scale_factor)

        self.out = Residual_net(out_channel*2,out_channel,3)
    def forward(self,x,x_encode):
        up = self.up(x)
        return self.out(torch.cat((up,x_encode),1))