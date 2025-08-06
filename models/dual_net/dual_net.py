from modules import *
class SegModel(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.change_channels = nn.Conv2d(in_channels,32,1,bias=False)
        self.down_0=nn.ModuleList([
            DownSampling(32,32),nn.Conv2d(32,32,3,2,1)])
        self.down_1=nn.ModuleList([
            DownSampling(32,64),nn.Conv2d(64,64,3,2,1)])
        self.down_2=nn.ModuleList([
            DownSampling(64,128),nn.Conv2d(128,128,3,2,1)])
        self.down_3=nn.ModuleList([
            DownSampling(128,256),nn.Conv2d(256,256,3,2,1)])
        
        self.bneck = DownSampling(256,512)

        self.up_0=UpSampling(512,256)
        self.up_1=UpSampling(256,128)
        self.up_2=UpSampling(128,64)
        self.up_3=UpSampling(64,32)

        self.out = nn.Sequential(
            nn.Conv2d(32,out_channels,1,bias=False),nn.Sigmoid())
    def forward(self,x):
        x=self.change_channels(x)

        pre_down_0=self.down_0[0](x)
        down_0=self.down_0[1](pre_down_0)

        pre_down_1=self.down_1[0](down_0)
        down_1=self.down_1[1](pre_down_1)

        pre_down_2=self.down_2[0](down_1)
        down_2=self.down_2[1](pre_down_2)

        pre_down_3=self.down_3[0](down_2)
        down_3=self.down_3[1](pre_down_3)
        bneck=self.bneck(down_3)

        up_0=self.up_0(bneck,pre_down_3)
        up_1=self.up_1(up_0,pre_down_2)
        up_2=self.up_2(up_1,pre_down_1)
        up_3=self.up_3(up_2,pre_down_0)
        return self.out(up_3)
