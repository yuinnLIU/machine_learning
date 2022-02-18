class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layer):
        super(FeatureExtractor,self).__init__()
        self.submodule=submodule
        self.extracted_layer=extracted_layer
    def forward(self, x):
        outputs=[]
        for name,module in self.submodule._modules.items():
            x=module(x)
            if name in self.extracted_layer:
                outputs.append(x)
        return outputs

extracted_layer=['0','2','3','5']

def Feature_visual(outputs):
    for i in range(len(outputs)):
        out=outputs[i].data.squeeze().numpy()
        feature_img = out[0, :, :].squeeze()   #选择第一个特征图进行可视化
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        plt.imshow(feature_img,cmap='gray')
        plt.show()