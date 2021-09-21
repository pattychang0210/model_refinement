import sys
# sys.path.append('../../')
from backbone.iresnet import iresnet50

class BackboneFactory:
    def __init__(self, backbone_type):
        self.backbone_type = backbone_type
        
    def get_backbone(self):
        if self.backbone_type == 'IResNet':
            model = iresnet50()
            return model
        else:
            raise ValueError()