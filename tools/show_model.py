from model import *


def test_backbone():
    model = Backbone(50, 0.1, 'ir')
    print(model)
def test_backbone2():
    model = Backbone(50, 0.1, 'ir_se')
    print(model)
def test_mobilefacenet():
    model = MobileFaceNet(128)
    print(model)