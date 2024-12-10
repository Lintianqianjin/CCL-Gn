import torch
import torch.nn as nn
import torch.nn.functional as F

def js_loss(net_1_logits, net_2_logits):
    net_1_probs =  F.softmax(net_1_logits, dim=1)
    net_2_probs=  F.softmax(net_2_logits, dim=1)

    m = 0.5 * (net_1_probs + net_2_probs)

    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="none").sum(1)
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="none").sum(1)
    
    return (0.5 * loss)

if __name__ == '__main__':
    print('main')

    a = torch.randn(3,5)
    b = torch.randn(3,5)
    print(js_loss(a, b))
    print(js_loss(a, a))