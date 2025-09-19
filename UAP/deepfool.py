import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
# from torch.autograd.gradcheck import zero_gradients


# def deepfool(image, net, num_classes = 10, overshoot = 0.02, max_iter = 50):
#     """
#     :param image: Image of size 3*H*W
#     :param net: network (input: images, output: values of activation **BEFORE** softmax).
#     :param num_class:
#     :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#     :param max_iter:
#     :return:minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
#     """
#     # net.grad.zero_()
#     f_image = net(Variable(image, requires_grad = True)).data.cpu().numpy().flatten()
#     I = f_image.argsort()[::-1] # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
#     I = I[0:num_classes] # 挑最大的num_classes个(从0开始，到num_classes结束)
#     label = I[0] # 最大的判断的分类

#     input_shape = image.detach().cpu().numpy().shape # 原始照片
#     pert_image = copy.deepcopy(image) # 干扰照片
#     w = np.zeros(input_shape)
#     r_tot = np.zeros(input_shape)

#     loop_i = 0

#     x = Variable(pert_image, requires_grad = True)
#     # net.zero_grad()
#     fs = net(x)
#     k_i = label

#     while k_i == label and loop_i < max_iter: # 直到分错类别或者达到循环上限次数
#         pert = np.inf
#         fs[0,I[0]].backward(retain_graph=True) # x产生了grad
#         grad_orig = x.grad.data.cpu().numpy().copy() # original grad
#         for k in range(1, num_classes):
#             zero_gradients(x) # 将梯度置为0
#             fs[0,I[k]].backward(retain_graph=True)
#             cur_grad = x.grad.data.cpu().numpy().copy() # current grad(分类为k(不是目前所划分的那一类)的grad

#             # set new w_k and new f_k
#             w_k = cur_grad - grad_orig
#             f_k = (fs[0,I[k]] - fs[0,I[0]]).data.cpu().numpy()

#             pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

#             # determine which w_k to use
#             if pert_k < pert: # 要找到最小的pert
#                 pert = pert_k
#                 w = w_k


#         # compute r_i and r_tot
#         r_i = (pert + 1e-4) * w / np.linalg.norm(w) # 这一次迭代的r
#         r_tot = np.float32(r_tot + r_i) #r_total
#         pert_image = image + (1+overshoot) * torch.from_numpy(r_tot).cuda()

#         x = Variable(pert_image, requires_grad = True) # 将添加干扰后的图片输入net里面
#         fs = net(x)
#         k_i = np.argmax(fs.data.cpu().numpy().flatten()) # 选择最大的一个作为新的分类
#         loop_i += 1

#     r_tot = (1+overshoot)*r_tot

#     return r_tot, loop_i


def deepfool_nlp(input_embeddings, attention_mask, model, num_classes=2, max_iter=20, overshoot=0.02):
    """
    DeepFool attack in embedding space, for a general transformer-based language model.

    Args:
        input_embeddings: torch.Tensor, shape [batch=1, seq_len, embed_dim]
        attention_mask: torch.Tensor, shape [1, seq_len]
        model: a transformer-based classifier, must accept inputs_embeds + attention_mask
        num_classes: number of target classes
        max_iter: max DeepFool iterations
        overshoot: overshoot parameter for DeepFool

    Returns:
        perturbed_embeddings: tensor same shape as input_embeddings
        num_iterations: number of iterations used
    """
    model.eval()
    device = input_embeddings.device

    pert_embeddings = input_embeddings.clone().detach().requires_grad_(True)

    with torch.no_grad():
        logits = model(inputs_embeds=pert_embeddings, attention_mask=attention_mask).logits
    label = torch.argmax(logits).item()

    r_tot = torch.zeros_like(pert_embeddings).to(device)
    loop_i = 0

    while loop_i < max_iter:
        logits = model(inputs_embeds=pert_embeddings, attention_mask=attention_mask).logits
        pred = torch.argmax(logits).item()

        if pred != label:
            break  # Attack success

        logits[0, label].backward(retain_graph=True)
        grad_orig = pert_embeddings.grad.detach().clone()

        min_perturbation = float('inf')
        w = None

        for k in range(num_classes):
            if k == label:
                continue

            pert_embeddings.grad.zero_()
            logits[0, k].backward(retain_graph=True)
            grad_k = pert_embeddings.grad.detach().clone()

            w_k = grad_k - grad_orig
            f_k = logits[0, k] - logits[0, label]

            pert_k = torch.abs(f_k) / (torch.norm(w_k) + 1e-8)
            if pert_k < min_perturbation:
                min_perturbation = pert_k
                w = w_k

        r_i = (min_perturbation + 1e-4) * w / torch.norm(w)
        r_tot = r_tot + r_i
        pert_embeddings = (input_embeddings + (1 + overshoot) * r_tot).detach().requires_grad_(True)

        loop_i += 1

    return pert_embeddings.detach(), loop_i
