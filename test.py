import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd

from losses import relu_evidence
from helpers import rotate_img, one_hot_embedding, get_device



def test_single_image(model, img_path, uncertainty=False, device=None):
    img = Image.open(img_path).convert("L")
    if not device:
        device = get_device()
    num_classes = 10
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)

    if uncertainty:
        output = model(img_variable)
        evidence = relu_evidence(output)
        alpha = evidence + 1
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        _, preds = torch.max(output, 1)
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)

    else:

        output = model(img_variable)
        _, preds = torch.max(output, 1)
        prob = F.softmax(output, dim=1)
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        print("Predict:", preds[0])
        print("Probs:", prob)

    labels = np.arange(10)
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})

    plt.title("Classified as: {}, Uncertainty: {}".format(preds[0], uncertainty.item()))

    axs[0].set_title("One")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")

    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")

    fig.tight_layout()

    plt.savefig("./results/MNIST/{}".format(os.path.basename(img_path)))


def rotating_image_classification(
    model, img, filename, uncertainty=False, threshold=0.5, device=None
):
    if not device:
        device = get_device()
    num_classes = 10
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    classifications = []

    scores = np.zeros((1, num_classes))
    rimgs = np.zeros((28, 28 * Ndeg))
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)

        nimg = np.clip(a=nimg, a_min=0, a_max=1)

        rimgs[:, i * 28 : (i + 1) * 28] = nimg
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)

        if uncertainty:
            output = model(img_variable)
            evidence = relu_evidence(output)
            alpha = evidence + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            _, preds = torch.max(output, 1)
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())
            lu.append(uncertainty.mean().cpu().detach())

        else:

            output = model(img_variable)
            _, preds = torch.max(output, 1)
            prob = F.softmax(output, dim=1)
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            classifications.append(preds[0].item())

        scores += prob.detach().cpu().numpy() >= threshold
        ldeg.append(deg)
        lp.append(prob.tolist())

    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})

    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])

    if uncertainty:
        labels += ["uncertainty"]
        axs[2].plot(ldeg, lu, marker="<", c="red")

    print(classifications)

    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")
    axs[0].axis("off")
    plt.pause(0.001)

    empty_lst = []
    empty_lst.append(classifications)
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")

    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")

    plt.savefig(filename)

def test_Ki67(model,dataloaders,uncertainty,phase='test',device=None):

    if not device:
        device = get_device()
    gd_list, pred_list=[],[]
    result_df=pd.DataFrame(columns=["name","gd","pred","ki67","match","uncertainty"])
    num_classes = 5
    running_corrects = 0.0
    with torch.no_grad():
        for i, (inputs, labels, sup) in enumerate(dataloaders[phase]):
            inputs = inputs.to(device)
            gd_list+=labels.clone().cpu().tolist()
            labels = labels.to(device)  # [batch_size]
            name=sup[0]
            ki67=sup[1]
            if uncertainty:
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                outputs = model(inputs)
                # print(outputs.shape)
                _, preds = torch.max(outputs, 1)    # [batch_size]
                pred_list += preds.clone().cpu().tolist()


                match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                acc = torch.mean(match)
                evidence = relu_evidence(outputs)
                alpha = evidence + 1
                u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                result_df=pd.concat([result_df,pd.DataFrame({
                    "name": name,
                    "ki67": ki67,
                    "pred": preds.clone().cpu().tolist(),
                    "gd": labels.clone().cpu().tolist(), 
                    "match": match[:,0].clone().cpu().tolist(),
                    "uncertainty": u[:,0].clone().cpu().tolist()})])

                total_evidence = torch.sum(evidence, 1, keepdim=True)
                mean_evidence = torch.mean(total_evidence)
                mean_evidence_succ = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * match
                ) / torch.sum(match + 1e-20)
                mean_evidence_fail = torch.sum(
                    torch.sum(evidence, 1, keepdim=True) * (1 - match)
                ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

            else:
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
    
    epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
    print("Accuracy: ",epoch_acc)
    result_df.to_csv("./results/Ki67/{}_result.csv".format(phase),index=False)
    # conf_matrix = confusion_matrix(result_df["gd"],result_df["pred"])
    conf_matrix = pd.crosstab(gd_list,pred_list)
    print("Confusion Matrix:")
    print(conf_matrix)
    