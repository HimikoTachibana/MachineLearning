import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import pdb
import scipy.misc
import cv2
import numpy as np

NUM_EPOCH = 10

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True).cuda()
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules).cuda()
        # Create new layers
        self.backbone = nn.Sequential(*modules).cuda()
        self.fc1 = nn.Linear(2048, 32).cuda()
        self.dropout = nn.Dropout(p=0.5).cuda()
        self.fc2 = nn.Linear(32, 10).cuda()

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return nn.Softmax()(out)

def train():
    ## Define the training dataloader
    transform = transforms.Compose([transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
            shuffle=True, num_workers=2)

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
            lr=0.001, momentum=0.9)

    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    torch.save(model,'mymodel')
    print('Finished Training')


def render():
    model = torch.load('mymodel')

    # transformationos
    transform = transforms.Compose([transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5))])
    testset = datasets.CIFAR10('./data-test', download=True, train=False, transform=transform)
    # testloader
    testloader = torch.utils.data.DataLoader(testset)

    filename='what'
    
    filenamelist=[]
    html = ''
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        imgs = np.array(inputs)[0, :, :, :]
        imgs = np.transpose(imgs, [1, 2, 0])
        imgs = imgs * 0.5 + 0.5
        imgs = imgs * 255
        name=filename+str(i)+'.jpg'
        
        cv2.imwrite(name, imgs)
        filenamelist.append(name)
        html += '<img src=\"' + name  + '\">'
        html+= '<div>'
        annotations = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        outputs = outputs.tolist()
        for index in range(len(annotations)):
            html += annotations[index] + ': %.2f%% ' % (outputs[0][index] * 100)
        html += '</div>'
	language_setting='en'
	lang='<html lang="' + language_setting +'">'
        head_o='<head>'
	meta_c='<meta charset="' + 'utf-8' + '">'
        meta_d='<meta name="description'+'"' + 'content="The HTML5 Herald' + '"' + '>'
	
	title_o='<title>'
        title='Output HTML'
	title_c='<title>'
	head_c='</head>'

	body_o='<body>'
	body_c='</body>'
	html_c='</html>'
        total_html=lang+head_o+meta_c+title_o+title+title_c+meta_d+head_c+body_o+html+body_c+html_c

        return total_html
        #pdb.set_trace()
    #image_arr=[]
    #for i in range(outputs.shape[0]):
        #image_arr.append(outputs[i])

def FileString(string):
    with open('Output.html',"w") as html_file:
        print(string,file=html_file)


if __name__ == '__main__':
    train()
    html_string=render()
    FileString(html_string)
