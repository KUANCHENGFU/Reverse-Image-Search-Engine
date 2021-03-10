from deep_learning.DL_function import *
import pandas as pd
import numpy as np
from milvus import Milvus, IndexType, MetricType, Status
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('Agg')
import os
import sys
matplotlib.use('Agg')
import io
import base64

def uploaded_image():

    folder = 'uploaded_image/image'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.suptitle('Uploaded Image:', fontsize=20)
    img = mpimg.imread(file_path)
    ax.imshow(img)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    save_file = io.BytesIO()
    fig.savefig(save_file, format='png')
    figdata_png = base64.b64encode(save_file.getvalue()).decode('utf8')

    return figdata_png

def related_images():

    # Read the data of vectors and their information
    df_vectors = pd.read_csv('deep_learning/vectors.csv', index_col=0)
    df_info = pd.read_csv('deep_learning/info.csv', index_col=0)
    normalized_vectors = preprocessing.normalize(df_vectors.to_numpy())


    # Connect to the Milvus server
    client = Milvus(host='localhost', port='19530')
    # Create a collection
    param = {'collection_name':'test01', 'dimension':120, 'index_file_size':1024, 'metric_type':MetricType.IP}
    status = client.create_collection(param)   
    # Create an index
    ivf_param = {'nlist': 20580}
    status = client.create_index('test01', IndexType.IVF_FLAT, ivf_param)
    # Insert vectors in the collection
    status, inserted_vector_ids = client.insert(collection_name='test01', records=normalized_vectors)
    df_info['new_id'] = inserted_vector_ids


    # Get the class names
    batch_size = 128
    threads = 0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
                                   transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(torch.Tensor(mean),
                                                        torch.Tensor(std))])
    images_path1 = 'deep_learning/input/stanford-dogs-dataset/images/Images/'
    dataset = datasets.ImageFolder(root=images_path1, transform=transform)
    class_names = dataset.classes
    class_names = [classes[10:] for classes in class_names]
    # Read the uploaded image
    images_path2 = 'uploaded_image/'
    dataset = datasets.ImageFolder(root=images_path2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=threads, shuffle=False)      
    # Convert the uploaded image into a vector
    # net_model = resnet.resnet152(pretrained=True)
    # net_name = 'resnet152'
    # unfrozen_layers = ['layer4', 'fc'] 
    # dropout_ratio = 0.9
    # net_model = load_network(net_model, net_name, dropout_ratio, class_names, unfrozen_layers)   
    # prediction, true = test_model(net_name, net_model, dataloader)
    # normalized_vector = preprocessing.normalize(prediction)
    net_model = resnet.resnet152(pretrained=True)
    net_name = 'resnet152'
    unfrozen_layers = ['layer4', 'fc'] 
    for name, child in net_model.named_children():
        if name in unfrozen_layers:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False
    num_ftrs = net_model.fc.in_features
    net_model.fc = nn.Sequential(nn.Linear(num_ftrs, 256),
                                nn.ReLU(),
                                nn.Dropout(p=1),
                                nn.Linear(256, len(class_names)))
    model_path = 'deep_learning/input/model/resnet152.pth'
    net_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
    prediction, true = test_model(net_name, net_model, dataloader)
    normalized_vector = preprocessing.normalize(prediction)

    # Search its related images
    search_param = {'nprobe': 2000}
    status, results = client.search(collection_name='test01', query_records=normalized_vector, top_k=10, params=search_param)
    id_list = []
    for row in results:
        for item in row:
            id_list.append(item.id)
    results_info = df_info[df_info['new_id'].isin(id_list)]
    # Close client
    status = client.drop_collection(collection_name='test01')
    client.close()

    # Remove the uploaded image
    folder = 'uploaded_image/image'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    # Display its related images
    # print(results_info)
    image_names = results_info['id'].tolist()
    class_name = class_names[results_info['true_class'].value_counts().idxmax()].title().replace('_', ' ')
    if class_name[0].lower() in ['a', 'e', 'i', 'o', 'u']:
        title = 'Search Results: \n We guess your dog is an ' + class_name + '!'
    else:
        title = 'Search Results: \n We guess your dog is a ' + class_name + '!'
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(title, fontsize=20)
    for i in range(2):
        for j in range(5):
            img = mpimg.imread('all_images/'+image_names[5*i+j])
            ax[i, j].imshow(img)
            ax[i, j].axes.xaxis.set_visible(False)
            ax[i, j].axes.yaxis.set_visible(False)

    save_file = io.BytesIO()
    fig.savefig(save_file, format='png')
    figdata_png = base64.b64encode(save_file.getvalue()).decode('utf8')

    return figdata_png 


