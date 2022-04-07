import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from evaluate import evaluate_classifier

# Beware. Code highly repetitive
# People allergic to repetitive code are not safe here
# This code brings memory of my grandma. Repetitive
# For real tho, one day I should abstract these training methods
def training_stats(history):
  plt.plot(
    history['accuracy'],
  )
  plt.plot(
    history['test_accuracy'],
  )
  plt.legend(['train accuracy', 'test_accuracy'])
  plt.show()
  plt.plot(
    history['loss'],
  )
  plt.legend(['loss'])
  plt.show()
  
def train_classifier(model, dataloader, testloader, epochs=3, lr=5e-3, momentum=0.7, debug=False, criterion = nn.CrossEntropyLoss()):
  history = { 'loss': [], 'iloss': [], 'accuracy': [], 'test_accuracy': []}
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  N = dataloader.dataset.data.shape[0]
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = .0
    running_acc = .0
    total_loss = .0
    total_acc = .0
    for i, (inputs, labels) in enumerate(dataloader, 0):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      acc = torch.sum(torch.argmax(outputs, -1) == labels)
      total_loss += loss.item()
      total_acc += acc.item()
      if debug:
        running_loss += loss.item()
        running_acc += acc.item()
        infostep = 500
        if i % infostep == infostep-1:    # print every `infostep` mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}/{N//dataloader.batch_size}] loss: {running_loss / infostep:.3f} acc: {running_acc / (infostep*dataloader.batch_size):.3f}')
          running_loss = .0
          running_acc = .0
    if testloader:
        testhistory = evaluate_classifier(model, testloader)
        history['test_accuracy'].append(testhistory['accuracy'])
    history['accuracy'].append(total_acc / N)
    history['loss'].append(total_loss/ N)
  return history

def train_auto_encoder(model, dataloader, criterion, epochs=3, lr=5e-3, momentum=0.7, debug=False):
  history = { 'loss': [], 'iloss': [] }
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  N = dataloader.dataset.data.shape[0]
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = .0
    total_loss = .0
    for i, (inputs, _) in enumerate(dataloader, 0):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      loss = criterion(outputs, inputs)
      loss.backward()
      optimizer.step()

      # print statistics
      total_loss += loss.item()
      history['iloss'].append(loss.item())
      if debug:
        running_loss += loss.item()
        infostep = 500
        if i % infostep == infostep-1:    # print every `infostep` mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}/{N//dataloader.batch_size}] loss: {running_loss / infostep:.3f}')
          running_loss = .0
    history['loss'].append(total_loss/ N)
  return history

# note: xtrue is the input of the network, x is the autoencoded. x and y are outputs
def composed_loss(x, y, xtrue, ytrue, class_loss=nn.CrossEntropyLoss(), self_loss=nn.MSELoss()):
    classl = class_loss(y, ytrue)
    selfl = self_loss(x, xtrue)
    return classl + selfl
    
def train_autoencoder_classifier(model, dataloader, testloader, epochs=3, lr=5e-3, momentum=0.7, debug=False, criterion = nn.CrossEntropyLoss()):
  history = { 'loss': [], 'iloss': [], 'accuracy': [], 'test_accuracy': []}
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  N = dataloader.dataset.data.shape[0]
  for epoch in range(epochs):  # loop over the dataset multiple times
    running_loss = .0
    running_acc = .0
    total_loss = .0
    total_acc = .0
    for i, (inputs, labels) in enumerate(dataloader, 0):
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      autoencoded, outputs = model(inputs)
      loss = composed_loss(autoencoded, outputs, inputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      acc = torch.sum(torch.argmax(outputs, -1) == labels)
      total_loss += loss.item()
      total_acc += acc.item()
      if debug:
        running_loss += loss.item()
        running_acc += acc.item()
        infostep = 500
        if i % infostep == infostep-1:    # print every `infostep` mini-batches
          print(f'[{epoch + 1}, {i + 1:5d}/{N//dataloader.batch_size}] loss: {running_loss / infostep:.3f} acc: {running_acc / (infostep*dataloader.batch_size):.3f}')
          running_loss = .0
          running_acc = .0
    # if testloader:
    #     testhistory = evaluate_classifier(model, testloader)
    #     history['test_accuracy'].append(testhistory['accuracy'])
    history['accuracy'].append(total_acc / N)
    history['loss'].append(total_loss/ N)
  return history
